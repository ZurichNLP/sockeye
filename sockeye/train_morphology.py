import argparse
import sys
import time
import logging
import os
import mxnet as mx
from contextlib import ExitStack
from typing import Generator, Optional, List, Tuple, Union, NamedTuple
import sacremoses
import codecs
from subword_nmt.apply_bpe import BPE
from sklearn.linear_model import LogisticRegression
from collections import OrderedDict
import numpy as np

import conllu
import io
from . import vocab
from . import data_io
from . import inference
from . import arguments
from . import translate
from . import model
from . import constants as C
from . import source_model
from . import utils

logger = logging.getLogger(__name__)

Tokens = List[str]
SentenceId = Union[int, str]

class Input:
    __slots__ = ('sentence_id', 'tokens')

    def __init__(self,
                 sentence_id: SentenceId,
                 tokens: Tokens) -> None:
        self.sentence_id = sentence_id
        self.tokens = tokens

    def __str__(self):
        return 'Input(%s, %s, )' \
            % (self.sentence_id, self.tokens)

    def __len__(self):
        return len(self.tokens)


    def chunks(self, chunk_size: int) -> Generator['Input', None, None]:
        for chunk_id, i in enumerate(range(0, len(self), chunk_size)):
            yield Input(sentence_id=self.sentence_id,
                                  tokens=self.tokens[i:i + chunk_size])
            
    def with_eos(self) -> 'Input':
        """
        :return: A new translator input with EOS appended to the tokens and factors.
        """
        return Input(sentence_id=self.sentence_id,
                     tokens=self.tokens + [C.EOS_SYMBOL])
    
class ClassifierInput:
    __slots__ = ( 'form',
                  'bpe_list',
                  'encoder_states',
                  'units',
                  'features'
                  )
    
    def __init__(self,
                 bpe_list: List[str],
                 encoder_states: List[np.array],
                 form: str,
                 units: int,
                 features: dict) -> None:
        self.bpe_list = bpe_list
        self.encoder_states = encoder_states
        self.form = form
        self.units = units
        self.features = features
    
    def __str__(self):
        return 'ClassifierInput(%s, %s, %i, %s)' \
            % (self.form, self.bpe_list, self.units, self.features)
    
    
    #def __init__(self,
                 #bpe_list: List[str],
                 #encoder_states: List[np.array],
                 #form: str,
                 #units: int,
                 #gender: str = None,
                 #number: str = None,
                 #case: str = None,
                 #definiteness: str = None,
                 #mood: str = None,
                 #finiteness: str = None,
                 #tense: str = None,
                 #pos: str = None) -> None:
        #self.bpe_list = bpe_list
        #self.encoder_states = encoder_states
        #self.form = form
        #self.units = units
        #self.gender = gender
        #self.number = number
        #self.case = case
        #self.definiteness = definiteness
        #self.tense = tense
        #self.pos = pos
        
    #def setGender(self, gender):
        #self.gender = gender
    
    #def setNumber(self, number):
        #self.number = number
        
    #def setCase(self, case):
        #self.case = case
        
    #def setDefiniteness(self, definiteness):
        #self.definiteness = definiteness
        
    #def setMood(self, mood):
        #self.mood = mood
        
    #def setFiniteness(self, finiteness):
        #self.finiteness = finiteness
        
    #def setTense(self, tense):
        #self.tense = tense
        
    #def setPos(self, pos):
        #self.pos = pos
    

IndexedInput = NamedTuple('IndexedInput', [
    ('input_idx', int),
    ('chunk_idx', int),
    ('input', Input)
])

# complete list of Unimorph tags: https://unimorph.github.io/doc/unimorph-schema.pdf
morph_features = {
    "ADJ" : "pos",
    "ADP": "pos",
    "ADV": "pos",
    "CONJ": "pos",
    "DET": "pos",
    "N": "pos",
    "NUM": "pos",
    "PART": "pos",
    "PRO": "pos",
    "PROPN": "pos",
    "V": "pos",
    "V.PTCP": "pos",
    "X": "pos",
    "FEM" : "gender",
    "MASC" : "gender",
    "NEUT" : "gender",
    "SG" : "number",
    "PL" : "number",
    "DEF" : "definiteness",
    "INDF" : "definiteness",
    "SBJV" : "mood",
    "IND" : "mood",
    "IMP" : "mood",
    "FIN" : "finiteness",
    "NFIN" : "finiteness",
    "PST" : "tense",
    "PRS" : "tense",
    "FUT" : "tense",
    "NOM" : "case",
    "DAT" : "case",
    "ACC" : "case",
    "GEN" : "case",
    "1" : "person",
    "2" : "person",
    "3" : "person"
    }

def make_inputs(sentences: List[str]):
    for sentence_id, inputs in enumerate(sentences, 1):
        yield Input(sentence_id, tokens=list(data_io.get_tokens(inputs)))
        
def _get_encode_input(inputs: List[Input],
                      buckets_source: List[int],
                      source_vocabs: List[vocab.Vocab],
                      context: mx.context.Context) -> Tuple[mx.nd.NDArray,
                                                                           int]:

        batch_size = len(inputs)
        bucket_key = data_io.get_bucket(max(len(inp.tokens) for inp in inputs), buckets_source)
        num_source_factors = 1
        source = mx.nd.zeros((batch_size, bucket_key, num_source_factors), ctx=context)
       
        for j, input in enumerate(inputs):
            num_tokens = len(input)
            source[j, :num_tokens, 0] = data_io.tokens2ids(input.tokens, source_vocabs[0])

        return source, bucket_key
        
        
def encode(inputs: List[Input], 
           max_input_length: int,
           max_batch_size: int,
           source_vocabs: List[vocab.Vocab],
           context: mx.context.Context,
           s_model: source_model.SourceModel,
           bucket_source_width: Optional[int] = 10,
           fill_up_batches: bool = True):
        
        if bucket_source_width > 0:
            buckets_source = data_io.define_buckets(max_input_length, step=bucket_source_width)
        else:
            buckets_source = [max_input_length]

        # split into chunks
        input_chunks = []  # type: List[IndexedTranslatorInput]
        for input_idx, input in enumerate(inputs):
                max_input_length_without_eos = max_input_length -1
                # oversized input
                if len(input.tokens) > max_input_length_without_eos:
                    logger.debug(
                            "Input %s has length (%d) that exceeds max input length (%d). "
                            "Splitting into chunks of size %d.",
                            trans_input.sentence_id, len(trans_input.tokens),
                            buckets_source[-1], max_input_length_without_eos)
                    chunks = [input_chunk.with_eos()
                                  for input_chunk in input.chunks(max_input_length_without_eos)]
                    input_chunks.extend([IndexedInput(trans_input_idx, chunk_idx, chunk_input)
                                             for chunk_idx, chunk_input in enumerate(chunks)])
                    # regular input
                else:
                    input_chunks.append(IndexedInput(input_idx,
                                                         chunk_idx=0,
                                                         input=input.with_eos()))



        # Sort longest to shortest (to rather fill batches of shorter than longer sequences)
        input_chunks = sorted(input_chunks, key=lambda chunk: len(chunk.input.tokens), reverse=True)

        # translate in batch-sized blocks over input chunks
        batch_size = max_batch_size if fill_up_batches else min(len(input_chunks), self.max_batch_size)
        encoded_sequences = [] # list(number of batches, batch_size, seq_len, num_hidden)

        num_batches = 0
        for batch_id, batch in enumerate(utils.grouper(input_chunks, batch_size)):
            logger.debug("Encoding batch %d", batch_id)

            rest = batch_size - len(batch)
            if fill_up_batches and rest > 0:
                logger.debug("Padding batch of size %d to full batch size (%d)", len(batch), batch_size)
                batch = batch + [batch[0]] * rest

            inputs = [indexed_input.input for indexed_input in batch]
            source, bucket_key = _get_encode_input(inputs, buckets_source, source_vocabs, context)
            encoder_states = s_model.run_encoder(source, bucket_key) 
            #remove repeated sequences from filling up batch
            if rest>0:
                batches = encoder_states[0]
                batches = batches[:len(batches)-rest]
                encoder_states[0]=batches
            encoded_sequences.extend(encoder_states) # list of batches [batch, batch]
     
        return encoded_sequences
            
def preprocess(sequences: List[str], 
               features: List[OrderedDict],
               truecase_model: str, 
               bpe_model: str):
    
    bpe_codes = codecs.open(bpe_model, encoding='utf-8')
    bpe = BPE(codes=bpe_codes, merges=-1, separator='@@', vocab=None, glossaries=None)
    truecaser = sacremoses.MosesTruecaser(truecase_model)
    mpn = sacremoses.MosesPunctNormalizer()
    preprocessed_sequences = []
    tag_sequences = []
    
    for sentence, feature_list in zip(sequences, features):
        # normalize punctuation + tokenize (moses)
        string_sentence = ' '.join(sentence)
        normalized = mpn.normalize(string_sentence)
        #tokenized = tokenizer.tokenize(normalized, return_str=True)
        truecased = truecaser.truecase(normalized)
        # apply bpe
        bpe_split = bpe.segment(' '.join(truecased))
        ## map new subword units to their morphology tags
        feature_iterator = 0
        bpe_list = bpe_split.split()
        tags_list = [None] * len(bpe_list)
        for i, subword in enumerate(bpe_list):
            bpe_subword = bpe_list[i]
           
            if i==0:
                tags_list[i] = feature_list[feature_iterator]
            # if previous subword ends with @@, this subword belongs to the same word as the previous unit
            elif bpe_list[i-1].endswith('@@'):
                tags_list[i] = feature_list[feature_iterator]
            else:
                feature_iterator +=1
                tags_list[i] = feature_list[feature_iterator]
            #print("i=",i, "feat_it=", feature_iterator,"subword=", bpe_subword)
            #print("tags split: ", tags_split)
            
        preprocessed_sequences.append(bpe_split)
        tag_sequences.append(tags_list)
        if len(bpe_list) != len(tags_list):
            print("subword units and tag sequence have different length!")
            print("bpe: {}, tags {}".format(len(bpe_list), len(tags_list)))
            exit(0)
    return preprocessed_sequences, tag_sequences
    


def main():
    parser = argparse.ArgumentParser(description='Encode given sequences and return encoder hidden states.')
    arguments.add_device_args(parser)
    parser.add_argument('--conll',  type=str, 
                    help='File containing conll training data.')
    parser.add_argument('--sockeye-model',  type=str, 
                    help='Folder with trained sockeye model.')
    parser.add_argument('--batch-size',
                        type=arguments.int_greater_or_equal(1),
                        default=20,
                        help='Batch size. Default: %(default)s.')
    parser.add_argument('--bucket-width',
                        type=arguments.int_greater_or_equal(1),
                        default=10,
                        help='Width of buckets in tokens. Default: %(default)s.')
    parser.add_argument('--truecase-model',
                        type=str,
                        help='Model for moses truecaser.')
    parser.add_argument('--bpe-model',
                        type=str,
                        help='BPE model to split input sentences.')
    
    
    
    args = parser.parse_args()
    
    data_file = open(args.conll, "r", encoding="utf-8")
    #conll = conllu.parse_incr(data_file)
    conll = conllu.parse(data_file.read())
    
    with ExitStack() as exit_stack:
        context = utils.determine_context(device_ids=args.device_ids,
                                    use_cpu=False,
                                    disable_device_locking=True,
                                    lock_dir=args.lock_dir,
                                    exit_stack=exit_stack)[0]
    logger.info("Device: %s", context)
    
    # load vocab, config + model
    source_vocabs = vocab.load_source_vocabs(args.sockeye_model)
    model_config = model.SockeyeModel.load_config(os.path.join(args.sockeye_model, C.CONFIG_NAME))
    params_fname = os.path.join(args.sockeye_model, C.PARAMS_BEST_NAME)
    
    #sequences = [ sentence.metadata['text'] for sentence in conll]
    sequences = [[token["form"] for token in sentence] for sentence in conll]
    tags = [[token["feats"] for token in sentence] for sentence in conll]

    bpe_sequences, tag_sequences = preprocess(sequences, tags, args.truecase_model, args.bpe_model)
    
    max_source = max([source_sentence.split() for source_sentence in bpe_sequences], key=len)
    max_seq_len_source = len(max_source) +1 # <eos>
    s_model = source_model.SourceModel(config=model_config,
                                         params_fname=params_fname,
                                         context=context,
                                         max_input_length=max_seq_len_source)
    s_model.initialize(max_batch_size=args.batch_size,
                       max_input_length=max_seq_len_source)
    inputs = make_inputs(bpe_sequences)
    encoded_sequences = encode(inputs=inputs,
           max_input_length=max_seq_len_source,
           max_batch_size=args.batch_size,
           source_vocabs=source_vocabs,
           context=context,
           s_model=s_model,
           bucket_source_width=args.bucket_width,
           fill_up_batches=True)
    
    for i, batch in enumerate(encoded_sequences):
        print("i",i)
        print("len batch", len(batch))
        for j, encoded_sequence in enumerate(batch):
            print("j", j)
            bpe_sentence = bpe_sequences[(i*args.batch_size)+j].split(' ')
            tag_sequence = tag_sequences[(i*args.batch_size)+j]
            token_sequence = sequences[(i*args.batch_size)+j]
            print("len encoded_sequence {}".format(len(encoded_sequence)))
            
            
            bpe_iterator=0
            for word in token_sequence:
                features = tag_sequence[bpe_iterator]
                units =[bpe_sentence[bpe_iterator]]
                subword = bpe_sentence[bpe_iterator]
                hidden_states = [encoded_sequence[bpe_iterator]]
                #print("subword ", subword)
                
                
                while(subword.endswith('@@')):
                    bpe_iterator +=1
                    subword = bpe_sentence[bpe_iterator]
                    units.append(subword)
                    hidden_state = encoded_sequence[bpe_iterator] # shape (num_hidden,)
                    hidden_states.append(hidden_state.asnumpy())
                
                
                
                training_token = ClassifierInput(form=word,
                                                 bpe_list=units,
                                                 encoder_states=hidden_states,
                                                 units=len(units),
                                                 features={"gender": None, 
                                                            "number": None,
                                                            "case":  None,
                                                            "definiteness" : None,
                                                            "mood": None,
                                                            "finiteness": None,
                                                            "tense":  None,
                                                            "pos":  None,
                                                            "person" : None}
                                                 )
                # get gender, number, case etc
                if features is not None:
                    for tag in features:
                        if tag != "PASS":
                            morph_class = morph_features[tag]
                            training_token.features[morph_class] = tag
                
                
                bpe_iterator +=1
                print("len hidden ", len(hidden_states))
                print("classifier input: ", training_token)
            
            exit(0)
            

if __name__ == '__main__':
    main()
