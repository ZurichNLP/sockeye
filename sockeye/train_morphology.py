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
from subword_nmt.apply_bpe import BPE, read_vocabulary
from sklearn.linear_model import LogisticRegression
import joblib
import json
from collections import OrderedDict
import numpy as np
import re
from six import text_type

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

label_classes = {"pos" : {"ADJ":1,
                   "ADP":2,
                   "ADV":3,
                   "CONJ":4,
                   "DET":5,
                   "N":6,
                   "NUM":7,
                   "PART":8,
                   "PRO":9,
                   "PROPN":10,
                   "V":11,
                   "V.PTCP":12,
                   "X":13
                   },
        "gender" : { "FEM":1,
                     "NEUT":2,
                     "MASC":3
                    },
        "number": { "PL":1,
                   "SG":2
                    },
        "definiteness" : {"DEF":1,
                          "INDF":2
                    },
        "mood" : {"SBJV" :1,
                  "IND": 2,
                  "IMP": 3
                 },
        "finiteness" : {"FIN" :1,
                        "NFIN":2
                 },
        "tense" : {"PST":1,
                   "PRS":2,
                   "FUT":3
                },
        "case" : {"NOM":1,
                  "DAT":2,
                  "ACC":3,
                  "GEN":4
                },
        "person" : {"1":1,
                    "2":2,
                    "3":3
                }
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
            #if 1 in data_io.tokens2ids(input.tokens, source_vocabs[0]): # check if bpe splits consistent with vocab: NOTE: few cases where "/" and ":" in the conll are not split from their word -> will result in <unk>.. but we can't change tokenization in the conll.
                #print(input.tokens)
                #print(data_io.tokens2ids(input.tokens, source_vocabs[0]))

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
            #print(encoder_states[0].shape, len(encoder_states))
            #remove repeated sequences from filling up batch
            if rest>0:
                batches = encoder_states[0]
                batches = batches[:len(batches)-rest]
                encoder_states[0]=batches
            encoder_states = [state.asnumpy() for state in encoder_states]
            #print(encoder_states[0].shape, len(encoder_states))
            encoded_sequences.extend(encoder_states) # list of batches [batch, batch]
     
        return encoded_sequences
    

def normalize_punctuation(string: str):
    
    substitutions = [ # lines 33 - 34
        #(u'&', r'&amp;'), # already &amp; in conll
        (r'`', r"&apos;"),
        (r'\'', r"&apos;"),
        (r"''", r'&quot;'),
        (u'„', r'&quot;'),
        (u'“', r'&quot;'),
        (u'”', r'&quot;'),
        (u'"', r'&quot;'),
        (u'–', r'-'),
        (u'—', r'-'),
        (u'´', r"&quot;"),
        (u'«', r'&quot;'),
        (u'«', r'&quot;'),
        (u'>', r'&gt;'),
        (u'<', r'&lt;'),
        (r'\|', r'&#124;'),
        (r'\[', r'&#91;'),
        (r'\]', r'&#93;')
    ]
    for regexp, substitution in substitutions:
            #print(regexp, substitution)
            string = re.sub(regexp, substitution, text_type(string))
            #print(string)
    return string
            
def preprocess(sequences: List[str], 
               features: List[OrderedDict],
               truecase_model: str, 
               bpe_model: str,
               bpe_vocab: str):
    
    bpe_codes = codecs.open(bpe_model, encoding='utf-8')
    vocab = codecs.open(bpe_vocab, encoding='utf-8')
    vocab = read_vocabulary(vocab_file=vocab, threshold=None)
    bpe = BPE(codes=bpe_codes, merges=-1, separator='@@', vocab=vocab, glossaries=None)
    truecaser = sacremoses.MosesTruecaser(truecase_model)
    #mpn = sacremoses.MosesPunctNormalizer(lang='de')
    preprocessed_sequences = []
    tag_sequences = []
    
    # normalizer changes the tokenization/removes the whitespace before  some characters: %, :, ?, !, ;
    # -> need to fix this 
    punct = re.compile(r'([%:\?\!;])')
    
    for sentence, feature_list in zip(sequences, features):
        # normalize punctuation + tokenize (moses)
        string_sentence = ' '.join(sentence)
        #print("snt: ", string_sentence)
        normalized = normalize_punctuation(string_sentence)
        #if re.search(punct, normalized):
            #normalized = re.sub(punct, r' \1', normalized)
        #print("normalized ", normalized)
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
            #print(bpe_subword, i)
            if i==0:
                tags_list[i] = feature_list[feature_iterator]
            # if previous subword ends with @@, this subword belongs to the same word as the previous unit
            elif bpe_list[i-1].endswith('@@'):
                tags_list[i] = feature_list[feature_iterator]
            else:
                feature_iterator +=1
                tags_list[i] = feature_list[feature_iterator]
            #print("i=",i, "feat_it=", feature_iterator,"subword=", bpe_subword, "tag=", feature_list[feature_iterator])
            #print("tags list: ", tags_list)
            
        preprocessed_sequences.append(bpe_split)
        tag_sequences.append(tags_list)
        if len(bpe_list) != len(tags_list): # TODO: check if elements != None are same length
            logger.warn("subword units and tag sequence have different length!")
            logger.warn("bpe: {}, tags {}".format(len(bpe_list), len(tags_list)))
            exit(0)
            
    return preprocessed_sequences, tag_sequences


def get_encoded_tokens(encoded_sequences: List[List[np.array]], 
                       batch_size: int,  
                       tag_sequences: List[str],
                       bpe_sequences: List[str],
                       token_sequences: List[str],
                       fixed_max_length_subwords: Optional[int]=None): # if used in testing, max length for subwords is fixed: sentences with longer words have to be discarded
    
    training_tokens =[]
    counted_max_length_subwords=0
    
    for i, batch in enumerate(encoded_sequences):
        #print("i",i)
        #print("len batch", len(batch))
        for j, encoded_sequence in enumerate(batch): # encoded_sequence: np.array of shape (max_seq_len, hidden_dimension)
            #print("j", j)
            bpe_sentence = bpe_sequences[(i*batch_size)+j].split(' ')
            tag_sequence = tag_sequences[(i*batch_size)+j]
            token_sequence = token_sequences[(i*batch_size)+j]
            #print("len encoded_sequence {}, len bpe {}, tag sequence {}".format(len(encoded_sequence), len(bpe_sentence), len(tag_sequence)))
            #print(tag_sequence)
            #print(bpe_sentence)
        
            
            bpe_iterator=0
            for word in token_sequence:
                #print("word: ", word)
                features = tag_sequence[bpe_iterator]
                units =[bpe_sentence[bpe_iterator]]
                subword = bpe_sentence[bpe_iterator]
                hidden_states = [encoded_sequence[bpe_iterator]]
                
                
                while(subword.endswith('@@')):
                    bpe_iterator +=1
                    subword = bpe_sentence[bpe_iterator]
                    units.append(subword)
                    hidden_state = encoded_sequence[bpe_iterator] # shape (num_hidden,)
                    hidden_states.append(hidden_state)
                #print("units :" ,units)
                #print("bpe it: " ,bpe_iterator)
                if len(units) > counted_max_length_subwords:
                    counted_max_length_subwords=len(units)
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
                        if tag != "PASS" and tag !="REFL" and tag !="NEG":
                            morph_class = morph_features[tag]
                            training_token.features[morph_class] = tag
                
                
                bpe_iterator +=1
                training_tokens.append(training_token)
                #print("len hidden ", len(hidden_states))
                #print("classifier input: ", training_token)
        if fixed_max_length_subwords == None:
            fixed_max_length_subwords = counted_max_length_subwords
        elif counted_max_length_subwords > fixed_max_length_subwords:
            logger.warn("Number of subword units exceeds given max length in data. Given max length: {}, counted max length {}".format(fixed_max_length_subwords, counted_max_length_subwords))
            fixed_max_length_subwords = counted_max_length_subwords

        return training_tokens, fixed_max_length_subwords
    
def make_classifier_input(training_tokens: List[ClassifierInput], 
                          feature: str,
                          max_length_subwords: int):
    encoded_tokens =[]
    labels = []
    
    for training_token in training_tokens:
        
        label = training_token.features[feature]
        if label is not None:
            encoded_word = training_token.encoder_states # encoded_word: List of NDArrays with shape(hidden_dimension, )
            missing_length = max_length_subwords - len(encoded_word)
            data_type=encoded_word[0].dtype
            
            ## pad encoded tokens to max length to get input of uniform length
            if missing_length > 0:
                filler = np.zeros(shape=encoded_word[0].shape, dtype=data_type)
                #print(encoded_word[0].shape)
                #print(filler.shape)
                while missing_length >0:
                    encoded_word.append(filler)
                    missing_length -=1
            
            #print(len(encoded_word))
            label_id = label_classes[feature][label]
            #print(label, label_id)
            labels.append(label_id)
            encoded_tokens.append(encoded_word) # encoded word = list of list of subword hidden states 
    
    labels = np.array(labels, dtype=data_type) # shape (sample_size,)
    encoded_tokens = np.array([np.array(seq, dtype=data_type) for seq in encoded_tokens], dtype=data_type) # shape (sample_size, max_length_subwords, hidden_dimension)
    #print(encoded_tokens.shape, encoded_tokens.dtype)
    return labels,encoded_tokens

def train_logistic_regression(labels: np.array, encoded_sequences: np.array):
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=5000)
    #model = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=5000)
    (sample_size, max_length_subwords, hidden_dimension) = encoded_sequences.shape
    (sample_size_labels, ) = labels.shape
    if sample_size != sample_size_labels:
        print("shapes do not match, encoded sequences have {} samples, but labels have {}".format(sample_size, sample_size_labels))
        exit(0)
    encoded_sequences = encoded_sequences.reshape(sample_size, max_length_subwords * hidden_dimension)

    #print("dtypes: ", encoded_sequences.dtype, labels.dtype)
    #print("shapes: ", encoded_sequences.shape, labels.shape)
    model.fit(encoded_sequences, labels)
    return model
    
    

def main():
    parser = argparse.ArgumentParser(description='Encode given sequences and use encoder hidden states to train a logistic regression model that labels morphology.')
    arguments.add_device_args(parser)
    parser.add_argument('--conll',  type=str, required=True,
                    help='File containing conll training data.')
    parser.add_argument('--sockeye-model',  type=str, required=True,
                    help='Folder with trained sockeye model.')
    parser.add_argument('--out-model',  type=str, required=True,
                    help='Filename for saving the regression model.')
    parser.add_argument('--batch-size',
                        type=arguments.int_greater_or_equal(1),
                        default=20,
                        help='Batch size. Default: %(default)s.')
    parser.add_argument('--bucket-width',
                        type=arguments.int_greater_or_equal(1),
                        default=10,
                        help='Width of buckets in tokens. Default: %(default)s.')
    parser.add_argument('--truecase-model', required=True,
                        type=str,
                        help='Model for moses truecaser.')
    parser.add_argument('--bpe-model', required=True,
                        type=str,
                        help='BPE model to split input sentences.')
    parser.add_argument('--bpe-vocab', required=True,
                        type=str,
                        help='Vocabulary for BPE model.')
    parser.add_argument('--max-bpe-units-per-word', required=False,
                        type=int, default=None,
                        help='Expected maximum number of bpe units per word. Note: Cannot be larger at test time than observed in training data.')
    parser.add_argument('--feature', required=True,
                        type=str,
                        help='Feature to train classifer for. Features in conllu: gender, pos, tense, case, finiteness, definiteness, person, mood, number.')
    
    
    
    args = parser.parse_args()
    
    logger.info("Features in conllu need to be separated by '|', not ';'") # cannot handle sentences that have a non-final punctuation mark as their last token -> normalizer attaches it to previous word, changed tokenization ends in chaos
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
    token_sequences = [[token["form"] for token in sentence] for sentence in conll]
    tags = [[token["feats"] for token in sentence] for sentence in conll]

    bpe_sequences, tag_sequences = preprocess(token_sequences, tags, args.truecase_model, args.bpe_model, args.bpe_vocab)
    
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
    
    training_tokens, max_length_subwords = get_encoded_tokens(encoded_sequences=encoded_sequences,
                                         batch_size=args.batch_size,
                                         tag_sequences=tag_sequences,
                                         bpe_sequences=bpe_sequences,
                                         token_sequences=token_sequences,
                                         fixed_max_length_subwords=args.max_bpe_units_per_word)

    labels, encoded_tokens = make_classifier_input(training_tokens, args.feature ,max_length_subwords) # labels: (sample_size,), encoded_tokens: (sample_size, max_length_subwords, hidden_dimension)
    
    trained_model = train_logistic_regression(labels, encoded_tokens)
    joblib.dump(trained_model, args.out_model)
            
    regression_config = { "sockeye-model": args.sockeye_model,
                          "truecase-model" : args.truecase_model,
                          "bpe-model" : args.bpe_model,
                          "bpe-vocab" : args.bpe_vocab,
                          "feature" :  args.feature,
                          "max_length_subwords" : max_length_subwords
                        }
    
    with open( args.out_model + ".conf.json", 'w') as json_file:
        json.dump(regression_config, json_file)

if __name__ == '__main__':
    main()
