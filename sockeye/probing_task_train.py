import argparse
import sys
import time
import logging
import os
import mxnet as mx
from contextlib import ExitStack
from typing import Generator, Optional, List, Tuple, Union, NamedTuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import json
from collections import OrderedDict, Counter
import numpy as np

import io
from . import vocab
from . import data_io
from . import inference
from . import arguments
from . import translate
from . import model
from . import constants as C
from . import encoder_model
from . import utils
from .log import setup_main_logger

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
                  'encoder_state',
                  'word_id'
                  )
    
    def __init__(self,
                 encoder_state: np.array,
                 form: str,
                 word_id: int) -> None:
        self.encoder_state = encoder_state
        self.form = form
        self.word_id = word_id
    
    def __str__(self):
        return 'ClassifierInput(%s, %s, %i, %s)' \
            % (self.form, self.word_id)
    
    

IndexedInput = NamedTuple('IndexedInput', [
    ('input_idx', int),
    ('chunk_idx', int),
    ('input', Input)
])



def make_inputs(sentences: List[str]):
    for sentence_id, inputs in enumerate(sentences, 1):
        yield Input(sentence_id, tokens=list(data_io.get_tokens(inputs)))
        
def _get_encode_input(inputs: List[Input],
                      #buckets_source: List[int],
                      max_input_length: int,
                      source_vocabs: List[vocab.Vocab],
                      context: mx.context.Context) -> Tuple[mx.nd.NDArray,
                                                                           int]:

        batch_size = len(inputs)
        #bucket_key = data_io.get_bucket(max(len(inp.tokens) for inp in inputs), buckets_source)
        num_source_factors = 1
        source = mx.nd.zeros((batch_size, max_input_length, num_source_factors), ctx=context)
       
        for j, input in enumerate(inputs):
            num_tokens = len(input)
            source[j, :num_tokens, 0] = data_io.tokens2ids(input.tokens, source_vocabs[0])
            #if 1 in data_io.tokens2ids(input.tokens, source_vocabs[0]): # check if bpe splits consistent with vocab: NOTE: few cases where "/" and ":" in the conll are not split from their word -> will result in <unk>.. but we can't change tokenization in the conll.
                #print(input.tokens)
                #print(data_io.tokens2ids(input.tokens, source_vocabs[0]))

        return source
        
        
def encode(inputs: List[Input], 
           max_input_length: int,
           max_batch_size: int,
           source_vocabs: List[vocab.Vocab],
           context: mx.context.Context,
           s_model: encoder_model.EncoderModel,
           bucket_source_width: Optional[int] = 10,
           fill_up_batches: bool = True):
        
        # no bucketing, pad to max_input_length -> input for classifier can only be 2-dimensional, need same length for all samples
        #if bucket_source_width > 0:
            #buckets_source = data_io.define_buckets(max_input_length, step=bucket_source_width)
        #else:
            #buckets_source = [max_input_length]
            
        
        # split into chunks
        input_chunks = []  # type: List[IndexedTranslatorInput]
        for input_idx, input in enumerate(inputs):
              
                input_chunks.append(IndexedInput(input_idx,
                                                         chunk_idx=0,
                                                         input=input.with_eos()))

        # translate in batch-sized blocks over input chunks
        batch_size = max_batch_size if fill_up_batches else min(len(input_chunks), self.max_batch_size)
        encoded_sequences = np.array([])
        labels = np.array([])
        num_batches = 0
        for batch_id, batch in enumerate(utils.grouper(input_chunks, batch_size)):
            logger.debug("batch ", batch_id)
            logger.debug("Encoding batch %d", batch_id)
            rest = batch_size - len(batch)
            if fill_up_batches and rest > 0:
                logger.debug("Padding batch of size %d to full batch size (%d)", len(batch), batch_size)
                batch = batch + [batch[0]] * rest

            inputs = [indexed_input.input for indexed_input in batch]
            source = _get_encode_input(inputs, max_input_length, source_vocabs, context)
            (encoder_states, pos_embed, positional_attention) = s_model.run_encoder(source, max_input_length)
            #remove repeated sequences from filling up batch
            if rest>0:
                batches = encoder_states
                batches = batches[:len(batches)-rest]
                encoder_states =batches
                
                batches_pos_embed = pos_embed
                batches_pos_embed = batches_pos_embed[:len(batches_pos_embed)-rest]
                pos_embed = batches_pos_embed
                
                batches_pos_att = positional_attention
                batches_pos_att = batches_pos_att[:len(batches_pos_att)-rest]
                positional_attention = batches_pos_att
            
                source = source[:len(source)-rest]                
            #if batch_id == 0:
                #encoded_sequences = encoder_states.copy()
                #labels = source
            #else:
                #encoded_sequences = mx.nd.concat(encoded_sequences, encoder_states.copy(), dim=0)
                #labels = mx.nd.concat(labels, source, dim=0)
            position_range = np.arange(start=0, stop=source.shape[1])
            ones = np.ones(shape=(source.shape[0], source.shape[1]))
            pos_labels = position_range * ones # (batch, max_len)
            pos_labels = pos_labels.reshape(-1)
          
            if batch_id == 0:
                encoded_sequences = encoder_states.reshape(shape=(-3, -1)).asnumpy() # (batch, seq_len, hidden) -> (batch*seq_len, hidden)
                labels = source.reshape(shape=(-3, -1)).asnumpy()
                position_labels = pos_labels
                
            else:
                encoded_sequences = np.concatenate((encoded_sequences, encoder_states.reshape(shape=(-3, -1)).asnumpy()))
                labels = np.concatenate((labels, source.reshape(shape=(-3, -1)).asnumpy()) )
                position_labels = np.concatenate((position_labels, pos_labels))
                
            #encoder_states = encoder_states.reshape(-3, -1).asnumpy()
            ### important: need to make a copy here, previous output will change with next batch!
            
            #encoded_sequences.append(encoder_states) # list of batches [batch, batch]
            #labels.append(source.reshape(-3,-1).asnumpy())
            #encoded_sequences = np.concatenate(encoded_sequences, encoder_state)
            #labels = np.concatenate(labels, source.reshape(shape=(-3, -1).asnumpy()))
           
        #print(encoded_sequences)
        # labels (batch * src_len, factors=1) -> remove factors dimension 
        labels = labels.squeeze()
        return encoded_sequences, labels, position_labels # len(encoded_sequences)= number of batches

        



def train_logistic_regression(labels: np.array, encoded_sequences: np.array, max_iter: int):
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=max_iter)
    #model = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=5000)
    (sample_size,  hidden_dimension) = encoded_sequences.shape
    (sample_size_labels, ) = labels.shape
    if sample_size != sample_size_labels:
        print("shapes do not match, encoded sequences have {} samples, but labels have {}".format(sample_size, sample_size_labels))
        exit(1)
    # use gpu for reshape, takes too long on cpu
   
    #print("dtypes: ", encoded_sequences.dtype, labels.dtype)
    #print("shapes: ", encoded_sequences.shape, labels.shape)
    logger.info("training on sample size: %s", sample_size)
    model.fit(encoded_sequences, labels)
    return model
    
def train_logistic_regression_gpu(labels: np.array, encoded_sequences: np.array, max_iter: int, context: mx.context.Context, batch_size: int, net: mx.gluon.nn.Sequential, epochs: int=10, patience: int=10):
    
    try:
        encoded_sequences = mx.nd.array((encoded_sequences),context)
        
    except mx.base.MXNetError: 
        # weird oom error, no fix yet 
        logger.error("out of memory, stopped")
        exit(1)
       
    #(sample_size, max_length_subwords, hidden_dimension) = encoded_sequences.shape
    #encoded_sequences = encoded_sequences.reshape(sample_size, max_length_subwords * hidden_dimension)
    labels = mx.nd.array((labels),context)
    train_set = mx.gluon.data.ArrayDataset(encoded_sequences, labels)
    train_dataloader = mx.gluon.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    
    logger.info("training on sample size: %s", len(labels))
    softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd' : 0.0001})

    num_not_improved =0
    best_acc =0.0
    for epoch in range(epochs):
        cumulative_train_loss = 0
        for i, (data, label) in enumerate(train_dataloader):
            with mx.autograd.record():
                # Do forward pass on a batch of training data
                data = data.as_in_context(context)
                label = label.as_in_context(context)
                output = net(data)

                # Calculate loss for the training data batch
                loss_result = softmax_cross_entropy(output, label)

            # Calculate gradients
            loss_result.backward()

            # Update parameters of the network
            trainer.step(batch_size)

            # sum losses of every batch
            cumulative_train_loss += mx.nd.sum(loss_result).asscalar()

        avg_train_loss = cumulative_train_loss / len(encoded_sequences)
        acc = evaluate_accuracy(train_dataloader, net, context)
        logger.info("epoch {}, average training loss {}, accuracy {}".format(epoch, avg_train_loss ,acc))
        if acc > best_acc:
            best_acc = acc
        else:
            num_not_improved +=1
        if num_not_improved >= patience:
            logger.info("Accuracy has not improved for {} epochs. Stopping Training.".format(num_not_improved))
            return model
    return model

def evaluate_accuracy(data_iterator, net, context):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(context)
        label = label.as_in_context(context)
        output = net(data)
        predictions = mx.nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]
    



def main():
    parser = argparse.ArgumentParser(description='Encode given sequences and use encoder hidden states to train a logistic regression model that labels morphology.')
    arguments.add_device_args(parser)
    parser.add_argument('--sockeye-model',  type=str, required=True,
                    help='Folder with trained sockeye model.')
    parser.add_argument('--pb-model',  type=str, required=True,
                    help='Filename for saving the probing task model.')
    parser.add_argument('--batch-size',
                        type=arguments.int_greater_or_equal(1),
                        default=20,
                        help='Batch size. Default: %(default)s.')
    parser.add_argument('--bucket-width',
                        type=arguments.int_greater_or_equal(1),
                        default=10,
                        help='Width of buckets in tokens. Default: %(default)s.')
    parser.add_argument('--max-iter',
                        type=int,
                        default=3000,
                        help='Maximum number of iterations for logistic regression classifier training. Default: %(default)s.')
    parser.add_argument('--train-on-gpu',
                        action='store_true',
                        help='Use Gluon to train logistic regression model on GPU (instead of sklearn on CPU). Default: %(default)s.')
    parser.add_argument('--save-label-frequencies',
                        action='store_true',
                        help='Print labels sorted by frequency to model config file. Default: %(default)s.')
    parser.add_argument('--epochs',
                        type=int, 
                        default=10,
                        help='Train gluon model for n epochs. Default: %(default)s.')
    parser.add_argument('--patience',
                        type=int, 
                        default=50,
                        help='Stop training after n epochs that accuracy in training has not improved.  Default: %(default)s.')
    parser.add_argument('--max-train-size',
                       type=int,
                        default=250000,
                        help='Maximum number of samples for training. Default: %(default)s.')
    parser.add_argument('-i', '--input', 
                        required=True, 
                        type=str,
                        help='Text file with input to be encoded.')
    parser.add_argument('--predict-positions',
                        action='store_true',
                        help='Train a model to predict the positions from the encoded states. Default: %(default)s.')
    
    args = parser.parse_args()
    
    setup_main_logger(file_logging=False,
                      console=True)
    
    
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
    
    with open(args.input) as infile:
        input_sequences = infile.read().splitlines()
    
    max_seq = max([list(data_io.get_tokens(source_sentence)) for source_sentence in input_sequences], key=len)
    max_seq_len = len(max_seq) + 1 # eos
    
    logger.info("creating encoder model")
    s_model = encoder_model.EncoderModel(config=model_config,
                                         params_fname=params_fname,
                                         context=context,
                                         max_input_length=max_seq_len)
    s_model.initialize(max_batch_size=args.batch_size,
                       max_input_length=max_seq_len)
    inputs = make_inputs(input_sequences)
    
    logger.info("encoding sentences")
    # lists of np.array , encoder states: (batch, maxlen, hidden), labels: (batch, maxlen, num_factors=1)
    encoded_sequences, labels, position_labels = encode(inputs=inputs,
           max_input_length=max_seq_len,
           max_batch_size=args.batch_size,
           source_vocabs=source_vocabs,
           context=context,
           s_model=s_model,
           bucket_source_width=args.bucket_width,
           fill_up_batches=True)

    logger.info("train classifier")
    #print(encoded_sequences.shape, labels.shape)
    
    # limit size of train set to --max-train-size
    cutoff=args.max_train_size
    if len(labels) > cutoff:
        (encoded_sequences, rest) = np.split(encoded_sequences, [cutoff])
        (labels, rest) = np.split(labels, [cutoff])
        (position_labels, rest) = np.split(position_labels, [cutoff])
    
    
    # remove samples that are padding
    pad_id = source_vocabs[0]["<pad>"]
    pad_indexes = np.where(labels==pad_id)
    labels = np.delete(labels, pad_indexes)
    position_labels = np.delete(position_labels, pad_indexes)
    encoded_sequences = np.delete(encoded_sequences, pad_indexes, axis=0)
    
    regression_config = { "sockeye-model": args.sockeye_model, 
                         "max-iter":args.max_iter, 
                         "sample-size": len(labels), 
                         "max-seq-len": max_seq_len, 
                         "frequencies": {}
                        }
    
    if args.save_label_frequencies:
        uniq_labels, freqs = np.unique(labels, return_counts=True)
        src_vocab_inv = vocab.reverse_vocab(source_vocabs[0])
        n = len(uniq_labels)
        
        sorted_indexes = freqs.argsort()[::-1][:n]
        for tok, freq in zip(uniq_labels[sorted_indexes], freqs[sorted_indexes]):
            logger.debug("{} : {}".format(src_vocab_inv[tok], freq))
            regression_config["frequencies"][src_vocab_inv[tok]] = int(freq)
        
    
    if args.train_on_gpu:
        if args.predict_positions:
            num_outputs = max_seq_len
            net = mx.gluon.nn.Sequential()
            with net.name_scope():
                #net.add(mx.gluon.nn.Dense(1024, activation="tanh"))
                net.add(mx.gluon.nn.Dense(num_outputs))
            #net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=context)
            net.initialize(mx.init.Xavier(), ctx=context)
            trained_model = train_logistic_regression_gpu(position_labels, encoded_sequences, args.max_iter, context, args.batch_size, net, args.epochs, args.patience)
            net.save_parameters(args.pb_model)
            
        else:
            num_outputs = len(source_vocabs[0])
            net = mx.gluon.nn.Sequential()
            with net.name_scope():
                #net.add(mx.gluon.nn.Dense(1024, activation="tanh"))
                net.add(mx.gluon.nn.Dense(num_outputs))
            #net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=context)
            net.initialize(mx.init.Xavier(), ctx=context)
            trained_model = train_logistic_regression_gpu(labels, encoded_sequences, args.max_iter, context, args.batch_size, net, args.epochs, args.patience)
            net.save_parameters(args.pb_model)
    else:
        trained_model = train_logistic_regression(labels, encoded_sequences, args.max_iter)
        joblib.dump(trained_model, args.pb_model)
            
    
    
    with open( args.pb_model + ".conf.json", 'w') as json_file:
        json.dump(regression_config, json_file)

if __name__ == '__main__':
    main()
