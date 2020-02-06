import argparse
import sys
import time
import logging
import os
import mxnet as mx
from contextlib import ExitStack
from typing import Generator, Optional, List, Tuple, Union, NamedTuple
from sklearn.linear_model import LogisticRegression
import joblib
import json
from collections import OrderedDict
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
from . import probing_task_train

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Encode given sequences and return encoder hidden states.')
    arguments.add_device_args(parser)
    parser.add_argument('--input',  type=str, 
                    help='File containing test data.')
    parser.add_argument('--regression-model',  type=str, 
                    help='File containing the pickled regression model.')
    parser.add_argument('--gluon-model',  type=str, 
                    help='File containing the params of a trained gluon model.')
    parser.add_argument('--batch-size',
                        type=arguments.int_greater_or_equal(1),
                        default=20,
                        help='Batch size. Default: %(default)s.')
    parser.add_argument('--print-labels',
                        action='store_true',
                        default=False,
                        help='Print distribution of labels for each class. Default: %(default)s.')
    
    
    
    args = parser.parse_args()
    
    with open(args.input) as infile:
        input_sequences = infile.read().splitlines()
    
    max_seq = max([list(data_io.get_tokens(source_sentence)) for source_sentence in input_sequences], key=len)
    max_seq_len = len(max_seq) + 1 # eos
    
    with ExitStack() as exit_stack:
        context = utils.determine_context(device_ids=args.device_ids,
                                    use_cpu=False,
                                    disable_device_locking=True,
                                    lock_dir=args.lock_dir,
                                    exit_stack=exit_stack)[0]
    logger.info("Device: %s", context)
    
    #load regression model + config
    if args.regression_model is not None:
        regression_config = args.regression_model + ".conf.json"
    else:
        regression_config = args.gluon_model + ".conf.json"
    with open(regression_config) as json_file:
        regression_config = json.load(json_file)
        sockeye_model = regression_config["sockeye-model"] 
        
        
        # load vocab, config + sockeye model
        source_vocabs = vocab.load_source_vocabs(sockeye_model)
        sockeye_model_config = model.SockeyeModel.load_config(os.path.join(sockeye_model, C.CONFIG_NAME))
        params_fname = os.path.join(sockeye_model, C.PARAMS_BEST_NAME)
        
        
        logger.info("creating encoder model")
        s_model = encoder_model.EncoderModel(config=sockeye_model_config,
                                            params_fname=params_fname,
                                            context=context,
                                            max_input_length=max_seq_len)
        s_model.initialize(max_batch_size=args.batch_size,
                        max_input_length=max_seq_len)
        inputs = probing_task_train.make_inputs(input_sequences) 
        
        encoded_sequences, labels = probing_task_train.encode(inputs=inputs,
           max_input_length=max_seq_len,
           max_batch_size=args.batch_size,
           source_vocabs=source_vocabs,
           context=context,
           s_model=s_model,
           fill_up_batches=True)
        
        # remove samples that are padding
        pad_id = source_vocabs[0]["<pad>"]
        pad_indexes = np.where(labels==pad_id)
        labels = np.delete(labels, pad_indexes)
        encoded_sequences = np.delete(encoded_sequences, pad_indexes, axis=0)
        
        if args.print_labels:
            uniq_labels, freqs = np.unique(labels, return_counts=True)
            src_vocab_inv = vocab.reverse_vocab(source_vocabs[0])
            n = len(uniq_labels)
        
            sorted_indexes = freqs.argsort()[::-1][:n]
            for tok, freq in zip(uniq_labels[sorted_indexes], freqs[sorted_indexes]):
                logger.info("{} : {}".format(src_vocab_inv[tok], freq))
            
        if len(labels) != len(encoded_sequences):
            print("shapes do not match, encoded sequences have {} samples, but labels have {}".format(len(encoded_sequences), len(labels)))
            exit(1)
    
        if args.regression_model is not None:
            trained_model = joblib.load(args.regression_model)
            result = trained_model.score(encoded_sequences, labels)
            print(result)
        #elif args.gluon_model is not None:
            #num_outputs = len(train_morphology.label_classes[feature])
            #net = mx.gluon.nn.Sequential()
            #with net.name_scope():
                #net.add(mx.gluon.nn.Dense(num_outputs))
            #net.load_parameters(args.gluon_model, ctx=context)
            
            #test_set = mx.gluon.data.ArrayDataset(encoded_tokens, labels)
            #test_dataloader = mx.gluon.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
            #print(train_morphology.evaluate_accuracy(test_dataloader, net, context))

if __name__ == '__main__':
    main()
