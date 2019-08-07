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
import joblib
import json
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
from . import train_morphology

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Encode given sequences and return encoder hidden states.')
    arguments.add_device_args(parser)
    parser.add_argument('--conll',  type=str, 
                    help='File containing conll test data.')
    parser.add_argument('--regression-model',  type=str, 
                    help='File containing the pickled regression model.')
    parser.add_argument('--gluon-model',  type=str, 
                    help='File containing the params of a trained gluon model.')
    parser.add_argument('--batch-size',
                        type=arguments.int_greater_or_equal(1),
                        default=20,
                        help='Batch size. Default: %(default)s.')
    parser.add_argument('--bucket-width',
                        type=arguments.int_greater_or_equal(1),
                        default=10,
                        help='Width of buckets in tokens. Default: %(default)s.')
    
    
    
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
    
    #load regression model + config
    if args.regression_model is not None:
        regression_config = args.regression_model + ".conf.json"
    else:
        regression_config = args.gluon_model + ".conf.json"
    with open(regression_config) as json_file:
        regression_config = json.load(json_file)
        sockeye_model = regression_config["sockeye-model"] 
        truecase_model = regression_config["truecase-model"] 
        bpe_model = regression_config["bpe-model"] 
        bpe_vocab = regression_config["bpe-vocab"] 
        feature = regression_config["feature"] 
        max_length_subwords = regression_config["max_length_subwords"]
        language = regression_config["language"]
        target = regression_config["target"]
        
        # load vocab, config + sockeye model
        source_vocabs = vocab.load_source_vocabs(sockeye_model)
        sockeye_model_config = model.SockeyeModel.load_config(os.path.join(sockeye_model, C.CONFIG_NAME))
        params_fname = os.path.join(sockeye_model, C.PARAMS_BEST_NAME)
        
        #sequences = [ sentence.metadata['text'] for sentence in conll]
        token_sequences = [[token["form"] for token in sentence] for sentence in conll]
        tags = [[token["feats"] for token in sentence] for sentence in conll]
        
        # sentences longer than in the conll training data 

        bpe_sequences, tag_sequences = train_morphology.preprocess(token_sequences, tags, truecase_model, bpe_model, bpe_vocab, target)
    
        max_source = max([source_sentence.split() for source_sentence in bpe_sequences], key=len)
        max_seq_len_source = len(max_source) +1 # <eos>
        s_model = source_model.SourceModel(config=sockeye_model_config,
                                            params_fname=params_fname,
                                            context=context,
                                            max_input_length=max_seq_len_source)
        s_model.initialize(max_batch_size=args.batch_size,
                        max_input_length=max_seq_len_source)
        
        inputs = train_morphology.make_inputs(bpe_sequences)
        
        encoded_sequences = train_morphology.encode(inputs=inputs,
            max_input_length=max_seq_len_source,
            max_batch_size=args.batch_size,
            source_vocabs=source_vocabs,
            context=context,
            s_model=s_model,
            bucket_source_width=args.bucket_width,
            fill_up_batches=True)
        
        training_tokens, max_length_subwords = train_morphology.get_encoded_tokens(encoded_sequences=encoded_sequences,
                                            language=language,
                                            batch_size=args.batch_size,
                                            tag_sequences=tag_sequences,
                                            bpe_sequences=bpe_sequences,
                                            token_sequences=token_sequences,
                                            fixed_max_length_subwords=max_length_subwords)
        
        labels, encoded_tokens = train_morphology.make_classifier_input(training_tokens, feature ,max_length_subwords) # labels: (sample_size,), encoded_tokens: (sample_size, max_length_subwords, hidden_dimension)
        
        (sample_size, max_length_subwords, hidden_dimension) = encoded_tokens.shape
        (sample_size_labels, ) = labels.shape
        print("testing sample size: ", sample_size)
            
        if sample_size != sample_size_labels:
            print("shapes do not match, encoded sequences have {} samples, but labels have {}".format(sample_size, sample_size_labels))
            exit(0)
    
        if args.regression_model is not None:
            trained_model = joblib.load(args.regression_model)
            # reshape to 2-dimensional array
            encoded_tokens = encoded_tokens.reshape(sample_size, max_length_subwords * hidden_dimension)
            #print(encoded_tokens.shape, labels.shape)
            result = trained_model.score(encoded_tokens, labels)
            print(result)
        elif args.gluon_model is not None:
            num_outputs = len(train_morphology.label_classes[feature])
            net = mx.gluon.nn.Sequential()
            with net.name_scope():
                net.add(mx.gluon.nn.Dense(num_outputs))
            net.load_parameters(args.gluon_model, ctx=context)
            
            test_set = mx.gluon.data.ArrayDataset(encoded_tokens, labels)
            test_dataloader = mx.gluon.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
            print(train_morphology.evaluate_accuracy(test_dataloader, net, context))

if __name__ == '__main__':
    main()
