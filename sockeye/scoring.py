# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Implement scoring of existing translations.
"""
import os
import logging
import argparse
from typing import Optional, List, Dict, Tuple, cast, DefaultDict
from collections import defaultdict

import mxnet as mx
import numpy as np

import sockeye
from . import constants as C
from . import data_io
from . import vocab
from . import model
from . import utils
from . import loss
from . import inference

from sockeye.utils import check_condition

logger = logging.getLogger(__name__)


class ScoringModel(model.SockeyeModel):
    """
    Defines a model to score input/output data.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU)
    :param data_iter: The iterator over the data set.
    :param config: Configuration object holding details about the model.
    :param checkpoint: Checkpoint to load. If None, finds best parameters in
    model_folder.
    :param bucketing: If True bucketing will be used, if False the computation
    graph will always be unrolled to the full length.
    """

    def __init__(self,
                 model_folder: str,
                 context: List[mx.context.Context],
                 data_iter: data_io.BaseParallelSampleIter,
                 config: model.ModelConfig,
                 checkpoint: Optional[int] = None,
                 bucketing: bool = None) -> None:
        super().__init__(config)
        self.context = context
        self.bucketing = bucketing
        self._build_model_components()
        self.module = self._build_module(data_iter)

        self.config = config
        self.batch_size = data_iter.batch_size
        self.fname_params = os.path.join(model_folder, C.PARAMS_NAME % checkpoint if checkpoint else C.PARAMS_BEST_NAME)

        self.module.bind(data_shapes=data_iter.provide_data, label_shapes=data_iter.provide_label, for_training=False, force_rebind=True, grad_req='null')

        self.load_params_from_file(self.fname_params)
        self.module.init_params(arg_params=self.params, allow_missing=False)

    def _build_module(self, data_iter: data_io.BaseParallelSampleIter):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """

        source = mx.sym.Variable(C.SOURCE_NAME)
        source_words = source.split(num_outputs=self.config.config_embed_source.num_factors,
                                    axis=2, squeeze_axis=True)[0]
        source_length = utils.compute_lengths(source_words)
        target = mx.sym.Variable(C.TARGET_NAME)
        target_length = utils.compute_lengths(target)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        model_loss = loss.get_loss(self.config.config_loss)

        data_names = [x[0] for x in data_iter.provide_data]
        label_names = [x[0] for x in data_iter.provide_label]
        # scores = [[None] for x in data_iter.provide_label]

        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # target embedding
            (target_embed,
             target_embed_length,
             target_embed_seq_len) = self.embedding_target.encode(target, target_length, target_seq_len)

            # encoder
            # source_encoded: (batch_size, source_encoded_length, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len)

            # decoder
            # target_decoded: (batch-size, target_len, decoder_depth)
            target_decoded = self.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                          target_embed, target_embed_length, target_embed_seq_len)

            # target_decoded: (batch_size * target_seq_len, rnn_num_hidden)
            target_decoded = mx.sym.reshape(data=target_decoded, shape=(-3, 0))

            # output layer
            logits = self.output_layer(target_decoded)

            probs = model_loss.get_loss(logits, labels)
            return mx.symbol.Group(probs), data_names, label_names

        if self.bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", data_iter.default_bucket_key)
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                          logger=logger,
                                          default_bucket_key=data_iter.default_bucket_key,
                                          context=self.context)
        else:
            logger.info("No bucketing. Unrolled to (%d,%d)",
                        self.config.config_data.max_seq_len_source, self.config.config_data.max_seq_len_target)
            symbol, _, __ = sym_gen(data_iter.buckets[0])
            return mx.mod.Module(symbol=symbol,
                                 data_names=data_names,
                                 label_names=label_names,
                                 logger=logger,
                                 context=self.context)


Tokens = List[str]
ScoredBatch = List[Tuple[int, float]]
ScoredSamples = List[ScoredBatch]


class ScoringOutput:
    """
    Wraps scorer outputs as TranslatorOutputs.
    """

    __slots__ = ('sentence_id', 'source_tokens', 'target_tokens', 'scores')

    def __init__(self,
                 sentence_id: int,
                 source_tokens: Optional[Tokens] = None,
                 target_tokens: Optional[Tokens] = None,
                 scores: List[float]) -> None:
        self.sentence_id = sentence_id
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.scores = scores

    def __str__(self):
        return 'ScoringOutput(%d, %s, %s, %s)' % (self.sentence_id, self.source_tokens, self.target_tokens, str(self.scores))


class Scorer:

    def __init__(self,
                 batch_size: int,
                 no_bucketing: bool = False,
                 normalize: Optional[bool] = True) -> None:

        self.batch_size = batch_size
        self.no_bucketing = no_bucketing
        self.normalize = normalize

    def _score_batch(self,
                     model: ScoringModel,
                     batch: mx.io.DataBatch,
                     batch_index: Tuple[int, int],
                     mapid: DefaultDict[DefaultDict[int, int]]
                     ) -> ScoredBatch:
        """
        Scores a batch of examples, returns a list of tuples (sentence_id, score).
        :param data_iter: The iterator over the data set.
        :param mapid: dictionary that maps input order to samples in buckets.
        :param batch_size: Batch size.
        :param normalize: If true, normalizes score by length of target.
        :return: Dictionary where key=input_id, value=score.
        """
        scored_batch = []

        model.module.forward(batch, is_train=False)
        outputs = model.module.get_outputs()

        # split output array into probs per batch
        sample_length = int(len(outputs[0]) / self.batch_size)

        probs = mx.nd.array(outputs[0])  # shape is (t_len*batch_size, t_vocab)
        probs_per_batch = [probs[i*sample_length:(i+1)*sample_length] for i in range(self.batch_size)]

        # get bucket index of batch
        (bucket_index, offset) = batch_index

        for sample_number, sample_probs in enumerate(probs_per_batch):

            mapsample_number = sample_number + offset

            if mapsample_number in mapid[bucket_index]:
                labels = batch.label[0][sample_number]
                # print("probs sample nr, {}, map sample nr {}, probs {}, labels {}".format(sample_number, mapsample_number,sample_probs.shape, labels))
                scores = mx.nd.pick(sample_probs, labels)
                scores = scores.asnumpy()
                labels = labels.asnumpy()
                non_pad_indices = np.where(labels != 0)
                # print("labels: {}".format(labels))
                # print("non pad: {}".format(non_pad_indices))
                scores = np.take(scores, non_pad_indices)
                log_probs = - np.log(scores)
                score = np.nansum(log_probs)
                # TODO: inf?
                if self.normalize:
                    score = np.mean(log_probs)
                sentence_id = mapid[bucket_index][mapsample_number]

                scored_batch.append((sentence_id, score))
            else:  # TODO: turn off?
                logger.debug("sample {} in batch number {} was filler".format(sample_number, batch_index))

        return scored_batch

    def _score_single_model(self,
                            model: ScoringModel,
                            model_id: int,
                            data_iter: data_io.BaseParallelSampleIter,
                            mapid: DefaultDict) -> ScoredSamples:
        """
        """
        scored_samples = []

        for i, batch in enumerate(data_iter):

                batch_index = data_iter.batch_indices[i]
                scored_batch = self._score_batch(model=model,
                                                 model_id=model_id,
                                                 batch=batch,
                                                 batch_index=batch_index,
                                                 mapid=mapid)
                scored_samples.append(scored_batch)

        return sorted(scored_samples)

    def score(self,
              models: List[ScoringModel],
              data_iters: List[data_io.BaseParallelSampleIter],
              mapids: List[defaultdict(lambda: defaultdict(int))]
              ) -> List[ScoringOutput]:
        """
        Scores all examples returned by an iterator, once for each model.
        Returns a list of ScoringOutputs.
        """
        scored_samples = []

        for index, (model, data_iter, mapid) in enumerate(zip(models, data_iters, mapids)):
            single_model_scored_samples = self._score_single_model(model=model,
                                                                   model_id=index,
                                                                   data_iter=data_iter,
                                                                   mapid=mapid)
            scored_samples.append(single_model_scored_samples)

        return self._make_outputs(scored_samples)

    def _make_outputs(self,
                      scored_samples: ScoredSamples) -> List[ScoringOutput]:
        """
        Generates a list of ScoringOutputs from output.
        """
        scored_outputs = []

        for sample in zip(*scored_samples):
            scores = [score for (sentence_id, score) in sample]
            sentence_id = sample[0][0]

            scored_output = ScoringOutput(sentence_id=sentence_id,
                                          scores=scores)
            scored_outputs.append(scored_output)

        return scored_outputs


def load_models(context: mx.context.Context,
                batch_size: int,
                model_folders: List[str],
                data_iters: List[data_io.BaseParallelSampleIter],
                configs: List[data_io.DataConfig],
                checkpoints: Optional[List[int]] = None,
                bucketing: bool = None) -> List[ScoringModel]:

    """
    Loads a list of models for scoring.

    :param context: MXNet context to bind modules to.
    :param batch_size: Batch size.
    :param model_folders: List of model folders to load models from.
    :param data_iter: List of iterators over the data set (one per model, can use different vocabularies).
    :param configs: List of configuration objects holding details about the models (one per model).
    :param checkpoints: List of checkpoints to use for each model in model_folders. Use None to load best checkpoint.
    :return: List of models.
    """

    models = []
    if checkpoints is None:
        checkpoints = [None] * len(model_folders)

    for model_folder, checkpoint, config, data_iter in zip(model_folders, checkpoints, configs, data_iters):

        model = ScoringModel(model_folder=model_folder,
                             context=context,
                             data_iter=data_iter,
                             config=config,
                             checkpoint=checkpoint,
                             bucketing=bucketing)
        models.append(model)

    return models


def create_data_iter_and_vocab(args: argparse.Namespace,
                               max_seq_len_source: int,
                               max_seq_len_target: int,
                               model_dir: str)-> Tuple['data_io.BaseParallelSampleIter', 'data_io.DataConfig']:
    """
    Create the data iterator.

    :param args: Arguments as returned by argparse.
    :param max_seq_len_source: Max length input.
    :param max_seq_len_target: Max length output.
    :model_dir: model folder to load vocabularies and config from.
    :return: The data iterator.
    """
    ###

    batch_num_devices = 1 if args.use_cpu else sum(-di if di < 0 else 1 for di in args.device_ids)
    batch_by_words = args.batch_type == C.BATCH_TYPE_WORD

    source_vocabs = vocab.load_source_vocabs(model_dir)
    target_vocab = vocab.vocab_from_json(os.path.join(model_dir, C.VOCAB_TRG_NAME))

    ## Recover the vocabulary path from the data info file:
    data_info = cast(data_io.DataInfo, sockeye.config.Config.load(os.path.join(model_dir, C.DATA_INFO)))
    source_vocab_paths = data_info.source_vocabs
    target_vocab_path = data_info.target_vocab

    # get max_seq_len_source and max_seq_len_target from config, warn if smaller than given values
    config = model.SockeyeModel.load_config(os.path.join(model_dir, C.CONFIG_NAME))
    if max_seq_len_source > config.config_data.max_seq_len_source:
            logger.warning("Source sentence of length %d in test set exceeds maximum source sentence length in config of %d",max_seq_len_source, config.config_data.max_seq_len_source)
    if max_seq_len_target > config.config_data.max_seq_len_target:
            logger.warning("Target sentence of length %d in test set exceeds maximum target sentence length in config of %d",max_seq_len_target, config.config_data.max_seq_len_target)

    check_condition(len(args.source_factors) == len(args.source_factors_num_embed),
                        "Number of source factor data (%d) differs from provided source factor dimensions (%d)" % (len(args.source_factors), len(args.source_factors_num_embed)))

    sources = [args.source] + args.source_factors
    sources = [str(os.path.abspath(source)) for source in sources]

    # print("sources {} source vocabs {}".format(sources, source_vocabs))
    sources_sentences = [data_io.SequenceReader(source, vocab, add_bos=False) for source, vocab in zip(sources, source_vocabs)]
    target_sentences = data_io.SequenceReader(args.target, target_vocab, add_bos=True, limit=None)

    # Pass 1: get target/source length ratios.
    length_statistics = data_io.analyze_sequence_lengths(sources, args.target, source_vocabs, target_vocab, max_seq_len_source, max_seq_len_target)

    # define buckets
    bucketing=not args.no_bucketing
    buckets = data_io.define_parallel_buckets(max_seq_len_source, max_seq_len_target, args.bucket_width, length_statistics.length_ratio_mean) if bucketing else [(max_seq_len_source, max_seq_len_target)]


    ### get iter
    ## 2. pass: Get data statistics
    data_statistics = data_io.get_data_statistics(sources_sentences,
                                                          target_sentences,
                                                          buckets,
                                                          length_statistics.length_ratio_mean, length_statistics.length_ratio_std,
                                                          source_vocabs,
                                                          target_vocab)

    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets,
                                                   args.batch_size,
                                                   batch_by_words,
                                                   batch_num_devices,
                                                   data_statistics.average_len_target_per_bucket)

    data_statistics.log(bucket_batch_sizes)

    data_loader = data_io.RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=target_vocab[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID)

    parallel_data = data_loader.load(sources_sentences, target_sentences,
                                     data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes, args.fill_up)
    map_buckets2sentence_ids = data_loader.map_buckets2sentence_ids

    data_iter = data_io.ParallelSampleIter(parallel_data,
                                    buckets,
                                    args.batch_size,
                                    bucket_batch_sizes,
                                    no_shuffle=True)

    return data_iter, config, map_buckets2sentence_ids


def get_max_source_and_target(args: argparse.Namespace) -> Tuple[int, int]:
    source_lines = utils.smart_open(args.source).readlines()
    target_lines = utils.smart_open(args.target).readlines()
    max_len_source = max([len(line.rstrip().split()) for line in source_lines])
    max_len_target = max([len(line.rstrip().split()) for line in target_lines])
    # +1 for EOS
    return max_len_source+1, max_len_target+1

