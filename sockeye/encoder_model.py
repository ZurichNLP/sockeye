import logging
import os
import time
from collections import defaultdict
from typing import Callable, Dict, Generator, List, NamedTuple, Optional, Tuple, Union, Set, Any

import mxnet as mx
import numpy as np

from . import constants as C
from . import data_io
from . import model
from . import utils
from . import vocab

logger = logging.getLogger(__name__)



class EncoderModel(model.SockeyeModel):
    """
    InferenceModel is a SockeyeModel that supports three operations used for inference/decoding:

    (1) Encoder forward call: encode source sentence and return initial decoder states.
    (2) Decoder forward call: single decoder step: predict next word.

    :param config: Configuration object holding details about the model.
    :param params_fname: File with model parameters.
    :param context: MXNet context to bind modules to.
    :param beam_size: Beam size.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations as safety margin for maximum output length.
    :param decoder_return_logit_inputs: Decoder returns inputs to logit computation instead of softmax over target
                                        vocabulary.  Used when logits/softmax are handled separately.
    :param cache_output_layer_w_b: Cache weights and biases for logit computation.
    :param skip_softmax: If True, does not compute softmax for greedy decoding.
    """

    def __init__(self,
                 config: model.ModelConfig,
                 params_fname: str,
                 context: mx.context.Context,
                 max_input_length: int) -> None:
        super().__init__(config)
        self.params_fname = params_fname
        self.context = context
        self.max_input_length = max_input_length
        
        self.max_batch_size = None  # type: Optional[int]
        self.encoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.encoder_default_bucket_key = None  # type: Optional[int]
    


    def initialize(self, max_batch_size: int, max_input_length: int):
        """
        Delayed construction of modules to ensure multiple Inference models can agree on computing a common
        maximum output length.

        :param max_batch_size: Maximum batch size.
        :param max_input_length: Maximum input length.
        :param get_max_output_length_function: Callable to compute maximum output length.
        """
        self.max_batch_size = max_batch_size
        self.max_input_length = max_input_length
        if self.max_input_length > self.training_max_seq_len_source:
            logger.warning("Model was only trained with sentences up to a length of %d, "
                           "but a max_input_len of %d is used.",
                           self.training_max_seq_len_source, self.max_input_length)

        # check the maximum supported length of the encoder & decoder:
        if self.max_supported_seq_len_source is not None:
            utils.check_condition(self.max_input_length <= self.max_supported_seq_len_source,
                                  "Encoder only supports a maximum length of %d" % self.max_supported_seq_len_source)

        self.encoder_module, self.encoder_default_bucket_key = self._get_encoder_module()

        max_encoder_data_shapes = self._get_encoder_data_shapes(self.encoder_default_bucket_key,
                                                                self.max_batch_size)
        self.encoder_module.bind(data_shapes=max_encoder_data_shapes, for_training=False, grad_req="null")

        self.load_params_from_file(self.params_fname)
        self.encoder_module.init_params(arg_params=self.params, aux_params=self.aux_params, allow_missing=False)


    def _get_encoder_module(self) -> Tuple[mx.mod.BucketingModule, int]:
        """
        Returns a BucketingModule for the encoder. Given a source sequence, it returns
        the initial decoder states of the model.
        The bucket key for this module is the length of the source sequence.

        :return: Tuple of encoder module and default bucket key.
        """

        def sym_gen(source_seq_len: int):
            source = mx.sym.Variable(C.SOURCE_NAME)
            source_words = source.split(num_outputs=self.num_source_factors, axis=2, squeeze_axis=True)[0]
            source_length = utils.compute_lengths(source_words)

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # encoder
            # source_encoded: (source_encoded_length, batch_size, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len,
             pos_embed, position_probs) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len,
                                                           get_pos_embed=True, 
                                                           source=source)




            data_names = [C.SOURCE_NAME]
            label_names = []  # type: List[str]
            return mx.sym.Group([source_encoded]), data_names, label_names

        default_bucket_key = self.max_input_length
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key,
                                        context=self.context)
        return module, default_bucket_key
    
    @property
    def num_source_factors(self) -> int:
        """
        Returns the number of source factors of this InferenceModel (at least 1).
        """
        return self.config.config_data.num_source_factors
    
    @property
    def training_max_seq_len_source(self) -> int:
        """ The maximum sequence length on the source side during training. """
        return self.config.config_data.data_statistics.max_observed_len_source
    
    @property
    def max_supported_seq_len_source(self) -> Optional[int]:
        """ If not None this is the maximally supported source length during inference (hard constraint). """
        return self.encoder.get_max_seq_len()
    
    def _get_encoder_data_shapes(self, bucket_key: int, batch_size: int) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the encoder module.

        :param bucket_key: Maximum input length.
        :return: List of data descriptions.
        """
        return [mx.io.DataDesc(name=C.SOURCE_NAME,
                               shape=(batch_size, bucket_key, self.num_source_factors),
                               layout=C.BATCH_MAJOR)]
    
    def run_encoder(self,
                    source: mx.nd.NDArray,
                    source_max_length: int) -> 'ModelState':
        """
        Runs forward pass of the encoder.
        Encodes source given source length and bucket key.
        Returns encoder representation of the source, source_length, initial hidden state of decoder RNN,
        and initial decoder states tiled to beam size.

        :param source: Integer-coded input tokens. Shape (batch_size, source length, num_source_factors).
        :param source_max_length: Bucket key.
        :return: Initial model state.
        """
        batch_size = source.shape[0]
        batch = mx.io.DataBatch(data=[source],
                                label=None,
                                bucket_key=source_max_length,
                                provide_data=self._get_encoder_data_shapes(source_max_length, batch_size))

        self.encoder_module.forward(data_batch=batch, is_train=False)
        hidden_states = self.encoder_module.get_outputs()

       
        return hidden_states




