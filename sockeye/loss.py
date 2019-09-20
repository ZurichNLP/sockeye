# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Functions to generate loss symbols for sequence-to-sequence models.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import mxnet as mx
from mxnet.metric import EvalMetric

from . import config
from . import constants as C

logger = logging.getLogger(__name__)


class LossConfig(config.Config):
    """
    Loss configuration.

    :param name: Loss name.
    :param vocab_size: Target vocab size.
    :param normalization_type: How to normalize the loss.
    :param label_smoothing: Optional smoothing constant for label smoothing.
    """

    def __init__(self,
                 name: str,
                 vocab_size: int,
                 normalization_type: str,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.normalization_type = normalization_type
        self.label_smoothing = label_smoothing


def get_loss(loss_config: LossConfig) -> 'Loss':
    """
    Returns Loss instance.

    :param loss_config: Loss configuration.
    """
    if loss_config.name == C.CROSS_ENTROPY:
        return CrossEntropyLoss(loss_config)
    elif loss_config.name == C.COSINE_DIST:
        return CosineDistance(loss_config)
    else:
        raise ValueError("unknown loss name: %s" % loss_config.name)


class Loss(ABC):
    """
    Generic Loss interface.
    get_loss() method should return a loss symbol and the softmax outputs.
    The softmax outputs (named C.SOFTMAX_NAME) are used by EvalMetrics to compute various metrics,
    e.g. perplexity, accuracy. In the special case of cross_entropy, the SoftmaxOutput symbol
    provides softmax outputs for forward() AND cross_entropy gradients for backward().
    """

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss and softmax output symbols.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_metric(self) -> EvalMetric:
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        pass


class CrossEntropyLoss(Loss):
    """
    Computes the cross-entropy loss.

    :param loss_config: Loss configuration.
    """

    def __init__(self, loss_config: LossConfig) -> None:
        logger.info("Loss: CrossEntropy(normalization_type=%s, label_smoothing=%s)",
                    loss_config.normalization_type, loss_config.label_smoothing)
        self.loss_config = loss_config

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol, grad_scale: Optional[float]=1.0) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss symbol.
        """
        if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
            normalization = "valid"
        elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
            normalization = "null"
        else:
            raise ValueError("Unknown loss normalization type: %s" % self.loss_config.normalization_type)
        return [mx.sym.SoftmaxOutput(data=logits,
                                     label=labels,
                                     grad_scale=grad_scale,
                                     ignore_label=C.PAD_ID,
                                     use_ignore=True,
                                     normalization=normalization,
                                     smooth_alpha=self.loss_config.label_smoothing,
                                     name=C.SOFTMAX_NAME)]

    def create_metric(self) -> "CrossEntropyMetric":
        return CrossEntropyMetric(self.loss_config)


class CrossEntropyMetric(EvalMetric):
    """
    Version of the cross entropy metric that ignores padding tokens.

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param label_names: Name of labels that should be used when updating with update_dict.
    """

    def __init__(self,
                 loss_config: LossConfig,
                 name: str = C.CROSS_ENTROPY,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=output_names, label_names=label_names)
        self.loss_config = loss_config

    @staticmethod
    def cross_entropy(logprob, label):
        ce = -mx.nd.pick(logprob, label)  # pylint: disable=invalid-unary-operand-type
        return ce

    @staticmethod
    def cross_entropy_smoothed(logprob, label, alpha, num_classes):
        ce = CrossEntropyMetric.cross_entropy(logprob, label)
        # gain for each incorrect class
        per_class_gain = alpha / (num_classes - 1)
        # discounted loss for correct class
        ce *= 1 - alpha - per_class_gain
        # add gain for incorrect classes to total cross-entropy
        ce -= mx.nd.sum(logprob * per_class_gain, axis=-1, keepdims=False)
        return ce

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            batch_size = label.shape[0]
            label = label.as_in_context(pred.context).reshape((label.size,))

            logprob = mx.nd.log(mx.nd.maximum(1e-10, pred))

            # ce: (batch*time,)
            if self.loss_config.label_smoothing > 0.0:
                ce = self.cross_entropy_smoothed(logprob, label,
                                                 alpha=self.loss_config.label_smoothing,
                                                 num_classes=self.loss_config.vocab_size)
            else:
                ce = self.cross_entropy(logprob, label)

            # mask pad tokens
            valid = (label != C.PAD_ID).astype(dtype=pred.dtype)
            ce *= valid

            ce = mx.nd.sum(ce)
            if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
                num_valid = mx.nd.sum(valid)
                ce /= num_valid
                self.num_inst += 1
            elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
                # When not normalizing, we divide by the batch size (number of sequences)
                # NOTE: This is different from MXNet's metrics
                self.num_inst += batch_size

            self.sum_metric += ce.asscalar()

class CosineDistance(Loss):
    """
    Computes the cosine distance between two encoded sequences.

    :param loss_config: Loss configuration.
    """

    def __init__(self, loss_config: LossConfig) -> None:
        logger.info("Loss: CosineDistance")
        self.loss_config = loss_config
    
    def get_loss(self, 
                 encoded_sequence1: mx.sym.Symbol,
                 seq1_labels: mx.sym.Symbol,
                 source_len: int,
                 encoded_sequence2: mx.sym.Symbol, 
                 seq2_labels: mx.sym.Symbol,
                 trg_len: int,
                 grad_scale: Optional[float] = 1.0) -> List[mx.sym.Symbol]:
        return [
                mx.sym.MakeLoss(self.symbol_cosine_distance_avg(encoded_sequence1,
                                                                seq1_labels,
                                                                source_len,
                                                                encoded_sequence2,
                                                                seq2_labels,
                                                                trg_len),
                                grad_scale=grad_scale)]
    
    def create_metric(self) -> "CosineDistanceMetric":
        return CosineDistanceMetric(self.loss_config)
    
    #TODO: mask padded ids?
    def symbol_cosine_distance_avg(self, 
                                   seq1: mx.sym.Symbol, 
                                   seq1_labels: mx.sym.Symbol,
                                   seq1_len: int,
                                   seq2: mx.sym.Symbol, 
                                   seq2_labels: mx.sym.Symbol,
                                   seq2_len: int):
        # mask padded symbols, label shape (batch_size*sentence_length,), seq shape(batch, sentence_length, hidden)
        valid1 = (seq1_labels != C.PAD_ID).reshape(shape=(-1,1))
        valid2 = (seq2_labels != C.PAD_ID).reshape(shape=(-1,1)) # shape(batch*sentence_length,1), padded=0, else=1
        
        seq1_r = seq1.reshape(shape=(-3,-1)) # shape(batch_size*sentence_length, hidden)
        seq2_r = seq2.reshape(shape=(-3,-1))
        
        seq1 = mx.sym.broadcast_mul(seq1_r, valid1).reshape_like(seq1)
        seq2 = mx.sym.broadcast_mul(seq2_r, valid2).reshape_like(seq2)
        
        # -> shape(batch_size, dimension)
        avg1 = mx.sym.mean(seq1,axis=1)
        avg2 = mx.sym.mean(seq2,axis=1)
        expanded1 = mx.sym.expand_dims(avg1, axis=1)  # expanded1: (batch_size, 1, dimension)
        expanded2 = mx.sym.expand_dims(avg2, axis=2) # expanded2: (batch_size, dimension, 1)
        dot_prod = mx.sym.batch_dot(expanded1, expanded2) # dot_prod: (batch_size,1,1) 
        dot_prod = mx.sym.squeeze(dot_prod) # -> (batch_size)
        norm1 = mx.sym.sqrt(mx.sym.sum((avg1 * avg1), axis=1))
        norm2 = mx.sym.sqrt(mx.sym.sum((avg2 * avg2), axis=1))
        similarity = dot_prod / (norm1 * norm2)
        distance = 1.0 - similarity # mx.sym.Symbol, shape(batch_size,)
        return distance
    
    # TODO: pooling instead of average
    #def symbol_cosine_distance_pool(self, seq1, seq2):
        


class CosineDistanceMetric(EvalMetric):
    """
    Calculate the cosine distance between two encoded sequences.

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param label_names: Name of labels that should be used when updating with update_dict.
    """
    def __init__(self,
                 loss_config: LossConfig,
                 name: str = C.COSINE_DIST,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=output_names, label_names=label_names)
        self.loss_config = loss_config



    def update(self, distances, log_cosine_distance_per_batch):
        for distance in distances:
            (batch_size,) = distance.shape
            distance = mx.nd.sum(distance)
            self.num_inst += batch_size
            self.sum_metric += distance.asscalar()
            if log_cosine_distance_per_batch:
                logger.info("Average cosine distance for batch: {}".format(self.sum_metric/self.num_inst))

