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
from typing import List, Optional, Dict, Tuple

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
    :param link: Link function.
    :param weight: Loss weight.
    """

    def __init__(self,
                 name: str,
                 vocab_size: Optional[int] = None,
                 normalization_type: Optional[str] = None,
                 label_smoothing: float = 0.0,
                 length_task_link: Optional[str] = None,
                 length_task_weight: float = 1.0) -> None:
        super().__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.normalization_type = normalization_type
        self.label_smoothing = label_smoothing
        self.length_task_link = length_task_link
        self.length_task_weight = length_task_weight


def get_loss(config: LossConfig) -> 'Loss':
    """
    Returns a Loss instance.

    :param config: Loss configuration.
    :return: Instance implementing the Loss.
    """
    if config.name == C.CROSS_ENTROPY:
        return CrossEntropyLoss(config,
                                output_names=[C.SOFTMAX_OUTPUT_NAME],
                                label_names=[C.TARGET_LABEL_NAME])
    elif config.name == C.ATTENTION_MONOTONICITY_LOSS:
        return AttentionMonotonicity(config)
    else:
        raise ValueError("unknown loss name: %s" % config.name)


def get_length_task_loss(config: LossConfig) -> 'Loss':
    """
    Returns a Loss instance.

    :param config: Loss configuration.
    :return: Instance implementing Loss.
    """
    if config.length_task_link is not None:
        if config.length_task_link == C.LINK_NORMAL:
            return MSELoss(config,
                           output_names=[C.LENRATIO_OUTPUT_NAME],
                           label_names=[C.LENRATIO_LABEL_NAME])
        elif config.length_task_link == C.LINK_POISSON:
            return PoissonLoss(config,
                               output_names=[C.LENRATIO_OUTPUT_NAME],
                               label_names=[C.LENRATIO_LABEL_NAME])
        else:
            raise ValueError("unknown link function name for length task: %s" % config.length_task_link)
    return None


class Loss(ABC):
    """
    Generic Loss interface.
    get_loss() method should return a loss symbol.
    The softmax outputs (named C.SOFTMAX_NAME) are used by EvalMetrics to compute various metrics,
    e.g. perplexity, accuracy. In the special case of cross_entropy, the SoftmaxOutput symbol
    provides softmax outputs for forward() AND cross_entropy gradients for backward().
    """

    def __init__(self, loss_config: LossConfig, output_names: List[str], label_names: List[str]) -> None:
        self.output_names = output_names
        self.label_names = label_names
        self.loss_config = loss_config

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: Loss symbol.
        """
        raise NotImplementedError()

    def __repr__(self):
        return self.loss_config.name

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

    def __init__(self, loss_config: LossConfig,
                 output_names: List[str], label_names: List[str],
                 ignore_label: int=C.PAD_ID, name: str=C.SOFTMAX_NAME) -> None:
        logger.info("Loss: CrossEntropy(normalization_type=%s, label_smoothing=%s)",
                    loss_config.normalization_type, loss_config.label_smoothing)
        super().__init__(loss_config=loss_config, output_names=output_names, label_names=label_names)
        self.ignore_label = ignore_label
        self.name = name

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol, grad_scale: Optional[float] = 0.5) -> mx.sym.Symbol:
        """
        Returns loss symbol given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss symbols.
        """
        if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
            normalization = "valid"
        elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
            normalization = "null"
        else:
            raise ValueError("Unknown loss normalization type: %s" % self.loss_config.normalization_type)
        return mx.sym.SoftmaxOutput(data=logits,
                                    label=labels,
                                    grad_scale=grad_scale,
                                    ignore_label=self.ignore_label,
                                    use_ignore=True,
                                    normalization=normalization,
                                    smooth_alpha=self.loss_config.label_smoothing,
                                    name=self.name)

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

class AttentionMonotonicity(Loss):
    """
    Computes the attention monotonicity loss.

    :param loss_config: Loss configuration.
    """

    def __init__(self, loss_config: LossConfig) -> None:
        logger.info("Loss: AttentionMonotonicity")
        self.loss_config = loss_config
    
    def get_loss(self, 
                 attention_scores_list: List[mx.sym.Symbol],
                 num_attention_heads: int,
                 target_words: mx.sym.Symbol,
                 layers: Optional[str] = 'last', # 'last' or 'all' layers
                 grad_scale: Optional[float] = 0.5) -> List[mx.sym.Symbol]:
        
        if layers == "last":
            loss = self.monotonicity_score_per_layer(attention_scores_list[-1], num_attention_heads, target_words)
        
        else:
            accumulated_loss = mx.sym.zeros_like(attention_scores_list[0])
            for layer_scores in attention_scores_list:
                layer_loss = self.monotonicity_score_per_layer(layer_scores, num_attention_heads, target_words) # (batch,)
                accumulated_loss = accumulated_loss + layer_loss
            
            loss = mx.sym.mean(accumulated_loss, axis=0)
        
        return mx.sym.MakeLoss(loss,
                                grad_scale=grad_scale)
    
    
    def monotonicity_score_per_layer(self, 
                                     attention_scores: mx.sym.Symbol,
                                     num_attention_heads: int,
                                     target_words: mx.sym.Symbol):
        """
        param attention_scores: decoder-encoder attention scores (MultiHeadAttention), shape (batch_size * attention_heads, target_length, source_length)
        param target_words: target words, used to remove padding. shape (batch_size * target_length)
        param default_bucket_key: Tuple of (max_source_length, max_target_length). Need max_source_length to create position matrix with arange.
        """
        
        # take average of attention_heads on each position
        attention_scores = attention_scores.reshape(shape=(-4, -1, num_attention_heads, -2)) # (batch_size, attention_heads, target_length, source_length)
        attention_scores = mx.sym.mean(attention_scores, axis=1) # (batch_size, target_length, source_length)
        
        p = mx.sym.ones_like(attention_scores) # (batch_size, target_length, source_length)
        positions = mx.contrib.sym.arange_like(data=attention_scores, start=1, axis=-1) # (source_length, ) NOTE: needs mxnet-1.6.0
        
        positions = positions.reshape(shape=(1,1, -1))  # (1, 1, source_length)
        positions = mx.sym.broadcast_mul(p, positions) # shape(batch, target_length, source_length), values in source_length = arange(source_length) 
        
        
        # no need to remove padding from source, padded positions are 0 in attention scores
        positionally_weighted_attention = mx.sym.broadcast_mul(attention_scores, positions) # shape(batch_size, target_length, source_length (attention_score*position))
        # take average over sequences
        avg = mx.sym.mean(positionally_weighted_attention, axis=2) # shape (batch, target_length)
        
        ### set padded positions in target to zero (we dont care about alignment scores from padded tokens)
        mask = (target_words != C.PAD_ID) # target_words (batch_size, target_length), mask: 0 where padded, 1 otherwise
        avg_r = avg.reshape(shape=(-3,))
        avg = mx.sym.broadcast_mul(avg_r, mask).reshape_like(avg)
        
        avg_to_shift = mx.sym.expand_dims(avg, axis=0)
        avg_to_shift = mx.sym.expand_dims(avg_to_shift, axis=0) # (1,1, batch, target_length)
        
        # shift target_length dimension to the left with BilinearSampler https://mxnet.apache.org/api/python/docs/api/symbol/symbol.html?highlight=bilinear#mxnet.symbol.BilinearSamplers
        ones = mx.sym.ones_like(avg) # (batch, target_length)
        zeros = mx.sym.zeros_like(avg)
        stacked = mx.sym.stack(ones, zeros, axis=0) # (2, batch, target_length)
        warp = mx.sym.expand_dims(stacked, axis=0) # (1, 2, batch, target_length)
        grid = mx.sym.GridGenerator(data=warp, transform_type='warp')
        shifted_avg = mx.sym.BilinearSampler(avg_to_shift, grid).squeeze() # (1,1,batch,target_length) -> (batch, target_length)
        
        adjacent_pos_difference = avg - shifted_avg # (batch, target_length)
        # loss= max(0, avg(y)-avg(y+1))
        greater_than_zero = mx.sym.broadcast_greater_equal(adjacent_pos_difference, mx.sym.zeros(shape=(1,))) # 1 if avg>0, 0 otherwise shape (batch, target_length)
        adjacent_pos_difference  = adjacent_pos_difference * greater_than_zero # (batch, target_length): target_length = 0 or diff attention score (y-(y+1))
        
        layer_loss = mx.sym.sum(adjacent_pos_difference, axis=1) # (batch, )
        return layer_loss

    def create_metric(self) -> "AttentionMonotonicityMetric":
        return AttentionMonotonicityMetric(self.loss_config)
        

class AttentionMonotonicityMetric(EvalMetric):
    """
    Calculate the monotonicity of attention scores (averaged over decoder layers).
    """
    def __init__(self,
                 loss_config: LossConfig,
                 name: str = C.ATTENTION_MONOTONICITY_LOSS,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=output_names, label_names=label_names)
        self.loss_config = loss_config



    def update(self, attention_losses):
        for attention_loss in attention_losses:
            (batch_size,) = attention_loss.shape
            loss = mx.nd.sum(attention_loss)
            self.num_inst += batch_size
            self.sum_metric += loss.asscalar()
                
                
class PoissonLoss(Loss):
    """
    Computes the Poisson regression loss.
    MSEMetric for this loss will be reporting the mean
    square error between lengths, not length ratios!

    :param loss_config: Loss configuration.
    """

    def __init__(self,
                 loss_config: LossConfig,
                 output_names: List[str], label_names: List[str],
                 name: str = C.LENRATIO_LOSS_NAME) -> None:
        super().__init__(loss_config=loss_config,
                         output_names=output_names, label_names=label_names)
        self.name = name

    def get_loss(self, pred: mx.sym.Symbol, labels: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Returns Poisson loss and output symbol given data and expected integers as labels.

        :param pred: Predictions. shape: (batch_size, 1).
        :param labels: Target integers. Shape: (batch_size,).
        :return: Loss symbol.
        """
        labels = mx.sym.reshape(labels, shape=(-1, 1))
        loss_value = pred - labels * mx.sym.log(mx.sym.maximum(1e-10, pred))
        # MakeLoss scales only the gradient, so scaling explicitly
        loss_value = self.loss_config.length_task_weight * loss_value
        loss_value = mx.sym.MakeLoss(data=loss_value,
                                     normalization='batch',
                                     name=self.name)
        return loss_value

    def create_metric(self) -> 'MSEMetric':
        return LengthRatioMSEMetric(name=C.LENRATIO_MSE,
                                    output_names=self.output_names,
                                    label_names=self.label_names)


class MSELoss(Loss):
    """
    Computes the Mean Squared Error loss.
    MSEMetric for this loss will be reporting the mea
    square error between length ratios.

    :param loss_config: Loss configuration.
    """

    def __init__(self,
                 loss_config: LossConfig,
                 output_names: List[str], label_names: List[str],
                 name: str = C.LENRATIO_LOSS_NAME) -> None:
        super().__init__(loss_config=loss_config,
                         output_names=output_names, label_names=label_names)
        self.name = name

    def get_loss(self, pred: mx.sym.Symbol, labels: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Returns MSE loss and output symbol given logits and expected integers as labels.

        :param pred: Predictions. Shape: (batch_size, 1).
        :param labels: Targets. Shape: (batch_size,).
        :return: Loss symbol.
        """
        labels = mx.sym.reshape(labels, shape=(-1, 1))
        loss_value = self.loss_config.length_task_weight / 2 * mx.sym.square(pred - labels)
        loss_value = mx.sym.MakeLoss(data=loss_value,
                                     normalization='batch',
                                     name=self.name)
        return loss_value

    def create_metric(self) -> 'MSEMetric':
        return LengthRatioMSEMetric(name=C.LENRATIO_MSE,
                                    output_names=self.output_names,
                                    label_names=self.label_names)


class MSEMetric(EvalMetric):
    """
    Version of the MSE metric that ignores padding tokens.

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param label_names: Name of labels that should be used when updating with update_dict.
    """

    def __init__(self,
                 name: str,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        """
        :param labels: List of (batch_size,)-shaped NDArrays.
        :param preds: List of (batch_size,1)-shaped NDArrays.
        """
        for label, pred in zip(labels, preds):
            batch_size = label.shape[0]
            # label: (batch_size, 1)
            label = label.as_in_context(pred.context).reshape((label.size,1))
            # mse: (batch_size,)
            mse = mx.nd.square(label - pred)
            # mse: (1,)
            mse = mx.nd.sum(mse)
            self.num_inst += batch_size

            self.sum_metric += mse.asscalar()


class LengthRatioMSEMetric(MSEMetric):
    """
    Version of the MSE metric specific to length ratio prediction, that
    looks for its labels in the network outputs instead of the iterator,
    as those are generated on the fly by the TrainingModel's sym_gen().

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param label_names: Name of labels that should be used when updating with update_dict.
    """

    def __init__(self,
                 name: str,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=output_names, label_names=label_names)

    def update_dict(self, label: Dict, pred: Dict):
        """
        If label is missing the right name, copy it from the prediction.
        """
        if not set(self.label_names).issubset(set(label.keys())):
            label.update({name:pred[name] for name in self.label_names})
        super().update_dict(label, pred)


    
    
    
