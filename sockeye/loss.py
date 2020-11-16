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
                 length_task_weight: float = 1.0,
                 en_trg_id: Optional[int] = None,
                 non_en_id: Optional[int] = None,
                 separator_id: Optional[int] = None,
                 margin: Optional[float] = None,
                 monotonicity_on_heads: Optional[int] = None) -> None:
        super().__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.normalization_type = normalization_type
        self.label_smoothing = label_smoothing
        self.length_task_link = length_task_link
        self.length_task_weight = length_task_weight
        self.en_trg_id = en_trg_id
        self.non_en_id = non_en_id
        self.separator_id = separator_id
        self.margin = margin
        self.monotonicity_on_heads = monotonicity_on_heads


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
        return MonotoneAttention(config)
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

class MonotoneAttention(Loss):
    """
    Computes the attention monotonicity loss.

    :param loss_config: Loss configuration.
    TODO: change this -> use attention on positions
    """

    def __init__(self, loss_config: LossConfig) -> None:
        logger.info("Loss: AttentionMonotonicity")
        self.loss_config = loss_config

    def get_loss(self,
                 attention_scores_list: List[mx.sym.Symbol],
                 positional_attention: mx.sym.Symbol,
                 num_attention_heads: int,
                 target_words: mx.sym.Symbol,
                 source_words: mx.sym.Symbol,
                 source_length: mx.sym.Symbol,
                 target_length: mx.sym.Symbol,
                 grad_scale: Optional[float] = 0.5,
                 margin: Optional[float] = 1.0,
                 monotonicity_on_heads: Optional[Tuple[int, int]] = None,
                 monotonicity_on_layers: Optional[Tuple[int, int]] = None,
                 absolute_positions: Optional[bool] = False,
                 monotonicity_loss_double_normalize: Optional[bool] = False) -> List[mx.sym.Symbol]:


        total_loss = mx.sym.zeros_like(target_length)
        start = 0
        end = len(attention_scores_list)
        if monotonicity_on_layers is not None:
            start, end = monotonicity_on_layers
            start -= 1
            if end > len(attention_scores_list):
                logger.error("Cannot calculate loss on layer {} in a model with {} decoder layers.".format(end, len(attention_scores_list)))
                exit(1)

        for layer in range(start, end):
            loss = self.monotonicity_score_per_layer(attention_scores_list[layer], positional_attention, num_attention_heads, target_words, source_words, source_length, target_length, margin, monotonicity_on_heads, absolute_positions, monotonicity_loss_double_normalize)
            total_loss = mx.sym.broadcast_add(total_loss, loss)

        ## average layer loss
        num_layers = end-start
        num_layers = mx.sym.ones_like(total_loss) * num_layers
        avg_loss = mx.sym.broadcast_div(total_loss, num_layers, name="_mono_loss_broad_div3")
        return mx.sym.MakeLoss(avg_loss,
                                grad_scale=grad_scale)


    def monotonicity_score_per_layer(self,
                                     attention_scores: mx.sym.Symbol,
                                     positional_attention: mx.sym.Symbol,
                                     num_attention_heads: int,
                                     target_words: mx.sym.Symbol,
                                     source_words: mx.sym.Symbol,
                                     source_length: mx.sym.Symbol,
                                     target_length: mx.sym.Symbol,
                                     margin: float,
                                     monotonicity_on_heads: Tuple[int, int],
                                     absolute_positions: bool,
                                     monotonicity_loss_double_normalize: bool):
        """
        :param attention_scores: decoder-encoder attention scores (MultiHeadAttention). Shape (batch_size * attention_heads, target_length, source_length)
        :param positional_attention: Attention from encoder layer hidden states (after self-attention) to positions. Shape (batch, src_len, pos_len=src_len)
        :param target_words: target words, used to remove padding. Shape (batch_size * target_length)
        :param source_words: source sentences. Symbol of shape(batch, src_len, num_factors).
        :param source_length: lengths of source sentences in batch (without padding).
        :param target_lengths: lengths of target sentences in batch (without padding).
        :param margin: margin for increase in attention to be considered monotone.
        :param monotonicity_on_heads: apply monotonicity loss only to n-m heads (only applicable with multi-head attention).
        :param absolute_positions:
        """

        # take average of attention_heads on each position
        ## TODO: implement for --rnn-attention-mhdot-heads
        ## default: calculate only on first head (rnn)
        start = 0
        end =1
        if num_attention_heads >1:
            attention_scores = attention_scores.reshape(shape=(-4, -1, num_attention_heads, -2), name="_mono_loss_reshape1") # (batch_size, attention_heads, target_length, source_length)
            ## default: on all transformer heads
            end = num_attention_heads
            if monotonicity_on_heads is not None:
                start, end = monotonicity_on_heads
                start = start-1
                if end > num_attention_heads:
                    logger.error("Cannot use loss on head {} with num_attention_heads {}".format(end, num_attention_heads))
                    exit(1)

        ## take dot product of positional attention and actual positions
        source_positions = mx.contrib.sym.arange_like(data=source_words, start=1, axis=1, name="_mono_loss_arange_like1") # (src_len,), needs mxnet-1.6!

        # TODO only works with absolute positions
        # if prefix should be ignored: mask everything before and including separator token out for loss
        if hasattr(self.loss_config, "separator_id") and self.loss_config.separator_id is not None:
            # get source tokens without factor dimension
            # TODO does not work with multiple factors
            source_tokens = source_words.squeeze() # (batch, src_len)

            # lookup positions of <sep>
            source_token_positions = mx.sym.broadcast_mul(mx.sym.ones_like(source_tokens), source_positions, name="mono_loss_broad_mul_sep") # (batch, src_len)
            max_token_positions = mx.sym.broadcast_mul(mx.sym.ones_like(source_token_positions), mx.sym.max(source_token_positions)) # (batch, src_len)
            match_indices = mx.sym.where(condition=(source_tokens == self.loss_config.separator_id),
                                         x=source_token_positions,
                                         y=max_token_positions) # (batch, src_len)
            separator_ids = mx.sym.min(match_indices, axis=1, name="mono_loss_broad_min_sep") # (batch,)

            # update source length not to include tags : used to compute source-target length ratio
            source_length = source_length - separator_ids # (batch,)


        ## calculate loss separately for each head, then take mean of loss
        layer_loss = mx.sym.zeros_like(target_length) ## (batch_size, )

        for i in range(start, end):
            if num_attention_heads > 1:
                sliced_attention_scores = mx.sym.slice_axis(attention_scores, axis=1, begin=i, end=(i+1)).squeeze()
            else:
                sliced_attention_scores = attention_scores

            if absolute_positions:
                positions = mx.sym.broadcast_mul(mx.sym.ones_like(sliced_attention_scores), source_positions, name="_mono_loss_broad_mul1")
            else:
                weighted_source_positions = mx.sym.dot(positional_attention, source_positions, name="_mono_loss_broad_dot1") # (batch, src_len)
                weighted_source_positions = weighted_source_positions.expand_dims(axis=1, name="_mono_loss_broad_expand1") # (batch, 1, src_len)
                p = mx.sym.ones_like(sliced_attention_scores) # (batch_size, target_length, source_length)
                positions = mx.sym.broadcast_mul(p, weighted_source_positions, name="_mono_loss_broad_mul2") # shape(batch, target_length, source_length), values in source_length = arange(source_length) * positional_attention


            # TODO only works with absolute positions
            # if prefix should be ignored: mask everything before and including separator token out for loss
            if hasattr(self.loss_config, "separator_id") and self.loss_config.separator_id is not None:
                # update positions to be 0 for <sep> token
                separator_pos = separator_ids.expand_dims(axis=1, name="mono_loss_exp_sep").expand_dims(axis=2, name="mono_loss_exp_sep")  # (batch, 1, 1)
                separator_pos = mx.sym.broadcast_mul(mx.sym.ones_like(positions), separator_pos, name="mono_loss_broad_mul_sep2") # (batch, trg_len, src_len)
                positions = mx.sym.where(condition=(positions > separator_pos),
                                         x=positions,
                                         y=mx.sym.zeros_like(positions)) # (batch, trg_len, src_len)

            ## no need to remove padding from source, padded positions are 0 in attention scores
            positionally_weighted_attention = mx.sym.broadcast_mul(sliced_attention_scores, positions, name="_mono_loss_broad_mul3") # shape(batch_size, target_length, source_length (attention_score*position))
            # take average over sequences
            avg = mx.sym.sum(positionally_weighted_attention, axis=2, name="_mono_loss_broad_sum1") # shape (batch, target_length)

            #### set padded positions in target to zero (we dont care about alignment scores from padded tokens)
            mask = (target_words != C.PAD_ID) # target_words (batch_size, target_length), mask: 0 where padded, 1 otherwise
            valid_t = mask.reshape_like(avg)
            avg = mx.sym.broadcast_mul(avg, valid_t, name="_mono_loss_broad_mul4")

            shifted_avg = mx.sym.slice_axis(avg, axis=-1, begin=1, end=None, name="_mono_loss_broad_slice1") # (batch_size, target_length-1)
            padding = mx.sym.slice_axis(mx.sym.zeros_like(shifted_avg), axis=-1, begin=0, end=1, name="_mono_loss_broad_slice2") # (batch_size, 1)
            shifted_avg = mx.sym.concat(shifted_avg, padding, dim=-1, name="_mono_loss_broad_concat1") # (batch_size, target_length)

            ## in shifted avg: one more padded position (since shifted to left), create new mask and apply to adjacent_pos_difference
            shifted_mask = (shifted_avg != 0)
            ## margin: scale with length difference between source and target: margin *  |x|/|y|
            s_t_length_ratio = mx.sym.broadcast_div(source_length, target_length) # source to target length ratios (no padding)
            scaled_margin = (margin * s_t_length_ratio).expand_dims(axis=1) ## (batch, 1)
            trg_len_ones = mx.sym.ones_like(shifted_avg).slice_axis(axis=0, begin=0, end=1) ## (1, trg_len)
            scaled_margin = mx.sym.broadcast_mul(scaled_margin, trg_len_ones, name="_mono_loss_broad_mul5")
            shifted_avg = shifted_avg - scaled_margin
            adjacent_pos_difference = avg - shifted_avg # (batch, target_length)
            adjacent_pos_difference = adjacent_pos_difference * shifted_mask

            ## if learned reordered positions: mask zero-shot pairs for loss
            if not absolute_positions:
                ## set loss for non-English pairs to 0
                # if target and source language is not English, set loss == 0
                #print("non en id {}, en trg id {}".format(self.loss_config.non_en_id, self.loss_config.en_trg_id))
                en_trg_mask = (source_words == self.loss_config.en_trg_id) # (batch, src_len), 1 if <2en> present, 0 if <2en> not present
                en_trg_mask = mx.sym.sum(en_trg_mask, axis=1)
                non_en_trg_mask = (en_trg_mask == 0) # (batch, src_len), 0 if <2en> present, 1 if <2en> not present
                en_src_mask = (source_words == self.loss_config.non_en_id) # (batch, src_len), 1 if <non_en> present, 0 if not <non_en> present
                non_en_src_mask = mx.sym.sum(en_src_mask, axis=1) # (batch,)
                mask_sum = non_en_trg_mask + non_en_src_mask
                non_en_pairs_mask = mx.sym.broadcast_equal(mask_sum, (mx.sym.ones(shape=(1)) *2 ) ) ## 1 where sum was 2, i.e. src + trg not English, 0 otherwise
                non_en_pairs_mask_inv = (non_en_pairs_mask == 0)
                non_en_pairs_mask_inv = mx.sym.expand_dims(non_en_pairs_mask_inv, axis=1)
                adjacent_pos_difference =  mx.sym.broadcast_mul(non_en_pairs_mask_inv, adjacent_pos_difference, name="_mono_loss_broad_mul6")

            # loss= max(0, avg(y)-avg(y+1)), if y-(y+1) >0, this is the loss, else if y-(y+1) < 0, loss=0
            # with margin: loss= max(0, (avg(y)-avg(y+1))+margin ), if (y-(y+1))+margin >0, this is the loss, else if (y-(y+1))+margin < 0, loss=0
            adjacent_pos_difference = mx.sym.broadcast_maximum(lhs=mx.sym.zeros_like(adjacent_pos_difference), rhs=adjacent_pos_difference, name="_mono_loss_broad_max")

            head_loss = mx.sym.sum(adjacent_pos_difference, axis=1, name="_mono_loss_broad_sum2") # (batch, )

            # normalize by valid tokens in target
            ## add epsilon to num_valid_positions positions to avoid div by zero (can happen with short sequences due to dropout)
            epsilon = 1e-8
            target_length = target_length + epsilon
            head_loss = mx.sym.broadcast_div(head_loss, target_length, name="_mono_loss_broad_div")
            if monotonicity_loss_double_normalize:
                source_length = source_length +epsilon
                head_loss = mx.sym.broadcast_div(head_loss, source_length, name="_mono_loss_broad_div2")
            layer_loss = mx.sym.broadcast_add(layer_loss, head_loss, name="_mono_loss_broad_add")

        heads = end-start
        heads =  mx.sym.ones_like(layer_loss) * heads
        layer_loss = mx.sym.broadcast_div(layer_loss, heads, name="_mono_loss_broad_div2")
        return layer_loss

    def create_metric(self) -> "MonotoneAttentionMetric":
        return MonotoneAttentionMetric(self.loss_config)


class MonotoneAttentionMetric(EvalMetric):
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
