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

import mxnet as mx
import numpy as np
import pytest

import sockeye.constants as C
import sockeye.loss
import sockeye.model


def run_test_default_cross_entropy_loss(logits_np, labels_np, expected_softmax, expected_grads, label_smoothing,
                                        normalization_type=C.LOSS_NORM_BATCH):
    config = sockeye.loss.LossConfig(name=C.CROSS_ENTROPY,
                                     vocab_size=4,
                                     normalization_type=normalization_type,
                                     label_smoothing=label_smoothing)
    loss = sockeye.loss.get_loss(config)
    assert isinstance(loss, sockeye.loss.CrossEntropyLoss)

    logits = mx.sym.Variable("logits")
    labels = mx.sym.Variable("labels")
    sym = mx.sym.Group(loss.get_loss(logits, labels))

    assert sym.list_arguments() == ['logits', 'labels']
    assert sym.list_outputs() == [C.SOFTMAX_NAME + "_output"]

    _, out_shapes, _ = (sym.infer_shape(logits=logits_np.shape, labels=labels_np.shape))
    assert out_shapes[0] == logits_np.shape

    executor = sym.simple_bind(ctx=mx.cpu(),
                               logits=logits_np.shape,
                               labels=labels_np.shape)
    executor.arg_dict["logits"][:] = logits_np
    executor.arg_dict["labels"][:] = labels_np
    softmax = executor.forward(is_train=True)[0].asnumpy()
    assert np.isclose(softmax, expected_softmax).all()

    executor.backward()
    actual_grads = executor.grad_dict["logits"].asnumpy()

    assert np.isclose(actual_grads, expected_grads).all()
    label_grad_sum = executor.grad_dict["labels"].asnumpy().sum()
    assert label_grad_sum == 0

    return actual_grads


def run_test_custom_cross_entropy_loss(logits_np, labels_np, expected_arguments, expected_softmax,
                                       expected_loss, expected_grads, label_smoothing,
                                       normalization_type=C.LOSS_NORM_BATCH):
    config = sockeye.loss.LossConfig(name=C.CUSTOM_CROSS_ENTROPY,
                                     vocab_size=4,
                                     normalization_type=normalization_type,
                                     label_smoothing=label_smoothing)
    loss = sockeye.loss.get_loss(config)
    assert isinstance(loss, sockeye.loss.CustomCrossEntropyLoss)

    logits = mx.sym.Variable("logits")
    labels = mx.sym.Variable("labels")

    sym = mx.sym.Group(loss.get_loss(logits, labels))

    assert sym.list_arguments() == expected_arguments
    assert sym.list_outputs() == [C.CUSTOM_CROSS_ENTROPY + "_output", C.SOFTMAX_NAME + "_output"]

    _, out_shapes, _ = (sym.infer_shape(logits=logits_np.shape,
                                        labels=labels_np.shape))

    assert len(out_shapes) == 2
    assert out_shapes[0] == (4,)
    assert out_shapes[1] == logits_np.shape

    executor = sym.simple_bind(ctx=mx.cpu(),
                               logits=logits_np.shape,
                               labels=labels_np.shape)

    executor.arg_dict["logits"][:] = logits_np
    executor.arg_dict["labels"][:] = labels_np


    forward = executor.forward(is_train=True)

    actual_loss = forward[0].asnumpy()
    actual_softmax = forward[1].asnumpy()

    assert np.isclose(actual_loss, expected_loss).all()
    assert np.isclose(actual_softmax, expected_softmax).all()

    executor.backward()
    actual_grads = executor.grad_dict["logits"].asnumpy()

    assert np.isclose(actual_grads, expected_grads).all()

    label_grad_sum = executor.grad_dict["labels"].asnumpy().sum()
    assert label_grad_sum == 0

    return actual_grads, actual_loss, actual_softmax


def run_test_custom_weighted_cross_entropy_loss(logits_np, labels_np, weights_np, expected_arguments, expected_softmax,
                                                expected_loss, expected_grads, label_smoothing, use_instance_weight,
                                                normalization_type=C.LOSS_NORM_BATCH):
    config = sockeye.loss.LossConfig(name=C.WEIGHTED_CROSS_ENTROPY,
                                     vocab_size=4,
                                     normalization_type=normalization_type,
                                     use_instance_weight=use_instance_weight,
                                     label_smoothing=label_smoothing)
    loss = sockeye.loss.get_loss(config)
    assert isinstance(loss, sockeye.loss.WeightedCrossEntropyLoss)

    logits = mx.sym.Variable("logits")
    labels = mx.sym.Variable("labels")
    weights = mx.sym.Variable("weights")

    sym = mx.sym.Group(loss.get_loss(logits, labels, weights))

    assert sym.list_arguments() == expected_arguments
    assert sym.list_outputs() == [C.WEIGHTED_CROSS_ENTROPY + "_output", C.SOFTMAX_NAME + "_output"]

    if use_instance_weight:
        _, out_shapes, _ = (sym.infer_shape(logits=logits_np.shape,
                                            labels=labels_np.shape,
                                            weights=weights_np.shape))
    else:
        _, out_shapes, _ = (sym.infer_shape(logits=logits_np.shape,
                                            labels=labels_np.shape))

    assert len(out_shapes) == 2
    assert out_shapes[0] == (4,)
    assert out_shapes[1] == logits_np.shape

    executor = sym.simple_bind(ctx=mx.cpu(),
                               logits=logits_np.shape,
                               labels=labels_np.shape)

    executor.arg_dict["logits"][:] = logits_np
    executor.arg_dict["labels"][:] = labels_np

    if use_instance_weight:
        executor.arg_dict["weights"][:] = weights_np

    forward = executor.forward(is_train=True)

    actual_loss = forward[0].asnumpy()
    actual_softmax = forward[1].asnumpy()

    assert np.isclose(actual_loss, expected_loss).all()
    assert np.isclose(actual_softmax, expected_softmax).all()

    executor.backward()
    actual_grads = executor.grad_dict["logits"].asnumpy()

    assert np.isclose(actual_grads, expected_grads).all()

    label_grad_sum = executor.grad_dict["labels"].asnumpy().sum()
    assert label_grad_sum == 0

    return actual_grads, actual_loss, actual_softmax


@pytest.mark.parametrize("logits_np, labels_np, expected_softmax, expected_grads, label_smoothing",
                         [# without label smoothing
                          (mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]]),
                           mx.nd.array([1, 0, 2, 3]),
                           np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                   [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                   [0.25, 0.25, 0.25, 0.25],
                                   [0.25, 0.25, 0.25, 0.25]]),
                           np.asarray([[0.0320586, -0.91285568, 0.23688284, 0.64391428],
                                 [0., 0., 0., 0.],
                                 [0.25, 0.25, -0.75, 0.25],
                                 [0.25, 0.25, 0.25, -0.75]]),
                           0.0),
                          # with label smoothing
                          (mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]]),
                           mx.nd.array([1, 0, 2, 3]),
                           np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                       [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                       [0.25, 0.25, 0.25, 0.25],
                                       [0.25, 0.25, 0.25, 0.25]]),
                           np.asarray([[-0.13460806, -0.41285568, 0.07021617, 0.4772476],
                                       [0., 0., 0., 0.],
                                       [0.08333333, 0.08333333, -0.25, 0.08333333],
                                       [0.08333333, 0.08333333, 0.08333333, -0.25]]),
                           0.5)
                         ])
def test_default_cross_entropy_loss(logits_np, labels_np, expected_softmax, expected_grads, label_smoothing):
    run_test_default_cross_entropy_loss(logits_np, labels_np, expected_softmax, expected_grads, label_smoothing)


@pytest.mark.parametrize("logits_np, labels_np, expected_arguments, expected_softmax, "
                         "expected_loss, expected_grads, label_smoothing",
                         [# without label smoothing
                          (mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]]),
                           mx.nd.array([1, 0, 2, 3]),
                           ['labels', 'logits'],
                           np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                       [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                       [0.25, 0.25, 0.25, 0.25],
                                       [0.25, 0.25, 0.25, 0.25]]),
                           np.array([2.4401896, 0., 1.3862944, 1.3862944]),
                           np.asarray([[0.0320586, -0.91285568, 0.23688284, 0.64391428],
                                       [0., 0., 0., 0.],
                                       [0.25, 0.25, -0.75, 0.25],
                                       [0.25, 0.25, 0.25, -0.75]]),
                           0.0
                           ),
                          # with label smoothing
                          (mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]]),
                           mx.nd.array([1, 0, 2, 3]),
                           ['labels', 'logits'],
                           np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                       [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                       [0.25, 0.25, 0.25, 0.25],
                                       [0.25, 0.25, 0.25, 0.25]]),
                           np.array([2.10685635, 0., 1.3862944, 1.3862944]),
                           np.asarray([[-0.13460806, -0.41285568, 0.07021617, 0.4772476],
                                       [0., 0., 0., 0.],
                                       [0.08333333, 0.08333333, -0.25, 0.08333333],
                                       [0.08333333, 0.08333333, 0.08333333, -0.25]]),
                           0.5
                           )
                         ])
def test_custom_cross_entropy_loss(logits_np, labels_np, expected_arguments, expected_softmax,
                                   expected_loss, expected_grads, label_smoothing):
    run_test_custom_cross_entropy_loss(logits_np, labels_np, expected_arguments, expected_softmax,
                                   expected_loss, expected_grads, label_smoothing)


@pytest.mark.parametrize("logits_np, labels_np, weights_np, expected_arguments, expected_softmax, "
                         "expected_loss, expected_grads, label_smoothing, use_instance_weight",
                         [# without label smoothing, without weighting
                          (mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]]),
                           mx.nd.array([1, 0, 2, 3]),
                           mx.nd.array([1.0, 1.0, 1.0, 1.0]),
                           ['labels', 'logits'],
                           np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                       [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                       [0.25, 0.25, 0.25, 0.25],
                                       [0.25, 0.25, 0.25, 0.25]]),
                           np.array([2.4401896, 0., 1.3862944, 1.3862944]),
                           np.asarray([[0.0320586, -0.91285568, 0.23688284, 0.64391428],
                                       [0., 0., 0., 0.],
                                       [0.25, 0.25, -0.75, 0.25],
                                       [0.25, 0.25, 0.25, -0.75]]),
                           0.0,
                           False
                           ),
                          # with label smoothing, without weighting
                          (mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]]),
                           mx.nd.array([1, 0, 2, 3]),
                           mx.nd.array([1.0, 1.0, 1.0, 1.0]),
                           ['labels', 'logits'],
                           np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                       [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                       [0.25, 0.25, 0.25, 0.25],
                                       [0.25, 0.25, 0.25, 0.25]]),
                           np.array([2.10685635, 0., 1.3862944, 1.3862944]),
                           np.asarray([[-0.13460806, -0.41285568, 0.07021617, 0.4772476],
                                       [0., 0., 0., 0.],
                                       [0.08333333, 0.08333333, -0.25, 0.08333333],
                                       [0.08333333, 0.08333333, 0.08333333, -0.25]]),
                           0.5,
                           False
                           ),
                          # without label smoothing, with 1.0 weights
                          (mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]]),
                           mx.nd.array([1, 0, 2, 3]),
                           mx.nd.array([1.0, 1.0, 1.0, 1.0]),
                           ['labels', 'logits', 'weights'],
                           np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                       [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                       [0.25, 0.25, 0.25, 0.25],
                                       [0.25, 0.25, 0.25, 0.25]]),
                           np.array([2.4401896, 0., 1.3862944, 1.3862944]),
                           np.asarray([[0.0320586, -0.91285568, 0.23688284, 0.64391428],
                                       [0., 0., 0., 0.],
                                       [0.25, 0.25, -0.75, 0.25],
                                       [0.25, 0.25, 0.25, -0.75]]),
                           0.0,
                           True
                           ),
                          # with label smoothing, with 1.0 weights
                          (mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]]),
                           mx.nd.array([1, 0, 2, 3]),
                           mx.nd.array([1.0, 1.0, 1.0, 1.0]),
                           ['labels', 'logits', 'weights'],
                           np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                       [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                       [0.25, 0.25, 0.25, 0.25],
                                       [0.25, 0.25, 0.25, 0.25]]),
                           np.array([2.10685635, 0., 1.3862944, 1.3862944]),
                           np.asarray([[-0.13460806, -0.41285568, 0.07021617, 0.4772476],
                                       [0., 0., 0., 0.],
                                       [0.08333333, 0.08333333, -0.25, 0.08333333],
                                       [0.08333333, 0.08333333, 0.08333333, -0.25]]),
                           0.5,
                           True
                           ),
                          # without label smoothing, with actual weights
                          (mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]]),
                           mx.nd.array([1, 0, 2, 3]),
                           mx.nd.array([1.0, 0.3, 0.0, 0.7]),
                           ['labels', 'logits', 'weights'],
                           np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                       [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                       [0.25, 0.25, 0.25, 0.25],
                                       [0.25, 0.25, 0.25, 0.25]]),
                           np.array([2.4401896, 0., 0., 0.97040606]),
                           np.asarray([[0.0320586, -0.9128557, 0.23688284, 0.6439143],
                                 [0., 0., 0., 0.],
                                 [0., 0., 0., 0.],
                                 [0.175, 0.175, 0.175, -0.525]]),
                           0.0,
                           True
                           ),
                          # with label smoothing, with actual weights
                          (mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]]),
                           mx.nd.array([1, 0, 2, 3]),
                           mx.nd.array([1.0, 0.3, 0.0, 0.7]),
                           ['labels', 'logits', 'weights'],
                           np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                       [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                       [0.25, 0.25, 0.25, 0.25],
                                       [0.25, 0.25, 0.25, 0.25]]),
                           np.array([2.1068563 , 0., 0., 0.97040606]),
                           np.asarray([[-0.13460806, -0.41285568, 0.07021617, 0.4772476],
                                 [-0., -0., -0., -0.],
                                 [-0., -0., -0., -0.],
                                 [0.05833333, 0.05833333, 0.05833333, -0.175]]),
                           0.5,
                           True
                           )
                         ])
def test_custom_weighted_cross_entropy_loss(logits_np, labels_np, weights_np, expected_arguments, expected_softmax,
                                   expected_loss, expected_grads, label_smoothing, use_instance_weight):
    run_test_custom_weighted_cross_entropy_loss(logits_np, labels_np, weights_np, expected_arguments, expected_softmax,
                                   expected_loss, expected_grads, label_smoothing, use_instance_weight)


def test_default_custom_losses_batch_normalization_equal():

    logits_np = mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]])

    labels_np = mx.nd.array([1, 0, 2, 3])

    expected_softmax = np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                   [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                   [0.25, 0.25, 0.25, 0.25],
                                   [0.25, 0.25, 0.25, 0.25]])

    expected_loss = np.array([2.10685635, 0., 1.3862944, 1.3862944])

    expected_grads = np.asarray([[-0.13460806, -0.41285568, 0.07021617, 0.4772476],
                                 [0., 0., 0., 0.],
                                 [0.08333333, 0.08333333, -0.25, 0.08333333],
                                 [0.08333333, 0.08333333, 0.08333333, -0.25]])

    label_smoothing = 0.5

    default_grads = run_test_default_cross_entropy_loss(logits_np, labels_np, expected_softmax, expected_grads, label_smoothing)

    custom_grads, custom_loss, custom_softmax = \
        run_test_custom_cross_entropy_loss(
            logits_np, labels_np,
            ['labels', 'logits'], expected_softmax,
            expected_loss, expected_grads,
            label_smoothing)

    assert np.isclose(default_grads, custom_grads).all()


def test_default_custom_losses_valid_normalization_equal():

    logits_np = mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]])

    labels_np = mx.nd.array([1, 0, 2, 3])

    expected_softmax = np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                   [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                   [0.25, 0.25, 0.25, 0.25],
                                   [0.25, 0.25, 0.25, 0.25]])

    # loss is different since we do valid normalization in the forward pass for loss,
    # while default only does it in backward pass for gradients
    expected_loss_default = np.array([2.10685635, 0., 1.3862944, 1.3862944])

    expected_loss_custom = expected_loss_default / np.sum(labels_np.asnumpy() != 0)

    expected_grads = np.asarray([[-0.04486936, -0.13761857, 0.02340539, 0.15908253],
                                 [ 0., 0., 0., 0.],
                                 [ 0.02777778, 0.02777778, -0.08333334, 0.02777778],
                                 [ 0.02777778, 0.02777778, 0.02777778, -0.08333334]])

    label_smoothing = 0.5

    default_grads = run_test_default_cross_entropy_loss(logits_np, labels_np, expected_softmax, expected_grads, label_smoothing,
                                                        C.LOSS_NORM_VALID)

    custom_grads, custom_loss, custom_softmax = \
        run_test_custom_cross_entropy_loss(
            logits_np, labels_np,
            ['labels', 'logits'], expected_softmax,
            expected_loss_custom, expected_grads,
            label_smoothing, C.LOSS_NORM_VALID)

    assert np.isclose(default_grads, custom_grads).all()


def test_default_custom_weighted_losses_batch_normalization_equal():

    logits_np = mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]])

    labels_np = mx.nd.array([1, 0, 2, 3])

    weights_np = mx.nd.array([1.0, 1.0, 1.0, 1.0])

    expected_softmax = np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                   [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                   [0.25, 0.25, 0.25, 0.25],
                                   [0.25, 0.25, 0.25, 0.25]])

    expected_loss = np.array([2.10685635, 0., 1.3862944, 1.3862944])

    expected_grads = np.asarray([[-0.13460806, -0.41285568, 0.07021617, 0.4772476],
                                 [0., 0., 0., 0.],
                                 [0.08333333, 0.08333333, -0.25, 0.08333333],
                                 [0.08333333, 0.08333333, 0.08333333, -0.25]])

    label_smoothing = 0.5

    default_grads = run_test_default_cross_entropy_loss(logits_np, labels_np, expected_softmax, expected_grads, label_smoothing)

    # do not use weights

    custom_weight_false_grads, custom_weight_false_loss, custom_weight_false_softmax = \
        run_test_custom_weighted_cross_entropy_loss(
            logits_np, labels_np, weights_np,
            ['labels', 'logits'], expected_softmax,
            expected_loss, expected_grads,
            label_smoothing, False)

    assert np.isclose(default_grads, custom_weight_false_grads).all()

    # set all weights to 1.0

    custom_weight_1_0_grads, custom_weight_1_0_loss, custom_weight_1_0_softmax = \
        run_test_custom_weighted_cross_entropy_loss(
            logits_np, labels_np, weights_np,
            ['labels', 'logits', 'weights'], expected_softmax,
            expected_loss, expected_grads,
            label_smoothing, True)

    assert np.isclose(default_grads, custom_weight_1_0_grads).all()

    assert np.isclose(custom_weight_false_loss, custom_weight_1_0_loss).all()
    assert np.isclose(custom_weight_false_softmax, custom_weight_1_0_softmax).all()


def test_default_custom_weighted_losses_valid_normalization_equal():

    logits_np = mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]])

    labels_np = mx.nd.array([1, 0, 2, 3])

    weights_np = mx.nd.array([1.0, 1.0, 1.0, 1.0])

    expected_softmax = np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                   [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                   [0.25, 0.25, 0.25, 0.25],
                                   [0.25, 0.25, 0.25, 0.25]])

    # loss is different since we do valid normalization in the forward pass for loss,
    # while default only does it in backward pass for gradients
    expected_loss_default = np.array([2.10685635, 0., 1.3862944, 1.3862944])

    expected_loss_custom = expected_loss_default / np.sum(labels_np.asnumpy() != 0)

    expected_grads = np.asarray([[-0.04486936, -0.13761857, 0.02340539, 0.15908253],
                                 [ 0., 0., 0., 0.],
                                 [ 0.02777778, 0.02777778, -0.08333334, 0.02777778],
                                 [ 0.02777778, 0.02777778, 0.02777778, -0.08333334]])

    label_smoothing = 0.5

    default_grads = run_test_default_cross_entropy_loss(logits_np, labels_np, expected_softmax, expected_grads, label_smoothing,
                                                        C.LOSS_NORM_VALID)

    # do not use weights

    custom_weight_false_grads, custom_weight_false_loss, custom_weight_false_softmax = \
        run_test_custom_weighted_cross_entropy_loss(
            logits_np, labels_np, weights_np,
            ['labels', 'logits'], expected_softmax,
            expected_loss_custom, expected_grads,
            label_smoothing, False, C.LOSS_NORM_VALID)

    assert np.isclose(default_grads, custom_weight_false_grads).all()

    # set all weights to 1.0

    custom_weight_1_0_grads, custom_weight_1_0_loss, custom_weight_1_0_softmax = \
        run_test_custom_weighted_cross_entropy_loss(
            logits_np, labels_np, weights_np,
            ['labels', 'logits', 'weights'], expected_softmax,
            expected_loss_custom, expected_grads,
            label_smoothing, True, C.LOSS_NORM_VALID)

    assert np.isclose(default_grads, custom_weight_1_0_grads).all()

    assert np.isclose(custom_weight_false_loss, custom_weight_1_0_loss).all()
    assert np.isclose(custom_weight_false_softmax, custom_weight_1_0_softmax).all()


@pytest.mark.parametrize("preds, labels, normalization_type, label_smoothing, expected_value",
                         [(mx.nd.array([[0.0, 0.2, 0.8],
                                        [0.0, 1.0, 0.0]]),
                           mx.nd.array([[2],
                                        [0]]),
                           'valid',
                           0.0,
                           -np.log(0.8 + 1e-8) / 1.0),  # pylint: disable=invalid-unary-operand-type
                          (mx.nd.array([[0.0, 0.2, 0.8],
                                        [0.0, 1.0, 0.0]]),
                           mx.nd.array([[2],
                                        [0]]),
                           'batch',
                           0.0,
                           -np.log(0.8 + 1e-8) / 2.0)]  # pylint: disable=invalid-unary-operand-type
                         )
def test_cross_entropy_metric(preds, labels, normalization_type, label_smoothing, expected_value):
    config = sockeye.loss.LossConfig(name=C.CROSS_ENTROPY,
                                     vocab_size=preds.shape[1],
                                     normalization_type=normalization_type,
                                     label_smoothing=label_smoothing)
    metric = sockeye.loss.CrossEntropyMetric(config)
    metric.update([labels], [preds])
    name, value = metric.get()
    assert name == 'cross-entropy'
    assert np.isclose(value, expected_value)


def test_cross_entropy_internal():
    pred = mx.nd.array([[0.0, 0.2, 0.8]])
    logprob = mx.nd.log(pred + 1e-8)
    label = mx.nd.array([2])
    expected_cross_entropy = -np.log(0.8 + 1e-8) / 1.0  # pylint: disable=invalid-unary-operand-type

    cross_entropy = sockeye.loss.CrossEntropyMetric.cross_entropy(logprob, label).sum()
    cross_entropy_smoothed = sockeye.loss.CrossEntropyMetric.cross_entropy_smoothed(logprob, label,
                                                                                    alpha=0.0, num_classes=3).sum()

    assert np.isclose(cross_entropy.asnumpy(), expected_cross_entropy)
    assert np.isclose(cross_entropy_smoothed.asnumpy(), expected_cross_entropy)
