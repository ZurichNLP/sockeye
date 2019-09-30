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

import pytest

from test.common import run_train_translate, tmp_digits_dataset

_TRAIN_LINE_COUNT = 100
_DEV_LINE_COUNT = 10
_TEST_LINE_COUNT = 10
_TEST_LINE_COUNT_EMPTY = 2
_LINE_MAX_LENGTH = 9
_TEST_MAX_LENGTH = 20

ENCODER_DECODER_SETTINGS = [
    # Bilingual reconstruction - "vanilla" LSTM encoder-decoder with attention
    ("--reconstruction bilingual --encoder rnn --decoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 16"
     " --num-embed 16 --rnn-attention-type mlp --rnn-attention-num-hidden 16 --batch-size 13 --loss cross-entropy"
     " --optimized-metric perplexity --max-updates 10 --checkpoint-frequency 10 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 2",
     True, False, False),
    # Bilingual reconstruction - "kitchen sink" LSTM encoder-decoder with attention
    ("--reconstruction bilingual --encoder rnn --decoder rnn --num-layers 4:2 --rnn-cell-type lstm --rnn-num-hidden 16"
     " --rnn-residual-connections"
     " --num-embed 16 --rnn-attention-type coverage --rnn-attention-num-hidden 16"
     " --rnn-attention-use-prev-word --rnn-context-gating --layer-normalization --batch-size 13"
     " --loss cross-entropy --label-smoothing 0.1 --loss-normalization-type batch --optimized-metric perplexity"
     " --max-updates 10 --checkpoint-frequency 10 --optimizer adam --initial-learning-rate 0.01"
     " --rnn-dropout-inputs 0.5:0.1 --rnn-dropout-states 0.5:0.1 --embed-dropout 0.1 --rnn-decoder-hidden-dropout 0.01"
     " --rnn-decoder-state-init avg --rnn-encoder-reverse-input --rnn-dropout-recurrent 0.1:0.0"
     " --rnn-h2h-init orthogonal_stacked"
     " --learning-rate-decay-param-reset --weight-normalization",
     "--beam-size 2",
     False, False, False),
    # Bilingual reconstruction with instantiated hidden states - "vanilla" LSTM encoder-decoder with attention
    ("--reconstruction bilingual --instantiate-hidden gumbel-softmax --encoder rnn --decoder rnn --num-layers 1"
     " --rnn-cell-type lstm --rnn-num-hidden 16 --num-embed 16 --rnn-attention-type mlp --rnn-attention-num-hidden 16"
     " --batch-size 13 --loss cross-entropy --optimized-metric perplexity --max-updates 10"
     " --checkpoint-frequency 10 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 2",
     True, False, False),
    # Bilingual reconstruction with instantiated hidden states - "kitchen sink" LSTM encoder-decoder with attention
    ("--reconstruction bilingual --instantiate-hidden gumbel-softmax --softmax-temperature 5.0 --encoder rnn"
     " --decoder rnn --num-layers 4:2 --rnn-cell-type lstm --rnn-num-hidden 16 --rnn-residual-connections"
     " --num-embed 16 --rnn-attention-type coverage --rnn-attention-num-hidden 16"
     " --rnn-attention-use-prev-word --rnn-context-gating --layer-normalization --batch-size 13"
     " --loss cross-entropy --label-smoothing 0.1 --loss-normalization-type batch --optimized-metric perplexity"
     " --max-updates 10 --checkpoint-frequency 10 --optimizer adam --initial-learning-rate 0.01"
     " --rnn-dropout-inputs 0.5:0.1 --rnn-dropout-states 0.5:0.1 --embed-dropout 0.1 --rnn-decoder-hidden-dropout 0.01"
     " --rnn-decoder-state-init avg --rnn-encoder-reverse-input --rnn-dropout-recurrent 0.1:0.0"
     " --rnn-h2h-init orthogonal_stacked"
     " --learning-rate-decay-param-reset --weight-normalization",
     "--beam-size 2",
     False, False, False),
    #
    # Transformer reconstruction
    # Bilingual reconstruction - "vanilla" Transformer encoder-decoder
    ("--reconstruction bilingual --encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg --embed-weight-init=normal"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 2",
     True, False, False),
    # Bilingual reconstruction - "vanilla" Transformer encoder-decoder
    ("--reconstruction bilingual --encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg --embed-weight-init=normal"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 2",
     False, False, False),
    # Bilingual reconstruction with instantiated hidden states - "vanilla" Transformer encoder-decoder
    ("--reconstruction bilingual --instantiate-hidden gumbel-softmax --encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg --embed-weight-init=normal"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 2",
     True, False, False),
    # Bilingual reconstruction with instantiated hidden states - "vanilla" Transformer encoder-decoder
    ("--reconstruction bilingual --instantiate-hidden gumbel-softmax --encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg --embed-weight-init=normal"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01"
     " --metrics perplexity perplexity-reconstruction --optimized-metric perplexity",
     "--beam-size 2",
     False, False, False),]


@pytest.mark.parametrize("train_params, translate_params, restrict_lexicon, use_prepared_data, use_source_factors",
                         ENCODER_DECODER_SETTINGS)
def test_reconstruction(train_params: str,
                        translate_params: str,
                        restrict_lexicon: bool,
                        use_prepared_data: bool,
                        use_source_factors: bool):
    """Task: copy short sequences of digits"""

    with tmp_digits_dataset(prefix="test_reconstruction_train_translate",
                            train_line_count=_TRAIN_LINE_COUNT,
                            train_max_length=_LINE_MAX_LENGTH,
                            dev_line_count=_DEV_LINE_COUNT,
                            dev_max_length=_LINE_MAX_LENGTH,
                            test_line_count=_TEST_LINE_COUNT,
                            test_line_count_empty=_TEST_LINE_COUNT_EMPTY,
                            test_max_length=_TEST_MAX_LENGTH,
                            sort_target=False) as data:

        # make target the same as source
        data['target'] = data['source']
        # Test model configuration, including the output equivalence of batch and no-batch decoding
        translate_params_batch = translate_params + " --batch-size 2"

        # When using source factors
        train_source_factor_paths, dev_source_factor_paths, test_source_factor_paths = None, None, None
        if use_source_factors:
            train_source_factor_paths = [data['source']]
            dev_source_factor_paths = [data['validation_source']]
            test_source_factor_paths = [data['test_source']]

        # Ignore return values (perplexity and BLEU) for integration test
        run_train_translate(train_params=train_params,
                            translate_params=translate_params,
                            translate_params_equiv=translate_params_batch,
                            train_source_path=data['source'],
                            train_target_path=data['source'],
                            dev_source_path=data['validation_source'],
                            dev_target_path=data['validation_target'],
                            test_source_path=data['test_source'],
                            test_target_path=data['test_target'],
                            train_source_factor_paths=train_source_factor_paths,
                            dev_source_factor_paths=dev_source_factor_paths,
                            test_source_factor_paths=test_source_factor_paths,
                            max_seq_len=_LINE_MAX_LENGTH + 1,
                            restrict_lexicon=restrict_lexicon,
                            work_dir=data['work_dir'],
                            use_prepared_data=use_prepared_data)
