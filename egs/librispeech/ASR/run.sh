#!/usr/bin/bash

stage=4
stop_stage=4


export CUDA_VISIBLE_DEVICES="0"


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  python3 ./pruned_transducer_stateless7_ctc_attention/train.py \
    --world-size 1 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir pruned_transducer_stateless7_ctc/exp \
    --full-libri 0 \
    --use-fp16 1 \
    --num-attention-decoder-layers 0 \
    --max-duration 500
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

  python3 ./pruned_transducer_stateless7_ctc_attention/decode.py \
      --epoch 30 \
      --avg 9 \
      --exp-dir ./pruned_transducer_stateless7_ctc_attention/exp \
      --max-duration 600 \
      --decoding-method greedy_search

fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

  python3 ./pruned_transducer_stateless7_ctc_attention/ctc_decode.py \
      --epoch 999 \
      --avg 1 \
      --use-averaged-model False \
      --exp-dir ./pruned_transducer_stateless7_ctc_attention/exp2 \
      --max-duration 300 \
      --decoding-method "ctc-decoding-attention-decoder"

fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then

  python3 ./pruned_transducer_stateless7_ctc_attention/decode.py \
      --epoch 999 \
      --avg 1 \
      --use-averaged-model False \
      --exp-dir ./pruned_transducer_stateless7_ctc_attention/exp2 \
      --max-duration 600 \
      --decoding-method greedy_search_with_ctc

fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
./pruned_transducer_stateless7_ctc_attention/decode.py \
    --epoch 999 \
    --avg 1 \
    --use-averaged-model False \
    --exp-dir ./pruned_transducer_stateless7_ctc_attention/exp2 \
    --max-duration 300 \
    --decoding-method fast_beam_search_nbest_attention_decoder \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 100 \
    --nbest-scale 0.5
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
./pruned_transducer_stateless7_ctc_attention/decode.py \
    --epoch 999 \
    --avg 1 \
    --use-averaged-model False \
    --exp-dir ./pruned_transducer_stateless7_ctc_attention/exp2 \
    --max-duration 300 \
    --decoding-method fast_beam_search_nbest_ctc_rescoring\
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 100 \
    --nbest-scale 0.5
fi