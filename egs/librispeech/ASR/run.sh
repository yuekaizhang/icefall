#!/usr/bin/bash

stage=1
stop_stage=1


export CUDA_VISIBLE_DEVICES="1"


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
      --epoch 28 \
      --avg 15 \
      --exp-dir ./pruned_transducer_stateless7_ctc/exp \
      --max-duration 600 \
      --decoding-method greedy_search

fi