#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$PYTHONPATH:/workspace/slam/icefall_omni
set -eou pipefail

stage=2
stop_stage=2
# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: "


fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "stage 1: Download whisper-large-v2 multi-hans-zh fbank feature from huggingface"

  python3 local/compute_whisper_fbank.py
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "stage 2: "
  python3 ./slam_omni/decode.py \
    --max-duration 80 \
    --exp-dir slam_omni/exp_test_whisper_qwen2_1.5B \
    --speech-encoder-path-or-name models/whisper/v1.1/whisper-large-v2-multi-hans-zh-epoch-3-avg-10.pt  \
    --llm-path-or-name models/qwen \
    --epoch 999 --avg 1 \
    --manifest-dir data/fbank \
    --use-flash-attn True \
    --use-lora True

fi
