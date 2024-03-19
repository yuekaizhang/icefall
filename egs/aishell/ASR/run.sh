
export PYTHONPATH=$PYTHONPATH:/workspace/icefall
pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html
pip install -r whisper/requirements.txt

torchrun --nproc_per_node 8 ./whisper/train.py \
  --max-duration 200 \
  --exp-dir whisper/exp_large_v2 \
  --model-name large-v2 \
  --manifest-dir data/fbank \
  --deepspeed \
  --deepspeed_config ./whisper/ds_config_zero1.json

# python3 ./whisper/decode.py \
#   --exp-dir whisper/exp_large_v2_sft \
#   --model-name large-v2 \
#   --epoch 999 --avg 1 \
#   --start-index 0 --end-index 26 \
#   --remove-whisper-encoder-input-length-restriction True \
#   --manifest-dir data/fbank \
#   --beam-size 1 --max-duration 50