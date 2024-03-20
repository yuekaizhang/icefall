
export PYTHONPATH=$PYTHONPATH:/workspace/icefall
pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html
pip install -r whisper/requirements.txt

torchrun --nproc_per_node 8 ./parawhisper/train.py \
  --max-duration 200 \
  --exp-dir parawhisper/exp_large_v2_nar \
  --model-name large-v2 \
  --manifest-dir data/fbank \
  --deepspeed \
  --deepspeed_config ./whisper/ds_config_zero1.json
