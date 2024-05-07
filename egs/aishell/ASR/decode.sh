
export PYTHONPATH=$PYTHONPATH:/workspace/icefall
#pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html
#pip install -r whisper/requirements.txt

# python3 ./whisper/decode.py \
#   --exp-dir whisper/exp_large_v2_sft \
#   --model-name large-v2 \
#   --epoch 999 --avg 1 \
#   --start-index 0 --end-index 26 \
#   --remove-whisper-encoder-input-length-restriction True \
#   --manifest-dir data/fbank \
#   --beam-size 1 --max-duration 50

# export PYTHONPATH="${PYTHONPATH}:/workspace/icefall/egs/speechio/ASR/zipformer"
# python3 ./whisper/decode_speculative.py \
#   --exp-dir whisper/exp_large_v2_nar \
#   --model-name large-v2 \
#   --epoch 9 --avg 6 \
#   --remove-whisper-encoder-input-length-restriction True \
#   --manifest-dir data/fbank \
#   --nn-model-filename /workspace/icefall-asr-aishell-zipformer-small-2023-10-24/exp/pretrained.pt \
#   --tokens /workspace/icefall-asr-aishell-zipformer-small-2023-10-24/data/lang_char/tokens.txt \
#   --beam-size 1 --max-duration 1

python3 ./parawhisper/decode.py \
  --exp-dir parawhisper/exp_large_v2_cif \
  --model-name large-v2 \
  --epoch 1 --avg 1 \
  --remove-whisper-encoder-input-length-restriction True \
  --manifest-dir data/fbank \
  --beam-size 1 --max-duration 200