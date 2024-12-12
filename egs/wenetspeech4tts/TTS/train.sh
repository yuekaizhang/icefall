export PYTHONPATH=$PYTHONPATH:/workspace/icefall_valle

install_flag=false
if [ "$install_flag" = true ]; then
    echo "Installing packages..."

    pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html

    pip install -r /workspace/Amphion/requirements.txt
    pip install phonemizer pypinyin sentencepiece kaldialign matplotlib h5py

    apt-get update && apt-get -y install festival espeak-ng mbrola
else
    echo "Skipping installation."
fi

world_size=8
exp_dir=exp_test/valle_scratch_exp2

python3 moshi_scratch/train.py --max-duration 300 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 1000000000 --valid-interval 8000 \
      --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.001 --warmup-steps 200 --average-period 200 \
      --num-epochs 50 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 1 \
      --exp-dir ${exp_dir} --world-size ${world_size} --optimizer-name AdamW
