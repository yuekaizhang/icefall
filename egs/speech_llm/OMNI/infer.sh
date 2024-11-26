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
exp_dir=valle/exp/valle

epoch=40
avg=1
#python3 bin/generate_averaged_model.py \
#    --epoch ${epoch} \
#    --avg ${avg} \
#    --exp-dir ${exp_dir}



#python3 bin/infer.py --output-dir demos_epoch_${epoch}_avg_${avg} \
#    --checkpoint=${exp_dir}/epoch-${epoch}-avg-${avg}.pt \
#    --text-prompts "KNOT one point one five miles per hour." \
#    --audio-prompts ./prompts/8463_294825_000043_000000.wav \
#    --text "To get up and running quickly just follow the steps below."
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
top_p=0.2

ras=true
python3 valle/infer.py --output-dir demos_epoch_${epoch}_avg_${avg}_top_p_${top_p}_ras_${ras} \
        --top-k -1 --temperature 1.0 \
        --text-prompts "" \
        --audio-prompts "" \
        --text ./aishell3.txt \
        --checkpoint ${exp_dir}/epoch-${epoch}-avg-${avg}.pt \
	--text-extractor pypinyin_initials_finals --top-p ${top_p} --repetition-aware-sampling ${ras}
# ras=false
# python3 valle/infer.py --output-dir demos_epoch_${epoch}_avg_${avg}_top_p_${top_p}_ras_${ras} \
#         --top-k -1 --temperature 1.0 \
#         --text-prompts "" \
#         --audio-prompts "" \
#         --text ./aishell3.txt \
#         --checkpoint ${exp_dir}/epoch-${epoch}-avg-${avg}.pt \
# 	--text-extractor pypinyin_initials_finals --top-p ${top_p}
# world_size=1
# exp_dir=exp_test/valle

# ## Train AR model
# python3 valle/train.py --max-duration 320 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
#       --num-buckets 6 --dtype "bfloat16" --save-every-n 1000 --valid-interval 2000 \
#       --share-embedding true --norm-first true --add-prenet false \
#       --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
#       --base-lr 0.03 --warmup-steps 200 --average-period 0 \
#       --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 1 \
#       --exp-dir ${exp_dir} --world-size ${world_size}
