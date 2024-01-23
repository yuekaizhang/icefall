#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=131
stop_stage=131

# Split L subset to this number of pieces
# This is to avoid OOM during feature extraction.
num_splits=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/WenetSpeech
#      You can find audio, WenetSpeech.json inside it.
#      You can apply for the download credentials by following
#      https://github.com/wenet-e2e/WenetSpeech#download
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech

dl_dir=$PWD/download
lang_char_dir=data/lang_char

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  [ ! -e $dl_dir/WenetSpeech ] && mkdir -p $dl_dir/WenetSpeech

  # If you have pre-downloaded it to /path/to/WenetSpeech,
  # you can create a symlink
  #
  # ln -sfv /path/to/WenetSpeech $dl_dir/WenetSpeech
  #
  if [ ! -d $dl_dir/WenetSpeech/wenet_speech ] && [ ! -f $dl_dir/WenetSpeech/metadata/v1.list ]; then
    log "Stage 0: You should download WenetSpeech first"
    exit 1;
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #ln -sfv /path/to/musan $dl_dir/musan

  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare WenetSpeech manifest"
  # We assume that you have downloaded the WenetSpeech corpus
  # to $dl_dir/WenetSpeech
  mkdir -p data/manifests
  lhotse prepare wenet-speech $dl_dir/WenetSpeech data/manifests -j $nj
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  mkdir -p data/manifests
  lhotse prepare musan $dl_dir/musan data/manifests
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Preprocess WenetSpeech manifest"
  if [ ! -f data/fbank/.preprocess_complete ]; then
    python3 ./local/preprocess_wenetspeech.py --perturb-speed True
    touch data/fbank/.preprocess_complete
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute features for DEV and TEST subsets of WenetSpeech (may take 2 minutes)"
  python3 ./local/compute_fbank_wenetspeech_dev_test.py
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Split S subset into ${num_splits} pieces"
  split_dir=data/fbank/S_split_${num_splits}
  if [ ! -f $split_dir/.split_completed ]; then
    lhotse split $num_splits ./data/fbank/cuts_S_raw.jsonl.gz $split_dir
    touch $split_dir/.split_completed
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Split M subset into ${num_splits} piece"
  split_dir=data/fbank/M_split_${num_splits}
  if [ ! -f $split_dir/.split_completed ]; then
    lhotse split $num_splits ./data/fbank/cuts_M_raw.jsonl.gz $split_dir
    touch $split_dir/.split_completed
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Split L subset into ${num_splits} pieces"
  split_dir=data/fbank/L_split_${num_splits}
  if [ ! -f $split_dir/.split_completed ]; then
    lhotse split $num_splits ./data/fbank/cuts_L_raw.jsonl.gz $split_dir
    touch $split_dir/.split_completed
  fi
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Compute features for S"
  python3 ./local/compute_fbank_wenetspeech_splits.py \
    --training-subset S \
    --num-workers 20 \
    --batch-duration 600 \
    --start 0 \
    --num-splits $num_splits
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Compute features for M"
  python3 ./local/compute_fbank_wenetspeech_splits.py \
    --training-subset M \
    --num-workers 20 \
    --batch-duration 600 \
    --start 0 \
    --num-splits $num_splits
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  log "Stage 10: Compute features for L"
  python3 ./local/compute_fbank_wenetspeech_splits.py \
    --training-subset L \
    --num-workers 20 \
    --batch-duration 600 \
    --start 0 \
    --num-splits $num_splits
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
  log "Stage 11: Combine features for S"
  if [ ! -f data/fbank/cuts_S.jsonl.gz ]; then
    pieces=$(find data/fbank/S_split_1000 -name "cuts_S.*.jsonl.gz")
    lhotse combine $pieces data/fbank/cuts_S.jsonl.gz
  fi
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
  log "Stage 12: Combine features for M"
  if [ ! -f data/fbank/cuts_M.jsonl.gz ]; then
    pieces=$(find data/fbank/M_split_1000 -name "cuts_M.*.jsonl.gz")
    lhotse combine $pieces data/fbank/cuts_M.jsonl.gz
  fi
fi

if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
  log "Stage 13: Combine features for L"
  if [ ! -f data/fbank/cuts_L.jsonl.gz ]; then
    pieces=$(find data/fbank/L_split_1000 -name "cuts_L.*.jsonl.gz")
    lhotse combine $pieces data/fbank/cuts_L.jsonl.gz
  fi
fi

whisper_mel_bins=80
if [ $stage -le 129 ] && [ $stop_stage -ge 129 ]; then
  log "Stage 129: compute whisper fbank for dev and test sets"
  python3 ./local/compute_fbank_wenetspeech_dev_test.py --num-mel-bins ${whisper_mel_bins} --whisper-fbank true
fi
if [ $stage -le 130 ] && [ $stop_stage -ge 130 ]; then
  log "Stage 130: Comute features for whisper training set"

  split_dir=data/fbank/L_split_${num_splits}
  if [ ! -f $split_dir/.split_completed ]; then
    lhotse split $num_splits ./data/fbank/cuts_L_raw.jsonl.gz $split_dir
    touch $split_dir/.split_completed
  fi

  python3 ./local/compute_fbank_wenetspeech_splits.py \
    --training-subset L \
    --num-workers 8 \
    --batch-duration 1600 \
    --start 0 \
    --num-mel-bins ${whisper_mel_bins} --whisper-fbank true \
    --num-splits $num_splits

  if [ ! -f data/fbank/cuts_L.jsonl.gz ]; then
    pieces=$(find data/fbank/L_split_1000 -name "cuts_L.*.jsonl.gz")
    lhotse combine $pieces data/fbank/cuts_L.jsonl.gz
  fi
fi

if [ $stage -le 131 ] && [ $stop_stage -ge 131 ]; then
  log "Stage 131: test"

  python3 ./local/compute_fbank_wenetspeech_splits.py \
    --training-subset L \
    --num-workers 8 \
    --batch-duration 1000 \
    --start 48 \
    --stop 58 \
    --num-mel-bins ${whisper_mel_bins} --whisper-fbank true \
    --num-splits $num_splits & 

  python3 ./local/compute_fbank_wenetspeech_splits.py \
    --training-subset L \
    --num-workers 8 \
    --batch-duration 1000 \
    --start 58 \
    --stop 68 \
    --num-mel-bins ${whisper_mel_bins} --whisper-fbank true \
    --num-splits $num_splits &

  python3 ./local/compute_fbank_wenetspeech_splits.py \
    --training-subset L \
    --num-workers 8 \
    --batch-duration 1000 \
    --start 68 \
    --stop 78 \
    --num-mel-bins ${whisper_mel_bins} --whisper-fbank true \
    --num-splits $num_splits &

  python3 ./local/compute_fbank_wenetspeech_splits.py \
    --training-subset L \
    --num-workers 8 \
    --batch-duration 1000 \
    --start 78 \
    --stop 88 \
    --num-mel-bins ${whisper_mel_bins} --whisper-fbank true \
    --num-splits $num_splits &

  python3 ./local/compute_fbank_wenetspeech_splits.py \
    --training-subset L \
    --num-workers 8 \
    --batch-duration 1000 \
    --start 88 \
    --num-mel-bins ${whisper_mel_bins} --whisper-fbank true \
    --num-splits $num_splits &
  
  wait
fi

if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
  log "Stage 14: Compute fbank for musan"
  mkdir -p data/fbank
  ./local/compute_fbank_musan.py
fi

if [ $stage -le 15 ] && [ $stop_stage -ge 15 ]; then
  log "Stage 15: Prepare char based lang"
  mkdir -p $lang_char_dir

  if ! which jq; then
      echo "This script is intended to be used with jq but you have not installed jq
      Note: in Linux, you can install jq with the following command:
      1. wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
      2. chmod +x ./jq
      3. cp jq /usr/bin" && exit 1
  fi
  if [ ! -f $lang_char_dir/text ] || [ ! -s $lang_char_dir/text ]; then
    log "Prepare text."
    gunzip -c data/manifests/wenetspeech_supervisions_L.jsonl.gz \
      | jq '.text' | sed 's/"//g' \
      | ./local/text2token.py -t "char" > $lang_char_dir/text
  fi

  # The implementation of chinese word segmentation for text,
  # and it will take about 15 minutes.
  if [ ! -f $lang_char_dir/text_words_segmentation ]; then
    python3 ./local/text2segments.py \
      --num-process $nj \
      --input-file $lang_char_dir/text \
      --output-file $lang_char_dir/text_words_segmentation
  fi

  cat $lang_char_dir/text_words_segmentation | sed 's/ /\n/g' \
    | sort -u | sed '/^$/d' | uniq > $lang_char_dir/words_no_ids.txt

  if [ ! -f $lang_char_dir/words.txt ]; then
    python3 ./local/prepare_words.py \
      --input-file $lang_char_dir/words_no_ids.txt \
      --output-file $lang_char_dir/words.txt
  fi
fi

if [ $stage -le 16 ] && [ $stop_stage -ge 16 ]; then
  log "Stage 16: Prepare char based L_disambig.pt"
  if [ ! -f data/lang_char/L_disambig.pt ]; then
    python3 ./local/prepare_char.py \
      --lang-dir data/lang_char
  fi
fi

# If you don't want to use LG for decoding, the following steps are not necessary.
if [ $stage -le 17 ] && [ $stop_stage -ge 17 ]; then
  log "Stage 17: Prepare G"
  # It will take about 20 minutes.
  # We assume you have installed kaldilm, if not, please install
  # it using: pip install kaldilm
  if [ ! -f $lang_char_dir/3-gram.unpruned.arpa ]; then
    python3 ./shared/make_kn_lm.py \
      -ngram-order 3 \
      -text $lang_char_dir/text_words_segmentation \
      -lm $lang_char_dir/3-gram.unpruned.arpa
  fi

  mkdir -p data/lm
  if [ ! -f data/lm/G_3_gram.fst.txt ]; then
    # It is used in building LG
    python3 -m kaldilm \
      --read-symbol-table="$lang_char_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $lang_char_dir/3-gram.unpruned.arpa > data/lm/G_3_gram.fst.txt
  fi
fi

if [ $stage -le 18 ] && [ $stop_stage -ge 18 ]; then
  log "Stage 18: Compile LG"
  python ./local/compile_lg.py --lang-dir $lang_char_dir
fi

# prepare RNNLM data
if [ $stage -le 19 ] && [ $stop_stage -ge 19 ]; then
  log "Stage 19: Prepare LM training data"

  log "Processing char based data"
  text_out_dir=data/lm_char

  mkdir -p $text_out_dir

  log "Genearating training text data"
  
  if [ ! -f $text_out_dir/lm_data.pt ]; then
    ./local/prepare_char_lm_training_data.py \
      --lang-char data/lang_char \
      --lm-data $lang_char_dir/text_words_segmentation \
      --lm-archive $text_out_dir/lm_data.pt
  fi

  log "Generating DEV text data"
  # prepare validation text data 
  if [ ! -f $text_out_dir/valid_text_words_segmentation ]; then
    valid_text=${text_out_dir}/

    gunzip -c data/manifests/wenetspeech_supervisions_DEV.jsonl.gz \
      | jq '.text' | sed 's/"//g' \
      | ./local/text2token.py -t "char" > $text_out_dir/valid_text
    
    python3 ./local/text2segments.py \
      --num-process $nj \
      --input-file $text_out_dir/valid_text \
      --output-file $text_out_dir/valid_text_words_segmentation
  fi

  ./local/prepare_char_lm_training_data.py \
    --lang-char data/lang_char \
    --lm-data $text_out_dir/valid_text_words_segmentation \
    --lm-archive $text_out_dir/lm_data_valid.pt

  # prepare TEST text data 
  if [ ! -f $text_out_dir/TEST_text_words_segmentation ]; then
    log "Prepare text for test set."
    for test_set in TEST_MEETING TEST_NET; do
        gunzip -c data/manifests/wenetspeech_supervisions_${test_set}.jsonl.gz \
          | jq '.text' | sed 's/"//g' \
          | ./local/text2token.py -t "char" > $text_out_dir/${test_set}_text

        python3 ./local/text2segments.py \
          --num-process $nj \
          --input-file $text_out_dir/${test_set}_text \
          --output-file $text_out_dir/${test_set}_text_words_segmentation
    done
    
    cat $text_out_dir/TEST_*_text_words_segmentation > $text_out_dir/test_text_words_segmentation
  fi

  ./local/prepare_char_lm_training_data.py \
    --lang-char data/lang_char \
    --lm-data $text_out_dir/test_text_words_segmentation \
    --lm-archive $text_out_dir/lm_data_test.pt

fi

# sort RNNLM data
if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
  text_out_dir=data/lm_char

  log "Sort lm data"

  ./local/sort_lm_training_data.py \
    --in-lm-data $text_out_dir/lm_data.pt \
    --out-lm-data $text_out_dir/sorted_lm_data.pt \
    --out-statistics $text_out_dir/statistics.txt

  ./local/sort_lm_training_data.py \
    --in-lm-data $text_out_dir/lm_data_valid.pt \
    --out-lm-data $text_out_dir/sorted_lm_data-valid.pt \
    --out-statistics $text_out_dir/statistics-valid.txt

  ./local/sort_lm_training_data.py \
    --in-lm-data $text_out_dir/lm_data_test.pt \
    --out-lm-data $text_out_dir/sorted_lm_data-test.pt \
    --out-statistics $text_out_dir/statistics-test.txt
fi

export CUDA_VISIBLE_DEVICES="0,1"

if [ $stage -le 21 ] && [ $stop_stage -ge 21 ]; then
  log "Stage 21: Train RNN LM model"
  python ../../../icefall/rnn_lm/train.py \
    --start-epoch 0 \
    --world-size 2 \
    --num-epochs 20 \
    --use-fp16 0 \
    --embedding-dim 2048 \
    --hidden-dim 2048 \
    --num-layers 2 \
    --batch-size 400 \
    --exp-dir rnnlm_char/exp \
    --lm-data data/lm_char/sorted_lm_data.pt \
    --lm-data-valid data/lm_char/sorted_lm_data-valid.pt \
    --vocab-size 5537 \
    --master-port 12340
fi