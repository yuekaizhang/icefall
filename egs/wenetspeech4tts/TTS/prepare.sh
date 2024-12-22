#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/home/yuekaiz/icefall_matcha
set -eou pipefail



# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

stage=7
stop_stage=7

dl_dir=$PWD/download

dataset_parts="Premium" # Basic for all 10k hours data, Premium for about 10% of the data

text_extractor="pypinyin_initials_finals" # default is espeak for English
audio_extractor="Encodec"  # or Fbank
audio_feats_dir=data/tokenized

. shared/parse_options.sh || exit 1


# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data
log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "dl_dir: $dl_dir"
  log "Stage 0: Download data"
  huggingface-cli login
  huggingface-cli download --repo-type dataset --local-dir $dl_dir Wenetspeech4TTS/WenetSpeech4TTS

  # Extract the downloaded data:
  for folder in Standard Premium Basic; do
    for file in "$dl_dir/$folder"/*.tar.gz; do
      tar -xzvf "$file" -C "$dl_dir/$folder"
    done
  done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare wenetspeech4tts manifest"
  # We assume that you have downloaded the wenetspeech4tts corpus
  # to $dl_dir/wenetspeech4tts
  mkdir -p data/manifests
  if [ ! -e data/manifests/.wenetspeech4tts.done ]; then
    lhotse prepare wenetspeech4tts $dl_dir data/manifests --dataset-parts "${dataset_parts}"
    touch data/manifests/.wenetspeech4tts.done
  fi
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Tokenize/Fbank wenetspeech4tts"
  mkdir -p ${audio_feats_dir}
  if [ ! -e ${audio_feats_dir}/.wenetspeech4tts.tokenize.done ]; then
    python3 ./local/compute_neural_codec_and_prepare_text_tokens.py --dataset-parts "${dataset_parts}" \
        --text-extractor ${text_extractor} \
        --audio-extractor ${audio_extractor} \
        --batch-duration 2500 --prefix "wenetspeech4tts" \
        --src-dir "data/manifests" \
	      --split 100 \
        --output-dir "${audio_feats_dir}/wenetspeech4tts_${dataset_parts}_split_100"
    cp ${audio_feats_dir}/wenetspeech4tts_${dataset_parts}_split_100/unique_text_tokens.k2symbols ${audio_feats_dir}
  fi
  touch ${audio_feats_dir}/.wenetspeech4tts.tokenize.done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Combine features"
  if [ ! -f ${audio_feats_dir}/wenetspeech4tts_cuts_${dataset_parts}.jsonl.gz ]; then
    pieces=$(find ${audio_feats_dir}/wenetspeech4tts_${dataset_parts}_split_100 -name "*.jsonl.gz")
    lhotse combine $pieces ${audio_feats_dir}/wenetspeech4tts_cuts_${dataset_parts}.jsonl.gz
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare wenetspeech4tts train/dev/test"
  if [ ! -e ${audio_feats_dir}/.wenetspeech4tts.train.done ]; then

    lhotse subset --first 400 \
        ${audio_feats_dir}/wenetspeech4tts_cuts_${dataset_parts}.jsonl.gz \
        ${audio_feats_dir}/cuts_dev.jsonl.gz

    lhotse subset --last 400 \
        ${audio_feats_dir}/wenetspeech4tts_cuts_${dataset_parts}.jsonl.gz \
        ${audio_feats_dir}/cuts_test.jsonl.gz

    lhotse copy \
      ${audio_feats_dir}/wenetspeech4tts_cuts_${dataset_parts}.jsonl.gz \
      ${audio_feats_dir}/cuts_train.jsonl.gz

    touch ${audio_feats_dir}/.wenetspeech4tts.train.done
  fi
  python3 ./local/display_manifest_statistics.py --manifest-dir ${audio_feats_dir}
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: build monotonic_align lib (used by matcha recipes)"
  for recipe in matcha; do
    if [ ! -d $recipe/monotonic_align/build ]; then
      cd $recipe/monotonic_align
      python3 setup.py build_ext --inplace
      cd ../../
    else
      log "monotonic_align lib for $recipe already built"
    fi
  done
fi

subset="Basic"
prefix="wenetspeech4tts"
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Generate fbank (used by ./matcha)"
  mkdir -p data/fbank
  if [ ! -e data/fbank/.${prefix}.done ]; then
    ./local/compute_mel_feat.py --dataset-parts $subset --split 100
    touch data/fbank/.${prefix}.done
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Split the ${prefix} cuts into train, valid and test sets (used by ./matcha)"
  if [ ! -f data/fbank/${prefix}_cuts_${subset}.jsonl.gz ]; then
    echo "Combining ${prefix} cuts"
    pieces=$(find data/fbank/ -name "${prefix}_cuts_${subset}.*.jsonl.gz")
    # lhotse combine $pieces data/fbank/${prefix}_cuts_${subset}.jsonl.gz
  fi
  if [ ! -e data/fbank/.${prefix}_split.done ]; then
    echo "Splitting ${prefix} cuts into train, valid and test sets"

    # lhotse subset --last 800 \
    #   data/fbank/${prefix}_cuts_${subset}.jsonl.gz \
    #   data/fbank/${prefix}_cuts_validtest.jsonl.gz
    # lhotse subset --first 400 \
    #   data/fbank/${prefix}_cuts_validtest.jsonl.gz \
    #   data/fbank/${prefix}_cuts_valid.jsonl.gz
    # lhotse subset --last 400 \
    #   data/fbank/${prefix}_cuts_validtest.jsonl.gz \
    #   data/fbank/${prefix}_cuts_test.jsonl.gz

    # rm data/fbank/${prefix}_cuts_validtest.jsonl.gz

    n=$(( $(gunzip -c data/fbank/${prefix}_cuts_${subset}.jsonl.gz | wc -l) - 800 ))
    lhotse subset --first $n  \
      data/fbank/${prefix}_cuts_${subset}.jsonl.gz \
      data/fbank/${prefix}_cuts_train.jsonl.gz
      touch data/fbank/.${prefix}_split.done
  fi
fi
