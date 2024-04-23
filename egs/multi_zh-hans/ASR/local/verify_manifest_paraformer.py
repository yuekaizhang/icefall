#!/usr/bin/env python3
# Copyright    2024  author: Yuekai Zhang
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import argparse
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import soundfile as sf

import re
import sherpa_onnx
from lhotse import CutSet, load_manifest_lazy
from kaldialign import edit_distance

"""
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
tar xvf sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
"""
def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/fbank/",
        help="Directory to store the manifest files",
    )

    parser.add_argument(
        "--paraformer-dir",
        type=str,
        default="/workspace/sherpa-onnx-paraformer-zh-2023-03-28/",
        help="Directory to store the paraformer model",
    )

    return parser   

def normalize_text_alimeeting(text: str, normalize: str = "m2met") -> str:
    """
    Text normalization similar to M2MeT challenge baseline.
    See: https://github.com/yufan-aslp/AliMeeting/blob/main/asr/local/text_normalize.pl
    """
    if normalize == "none":
        return text
    elif normalize == "m2met":
        import re
        text = text.replace(" ", "")
        text = text.replace("<sil>", "")
        text = text.replace("<%>", "")
        text = text.replace("<->", "")
        text = text.replace("<$>", "")
        text = text.replace("<#>", "")
        text = text.replace("<_>", "")
        text = text.replace("<space>", "")
        text = text.replace("`", "")
        text = text.replace("&", "")
        text = text.replace(",", "")
        if re.search("[a-zA-Z]", text):
            text = text.upper()
        text = text.replace("Ａ", "A")
        text = text.replace("ａ", "A")
        text = text.replace("ｂ", "B")
        text = text.replace("ｃ", "C")
        text = text.replace("ｋ", "K")
        text = text.replace("ｔ", "T")
        text = text.replace("，", "")
        text = text.replace("丶", "")
        text = text.replace("。", "")
        text = text.replace("、", "")
        text = text.replace("？", "")
        return text

def read_wave_multichannel(wave_filename: str, start_time: float = 0.0, duration: float = None) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It can be multi-channel and each sample can be of various bit depths.
      start_time:
        The start time in seconds from where to begin reading the wave file.
      duration:
        The duration in seconds to read from the wave file. If None, read the entire file from the start_time.
    Returns:
      Return a tuple containing:
       - A 2-D array of dtype np.float32 containing the samples, normalized to the range [-1, 1].
         The shape of the array will be (num_samples, num_channels).
       - Sample rate of the wave file.
    """
    # Open the file
    with sf.SoundFile(wave_filename) as f:
        # Get the sample rate
        sample_rate = f.samplerate
        
        # Calculate the number of frames to skip and the number of frames to read
        start_frame = int(start_time * sample_rate)
        if duration is not None:
            end_frame = start_frame + int(duration * sample_rate)
        else:
            end_frame = -1  # Read till the end
        
        # Seek to the start frame
        f.seek(start_frame)
        
        # Read the frames
        samples = f.read(frames=end_frame - start_frame, dtype='float32', always_2d=True)
        
        # If the file is mono, the output should still be 2D
        if samples.ndim == 1:
            samples = np.expand_dims(samples, axis=1)

        # If the file is stereo, we only want the first channel
        if samples.shape[1] > 1:
            samples = samples[:, 0:1]

        # choose the average of all channels
        # samples = np.mean(samples, axis=1, keepdims=True)
        
        # Normalize the samples to the range [-1, 1] if they aren't already
        if samples.dtype != np.float32 or samples.max() > 1 or samples.min() < -1:
            samples = samples / np.iinfo(samples.dtype).max

        print(samples.shape, sample_rate, samples.dtype, type(samples))
        
        return samples, sample_rate

def verify_manifest_paraformer(manifest_path, fixed_manifest_path, recognizer, save_fixed_transcript_path):
    cuts_manifest = load_manifest_lazy(manifest_path)
    keeped_id_set = {}
    with open (save_fixed_transcript_path, 'w', encoding='utf-8') as f:
        for i, cut in enumerate(cuts_manifest):
            if i % 10000 == 0:
                logging.info(f'Processing cut {i}')
            cut_id = cut.id
            if cut_id.endswith('_sp0.9'):
                continue
            elif cut_id.endswith('_sp1.1'):
                continue
            origin_text = cut.supervisions[0].text
            origin_text = normalize_text_alimeeting(origin_text)
            audio_source_path = cut.recording.sources[0].source
            
            samples, sample_rate = read_wave_multichannel(audio_source_path, cut.start, cut.duration)
            s = recognizer.create_stream()
            s.accept_waveform(sample_rate, samples)
            recognizer.decode_streams([s])
            results = s.result.text

            wer_results = edit_distance(list(origin_text), list(results))
            if wer_results['err_rate'] < 0.5:
                keeped_id_set.add(cut_id)
            else:
                f.write(f'{cut_id}\t{origin_text}\t{results}\n')
                logging.info(f'Cut {cut_id} origin text: {origin_text}, result text: {results}, wer: {wer_results["err_rate"]}')
        
    with CutSet.open_writer(fixed_manifest_path) as manifest_writer:
        for i, cut in enumerate(cuts_manifest):
            if i % 10000 == 0:
                logging.info(f'Saving cut {i}')
            cut_id = cut.id
            if cut_id.endswith('_sp0.9'):
                cut_id = cut_id[:-6]
            elif cut_id.endswith('_sp1.1'):
                cut_id = cut_id[:-6]
            if cut_id in keeped_id_set:
                manifest_writer.write(cut)

def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    paraformer_path = args.paraformer_dir + 'model.onnx'
    tokens_path = args.paraformer_dir + 'tokens.txt'

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=paraformer_path,
        tokens=tokens_path,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method='greedy_search',
        debug=False,
    )
    logging.info('Model loaded')

    manifest_paths = [args.manifest_dir + 'alimeeting-far_cuts_train.jsonl.gz']
    manifest_paths.append(args.manifest_dir + 'aishell4_cuts_train_L.jsonl.gz')
    manifest_paths.append(args.manifest_dir + 'aishell4_cuts_train_M.jsonl.gz')
    manifest_paths.append(args.manifest_dir + 'aishell4_cuts_train_S.jsonl.gz')

    for manifest_path in manifest_paths:
        fixed_manifest_path = manifest_path.replace('.jsonl.gz', '_fixed.jsonl.gz')
        logging.info(f'Loading manifest from {manifest_path}')
        removed_text_path = manifest_path.replace('.jsonl.gz', '_removed_text.txt')
        verify_manifest_paraformer(manifest_path, fixed_manifest_path, recognizer, removed_text_path)
        logging.info(f'Fixed manifest saved to {fixed_manifest_path}')

if __name__ == "__main__":
    main()