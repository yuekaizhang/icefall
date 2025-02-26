#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
# Copyright    2021  Xiaomi Corp.             (Fangjun Kuang)
# Copyright    2023  Xiaomi Corp.             (Zengrui Jin)
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

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from lhotse import (
    CutSet,
    LilcomChunkyWriter,
    WhisperFbank,
    WhisperFbankConfig,
)

from icefall.utils import str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=80,
        help="""The number of mel bins for Fbank""",
    )
    parser.add_argument(
        "--whisper-fbank",
        type=str2bool,
        default=True,
        help="Use WhisperFbank instead of Fbank. Default: False.",
    )
    parser.add_argument(
        "--resample-to-16kHz",
        type=str2bool,
        default=True,
        help="Resample audio to 16kHz. Default: False.",
    )
    parser.add_argument(
        "--speed-perturb",
        type=str2bool,
        default=False,
        help="Enable 0.9 and 1.1 speed perturbation for data augmentation. Default: False.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/fbank",
        help="Output directory for the computed features",
    )
    parser.add_argument(
        "--huggingface-dataset-path-or-name",
        type=str,
        default="/workspace/Belle_1.4M-SLAM-Omni",
        help="The path or name of the Huggingface dataset",
    )
    parser.add_argument(
        "--audio-key",
        type=str,
        default="question_audio",
        help="The key in the Huggingface dataset containing the audio data",
    )
    parser.add_argument(
        "--text-key",
        type=str,
        default="answer",
        help="The key in the Huggingface dataset containing the text data",
    )
    
    return parser


def compute_fbank(args):
    in_out_dir = Path(args.out_dir)
    in_out_dir.mkdir(parents=True, exist_ok=True)
    # number of workers in dataloader
    num_workers = 4

    # number of seconds in a batch
    batch_duration = 10

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    if args.whisper_fbank:
        extractor = WhisperFbank(
            WhisperFbankConfig(num_filters=args.num_mel_bins, device=device)
        )
    else:
        extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))

    logging.info(f"device: {device}")

    start = 0
    stop = 1601 
    num_digits = 5
    for i in range(start, stop):
        idx = f"{i}".zfill(num_digits)
        # dataset = load_dataset(args.huggingface_dataset_path_or_name, streaming=True, split=partition)
        parquet_files = [
            f"data/train-{idx}-of-01601.parquet",
        ]
        parquet_files = [f"{args.huggingface_dataset_path_or_name}/{f}" for f in parquet_files]
        file_name = parquet_files[0]
        logging.info(f"Loading dataset from {file_name}")
        dataset = load_dataset('parquet', data_files=parquet_files, streaming=True, split='train')

        cut_set = CutSet.from_huggingface_dataset(dataset, audio_key=args.audio_key, text_key=args.text_key)

        logging.info("Splitting cuts into smaller chunks")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        if args.resample_to_16kHz:
            cut_set = cut_set.resample(16000)
        if args.speed_perturb:
            cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)

        logging.info("Computing features")
        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{in_out_dir}/feats_{idx}",
            num_workers=num_workers,
            batch_duration=batch_duration,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )
        cuts_path = f"{in_out_dir}/cuts_belle.{idx}.jsonl.gz"
        logging.info(f"Saving to {cuts_path}")
        # cut_set.to_file(cuts_path)
        remove_recording_item(cut_set, cuts_path)

def remove_recording_item(
    cuts,
    output_cuts,
):
    """
    don't store recording item
    """
    with CutSet.open_writer(output_cuts) as writer:
        for cut in cuts:
            cut.recording.sources = None
            writer.write(cut)

def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    compute_fbank(args)


if __name__ == "__main__":
    main()
