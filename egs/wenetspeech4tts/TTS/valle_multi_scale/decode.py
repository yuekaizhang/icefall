#!/usr/bin/env python3
# Copyright    2021-2022  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo)
# Copyright    2023                           (authors: Feiteng Li)
# Copyright    2024                           (authors: Yuekai Zhang)
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
"""
Usage:
world_size=8
exp_dir=exp/valle

## Train AR model
python3 valle/train.py --max-duration 320 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 1000 --valid-interval 2000 \
      --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.03 --warmup-steps 200 --average-period 0 \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 1 \
      --exp-dir ${exp_dir} --world-size ${world_size}

## Train NAR model
# cd ${exp_dir}
# ln -s ${exp_dir}/best-valid-loss.pt epoch-99.pt  # --start-epoch 100=99+1
# cd -
python3 valle/train.py --max-duration 160 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
      --num-buckets 6 --dtype "float32" --save-every-n 1000 --valid-interval 2000 \
      --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.03 --warmup-steps 200 --average-period 0 \
      --num-epochs 40 --start-epoch 100 --start-batch 0 --accumulate-grad-steps 2 \
      --exp-dir ${exp_dir} --world-size ${world_size}
"""
import argparse
import copy
import logging
import os
import random
import warnings
from contextlib import nullcontext
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchaudio
from encodec.utils import convert_audio
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from model import SPEECH_LLM
from optim import Eden, ScaledAdam
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tts_datamodule import TtsDataModule

from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import AttributeDict, MetricsTracker, setup_logger, str2bool

LRSchedulerType = torch.optim.lr_scheduler._LRScheduler

from compute_neural_codec_and_prepare_text_tokens import AudioTokenizer, TextTokenizer
from tokenizer import TextTokenCollater, get_text_token_collater
from transformers import AutoTokenizer

def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=1024,
        help="Embedding dimension in the decoder model.",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/valle_dev",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default="aishell3.txt",
        help="""The input file for decoding.
        """,
    )

    parser.add_argument(
        "--text-tokens",
        type=str,
        default="data/tokenized/unique_text_tokens.k2symbols",
        help="Path to the unique text tokens file",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="The top-p value for sampling",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="The top-p value for sampling",
    )
    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

    - `env_info`: Information about the environment.
    """
    params = AttributeDict(
        {
            "env_info": get_env_info(),
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    if isinstance(model, DDP):
        raise ValueError("load_checkpoint before DDP")
    print(f"Loading checkpoint from {filename}")
    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return saved_params, filename


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    rng = random.Random(params.seed)

    setup_logger(f"{params.exp_dir}/log/log-decode")
    logging.info("Decoding started")

    device = torch.device("cuda")
    logging.info(f"Device: {device}")

    tokenizer = get_text_token_collater(params.text_tokens)
    audio_tokenizer = AudioTokenizer()
    logging.info(params)
    logging.info("About to create model")

    model = SPEECH_LLM()
    checkpoints, filename = load_checkpoint_if_available(
        params=params, model=model, model_avg=None
    )
    model.to(device)
    # get the basename of input_file
    label_base = Path(args.input_file).stem
    audio_save_dir = f"{params.exp_dir}/{label_base}-{filename.stem}_wavs-top-p-{params.top_p}-top-k-{params.top_k}"
    os.makedirs(audio_save_dir, exist_ok=True)

    with open(args.input_file, "r") as f:
        for line in f:
            # fields = line.strip().split("  ")
            # fields = line.strip().split(" ")
            # fields = [item for item in fields if item]
            # assert len(fields) == 4
            # prompt_text, prompt_audio, text, audio_path = fields
            fields = line.strip().split("|")
            fields = [item for item in fields if item]
            assert len(fields) == 4
            audio_path, prompt_text, prompt_audio, text = fields


            logging.info(f"synthesize text: {text}")

            input_text = prompt_text + " " + text
            text_tokens, _ = prepare_input_ids([input_text], tokenizer, device)
   
            audio_prompts = tokenize_audio(audio_tokenizer, prompt_audio)
            audio_prompts = audio_prompts[0][0].to(device)

            with torch.no_grad():
                generated_audio_tokens_shift_back = model.generate(
                    text_tokens.to(device),
                    audio_prompts,
                    [audio_path],
                    top_p=params.top_p,
                    top_k=params.top_k,
                )
            batch_size = 1
            for i in range(batch_size):
                real_audio_code_save = generated_audio_tokens_shift_back[i]
                audio_base_name = Path(audio_path).stem
                audio_save_path = f"{audio_save_dir}/{audio_base_name}.wav"
                save_audio(real_audio_code_save, audio_tokenizer, audio_save_path)

    logging.info("Done!")


def tokenize_audio(tokenizer: AudioTokenizer, audio_path: str):
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames

def save_audio(audio_codes, audio_tokenizer, path):
    audio_code = audio_codes.unsqueeze(0)
    audio_code = audio_code.to(torch.int64)
    samples_org = audio_tokenizer.decode([(audio_code, None)])
    torchaudio.save(path, samples_org[0].cpu(), 24000)

def prepare_input_ids(texts_input, tokenizer: TextTokenCollater, device: torch.device):
    """Parse batch data"""
    # TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
    # texts = []
    # for text in texts_input:
    #     print(text)
    #     message = [
    #         {"role": "user", "content": text},
    #     ]
    #     text = tokenizer.apply_chat_template(
    #         message, tokenize=False, chat_template=TEMPLATE, add_generation_prompt=False
    #     )
    #     texts.append(text)

    # text_tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    # input_ids = text_tokens["input_ids"].to(device)
    # attention_mask = text_tokens["attention_mask"].to(device)
    text_tokenizer = TextTokenizer(backend="pypinyin_initials_finals")
    phonemes_list = []
    for text in texts_input:
        phonemes = text_tokenizer([text.strip()])[0]
        phonemes_list.append(phonemes)
        print(text, phonemes)

    input_ids, text_tokens_lens = tokenizer(phonemes_list)

    input_ids, text_tokens_lens = input_ids.to(device), text_tokens_lens.to(device)
    # make attention mask from text_tokens_lens, shape is the same as input_ids, 1 for real token, 0 for padding
    attention_mask = (
        torch.arange(input_ids.size(1), device=device)[None, :]
        < text_tokens_lens[:, None]
    )

    attention_mask = attention_mask.to(torch.long)

    print(input_ids.shape, attention_mask.shape)

    return input_ids, attention_mask


def main():
    parser = get_parser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
