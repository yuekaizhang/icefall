#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
#              2024  Yuekai Zhang
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

#fine-tuning with deepspeed zero stage 1
torchrun --nproc-per-node 8 ./whisper/train.py \
  --max-duration 200 \
  --exp-dir whisper/exp_large_v2 \
  --model-name large-v2 \
  --deepspeed \
  --deepspeed_config ./whisper/ds_config_zero1.json
"""

import argparse
import copy
import logging
import os
import random
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import k2
# import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import whisper
from asr_datamodule import AsrDataModule
from model import SPEECH_LLM, EncoderProjector, IGNORE_TOKEN_ID
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from label_smoothing import LabelSmoothingLoss
from lhotse import CutSet, load_manifest
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from multi_dataset import MultiDataset
# from optim import Eden, ScaledAdam
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.functional import pad as pad_tensor
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from whisper_encoder_forward_monkey_patch import replace_whisper_encoder_forward

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import update_averaged_model
from icefall.dist import cleanup_dist, get_rank, get_world_size, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    filter_uneven_sized_batch,
    setup_logger,
    str2bool,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
DEFAULT_SPEECH_TOKEN = "<speech>"

def set_batch_count(model: nn.Module, batch_count: float) -> None:
    for module in model.modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count

def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--llm-path-or-name",
        type=str,
        default="/workspace/asr/Qwen1.5-0.5B-Chat",
        help="Path or name of the large language model.",
    )

    parser.add_argument(
        "--speech-encoder-path-or-name",
        type=str,
        default="whisper-large-v2",
        help="Path or name of the speech encoder.",
    )

    parser.add_argument(
        "--encoder-projector-ds-rate",
        type=int,
        default=4,
        help="Downsample rate for the encoder projector.",
    )

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train.",
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
        default="whisper_qwen/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="""The path to the pretrained model if it is not None. Training will
        start from this model. e.g. ./wenetspeech/ASR/whisper/exp_large_v2/epoch-4-avg-3.pt
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=1e-5, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=6,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=True,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--use-flash-attn",
        type=str2bool,
        default=True,
        help="Whether to use flash attention.",
    )

    parser = deepspeed.add_config_arguments(parser)
    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - frame_shift_ms: The frame shift in milliseconds.
        - allowed_excess_duration_ratio: The allowed excess duration ratio.
        - best_train_loss: The best training loss so far.
        - best_valid_loss: The best validation loss so far.
        - best_train_epoch: The epoch where the best training loss is achieved.
        - best_valid_epoch: The epoch where the best validation loss is achieved.
        - batch_idx_train: The batch index of the current batch.
        - log_interval: Log training stats every `log_interval` batches.
        - reset_interval: Reset the stats every `reset_interval` batches.
        - valid_interval: Run validation every `valid_interval` batches.
        - env_info: The environment information.
    """
    params = AttributeDict(
        {
            "allowed_excess_duration_ratio": 0.1,
            "subsampling_factor": 2,
            "frame_shift_ms": 10,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 10000,
            "env_info": get_env_info(),
        }
    )

    return params


# def load_checkpoint_if_available(
#     params: AttributeDict,
#     model: nn.Module,
#     model_avg: nn.Module = None,
#     optimizer: Optional[torch.optim.Optimizer] = None,
#     scheduler: Optional[LRSchedulerType] = None,
# ) -> Optional[Dict[str, Any]]:
#     """Load checkpoint from file.

#     If params.start_batch is positive, it will load the checkpoint from
#     `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
#     params.start_epoch is larger than 1, it will load the checkpoint from
#     `params.start_epoch - 1`.

#     Apart from loading state dict for `model` and `optimizer` it also updates
#     `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
#     and `best_valid_loss` in `params`.

#     Args:
#       params:
#         The return value of :func:`get_params`.
#       model:
#         The training model.
#       model_avg:
#         The stored model averaged from the start of training.
#       optimizer:
#         The optimizer that we are using.
#       scheduler:
#         The scheduler that we are using.
#     Returns:
#       Return a dict containing previously saved training info.
#     """
#     if params.start_batch > 0:
#         filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
#     elif params.start_epoch > 1:
#         filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
#     else:
#         return None

#     assert filename.is_file(), f"{filename} does not exist!"

#     saved_params = load_checkpoint(
#         filename,
#         model=model,
#         model_avg=model_avg,
#         optimizer=optimizer,
#         scheduler=scheduler,
#     )

#     keys = [
#         "best_train_epoch",
#         "best_valid_epoch",
#         "batch_idx_train",
#         "best_train_loss",
#         "best_valid_loss",
#     ]
#     for k in keys:
#         params[k] = saved_params[k]

#     if params.start_batch > 0:
#         if "cur_epoch" in saved_params:
#             params["start_epoch"] = saved_params["cur_epoch"]

#     return saved_params


def compute_loss(
    params: AttributeDict,
    tokenizer: AutoTokenizer,
    model: nn.Module,
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute the loss for the given batch.
    Args:
        params:
            It is returned by :func:`get_params`.
        tokenizer:
            The tokenizer used to encode the text.
        model:
            The model for training.
        batch:
            A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
            for the content in it.
        is_training:
            Whether it is training.
    Returns:
        Return a tuple of two elements. The first element is the loss tensor.
    """
    # For the uneven-sized batch, the total duration after padding would possibly
    # cause OOM. Hence, for each batch, which is sorted descendingly by length,
    # we simply drop the last few shortest samples, so that the retained total frames
    # (after padding) would not exceed `allowed_max_frames`:
    # `allowed_max_frames = int(max_frames * (1.0 + allowed_excess_duration_ratio))`,
    # where `max_frames = max_duration * 1000 // frame_shift_ms`.
    # We set allowed_excess_duration_ratio=0.1.

    def preprocess(
        messages,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
    ) -> Dict:
        """Preprocesses the data for supervised fine-tuning."""
        texts = []
        for i, msg in enumerate(messages):
            texts.append(
                tokenizer.apply_chat_template(
                    msg,
                    tokenize=True,
                    add_generation_prompt=False,
                    padding="max_length",
                    max_length=max_len,
                    truncation=True,
                )
            )

        input_ids = torch.tensor(texts, dtype=torch.int)
        # response = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
        target_ids = input_ids.clone()
        target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        return input_ids, attention_mask, target_ids

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

    max_frames = params.max_duration * 1000 // params.frame_shift_ms
    allowed_max_frames = int(max_frames * (1.0 + params.allowed_excess_duration_ratio))
    batch = filter_uneven_sized_batch(batch, allowed_max_frames)

    device = next(model.parameters()).device
    feature = batch["inputs"]

    assert feature.ndim == 3
    feature = feature.to(device)
    feature = feature.transpose(1, 2)  # (N, C, T)

    batch_idx_train = params.batch_idx_train
    supervisions = batch["supervisions"]
    texts = batch["supervisions"]["text"]
    # remove spaces in texts
    texts = [normalize_text_alimeeting(text) for text in texts]

    messages = []
    for i, text in enumerate(texts):
        message = [
        {"role": "system", "content": "你是一个能处理音频的助手。"},
        {"role": "user", "content": f"请转写音频为文字 {DEFAULT_SPEECH_TOKEN}"},
        {"role": "assistant", "content": text},
        ]
        messages.append(message)

    input_ids, attention_mask, target_ids = preprocess(
        messages, tokenizer, max_len=128
    )

    target_ids = target_ids.type(torch.LongTensor)
    input_ids = input_ids.type(torch.LongTensor)

    with torch.set_grad_enabled(is_training):
        model_outpus = model(
            fbank=feature,
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=target_ids.to(device),
        )
        loss = model_outpus.loss
    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feature_lens = supervisions["num_frames"]
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    tokenizer: whisper.tokenizer.Tokenizer,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        with torch.cuda.amp.autocast(enabled=params.use_fp16):
            loss, loss_info = compute_loss(
                params=params,
                tokenizer=tokenizer,
                model=model,
                batch=batch,
                is_training=False,
            )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    tokenizer: AutoTokenizer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.encoder_projector.train()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])
        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                tokenizer=tokenizer,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

            model.save_checkpoint(
                save_dir=params.exp_dir,
                tag=f"epoch-{params.cur_epoch}-checkpoint-{batch_idx}",
                client_state={},
                exclude_frozen_parameters=True
            )

            if rank == 0:
                convert_zero_checkpoint_to_fp32_state_dict(
                    params.exp_dir,
                    f"{params.exp_dir}/epoch-{params.cur_epoch}-checkpoint-{batch_idx}.pt",
                    tag=f"epoch-{params.cur_epoch}-checkpoint-{batch_idx}",
                    exclude_frozen_parameters=True,
                )
                # save sampler state dict into checkpoint
                sampler_state_dict = train_dl.sampler.state_dict()
                torch.save(
                    sampler_state_dict,
                    f"{params.exp_dir}/epoch-{params.cur_epoch}-checkpoint-{batch_idx}-sampler.pt",
                )
                os.system(
                    f"rm -rf {params.exp_dir}/epoch-{params.cur_epoch}-checkpoint-{batch_idx}"
                )
        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    tokenizer=tokenizer,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.

            # deepspeed's backward() is different from torch's backward()
            # in that it does not accept a loss tensor as input.
            # It computes the loss internally.
            model.backward(loss)
            model.step()

        except:  # noqa
            display_and_save_batch(batch, params=params)
            raise

        if batch_idx % params.log_interval == 0:
            try:
                cur_lr = scheduler.get_last_lr()[0]
            except:  # noqa
                cur_lr = 0.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


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

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info(params)

    logging.info("About to create model")
    
    # if 'whisper' in params.speech_encoder_path_or_name:
    replace_whisper_encoder_forward()
    # TODO: directly loading from whisper-ft checkpoint
    # whisper-large-v2-multi-hans-zh-epoch-3-avg-10.pt
    whisper_model = whisper.load_model(params.speech_encoder_path_or_name, "cpu")
    speech_encoder = whisper_model.encoder
    speech_encoder_dim = whisper_model.dims.n_audio_state
    
    if params.use_flash_attn:
        attn_implementation = "flash_attention_2"
        torch_dtype=torch.bfloat16

    else:
        attn_implementation = "eager"
        torch_dtype=torch.float16

    llm = AutoModelForCausalLM.from_pretrained(
        params.llm_path_or_name,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(params.llm_path_or_name)
    tokenizer.padding_side  = 'left'
    special_tokens_dict = {
        "additional_special_tokens": [DEFAULT_SPEECH_TOKEN]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    llm.config.pad_token_id = tokenizer.pad_token_id
    llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)

    encoder_projector = EncoderProjector(speech_encoder_dim, llm.config.hidden_size)

    model = SPEECH_LLM(
        speech_encoder,
        llm,
        encoder_projector,
    )

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    logging.info("Trainable parameters (excluding model.eval modules):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"{name}: {param.shape}")

    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")
    model.to(device)

    # assert params.start_epoch > 0, params.start_epoch
    # checkpoints = load_checkpoint_if_available(
    #     params=params, model=model, model_avg=model_avg
    # )

    assert params.deepspeed and world_size > 1
    logging.info("Using DeepSpeed")
    model, optimizer, _, scheduler = deepspeed.initialize(
        args=params, model=model, model_parameters=model.parameters()
    )

    data_module = AsrDataModule(args)
    multi_dataset = MultiDataset(args.manifest_dir)

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        if c.duration < 1.0 or c.duration > 20.0:
            logging.warning(
               f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            )
            return False
        return True

    # train_cuts = multi_dataset.train_cuts()
    train_cuts = multi_dataset.aishell_train_cuts()
    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    # if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
    #     # We only load the sampler's state dict when it loads a checkpoint
    #     # saved in the middle of an epoch
    #     sampler_state_dict = checkpoints["sampler"]
    # else:
    #     sampler_state_dict = None
    sampler_state_dict = None
    # TODO: load sampler state dict
    train_dl = data_module.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    # valid_cuts = multi_dataset.dev_cuts()
    valid_cuts = multi_dataset.aishell_dev_cuts()
    valid_dl = data_module.valid_dataloaders(valid_cuts)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    # if params.pretrained_model_path:
    #     checkpoint = torch.load(params.pretrained_model_path, map_location="cpu")
    #     if "model" not in checkpoint:
    #         model.load_state_dict(checkpoint, strict=True)
    #     else:
    #         load_checkpoint(params.pretrained_model_path, model)

    logging.info(f"start training from epoch {params.start_epoch}")
    for epoch in range(params.start_epoch, params.num_epochs + 1):

        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        
        model.save_checkpoint(
            save_dir=params.exp_dir,
            tag=f"epoch-{params.cur_epoch}",
            client_state={},
            exclude_frozen_parameters=True
        )
        if rank == 0:
            convert_zero_checkpoint_to_fp32_state_dict(
                params.exp_dir,
                f"{params.exp_dir}/epoch-{params.cur_epoch}.pt",
                tag=f"epoch-{params.cur_epoch}",
                exclude_frozen_parameters=True,
            )
            # save sampler state dict into checkpoint
            sampler_state_dict = train_dl.sampler.state_dict()
            torch.save(sampler_state_dict, f"{params.exp_dir}/epoch-{params.cur_epoch}-sampler.pt")
    
            os.system(f"rm -rf {params.exp_dir}/epoch-{params.cur_epoch}")


    logging.info("Done!")

def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")


def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = get_world_size()
    rank = get_rank()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(rank=rank, world_size=world_size, args=args)


if __name__ == "__main__":
    main()
