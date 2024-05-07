#!/usr/bin/env python3
# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo,
#                                            Fangjun Kuang,
#                                            Wei Kang)
#           2024 Yuekai Zhang
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
# Command for decoding using fine-tuned models:
git lfs install
git clone https://huggingface.co/yuekai/icefall_asr_aishell_whisper
ln -s icefall_asr_aishell_whisper/exp_large_v2/epoch-10-avg6.pt whisper/exp_large_v2/epoch-999.pt

python3 ./whisper/decode.py \
  --exp-dir whisper/exp_large_v2 \
  --model-name large-v2 \
  --epoch 999 --avg 1 \
  --manifest-dir data/fbank_whisper \
  --beam-size 10 --max-duration 50

# Command for decoding using pretrained models (before fine-tuning):

python3 ./whisper/decode.py \
  --exp-dir whisper/exp_large_v2 \
  --model-name large-v2 \
  --epoch -1 --avg 1 \
  --manifest-dir data/fbank_whisper \
  --remove-whisper-encoder-input-length-restriction False \
  --beam-size 10 --max-duration 50

"""

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import torch
import torch.nn as nn
import whisper
from asr_datamodule import AsrDataModule
from lhotse.cut import Cut
from multi_dataset import MultiDataset
from tn.chinese.normalizer import Normalizer
from whisper.normalizers import BasicTextNormalizer
from whisper_encoder_forward_monkey_patch import replace_whisper_encoder_forward
from whisper_decoder_forward_monkey_patch import replace_whisper_decoder_forward
from model import ParaWhisper
from zhconv import convert

from icefall.checkpoint import average_checkpoints_with_averaged_model, load_checkpoint
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)


def average_checkpoints(
    filenames: List[Path], device: torch.device = torch.device("cpu")
) -> dict:
    """Average a list of checkpoints.
    The function is mainly used for deepspeed converted checkpoint averaging, which only include model state_dict.

    Args:
      filenames:
        Filenames of the checkpoints to be averaged. We assume all
        checkpoints are saved by :func:`save_checkpoint`.
      device:
        Move checkpoints to this device before averaging.
    Returns:
      Return a dict (i.e., state_dict) which is the average of all
      model state dicts contained in the checkpoints.
    """
    n = len(filenames)

    if "model" in torch.load(filenames[0], map_location=device):
        avg = torch.load(filenames[0], map_location=device)["model"]
    else:
        avg = torch.load(filenames[0], map_location=device)

    # Identify shared parameters. Two parameters are said to be shared
    # if they have the same data_ptr
    uniqued: Dict[int, str] = dict()

    for k, v in avg.items():
        v_data_ptr = v.data_ptr()
        if v_data_ptr in uniqued:
            continue
        uniqued[v_data_ptr] = k

    uniqued_names = list(uniqued.values())

    for i in range(1, n):
        if "model" in torch.load(filenames[i], map_location=device):
            state_dict = torch.load(filenames[i], map_location=device)["model"]
        else:
            state_dict = torch.load(filenames[i], map_location=device)
        for k in uniqued_names:
            avg[k] += state_dict[k]

    for k in uniqued_names:
        if avg[k].is_floating_point():
            avg[k] /= n
        else:
            avg[k] //= n

    return avg


def remove_punctuation(text: str or List[str]):
    """Modified from https://github.com/yeyupiaoling/Whisper-Finetune/blob/master/utils/data_utils.py

    Args:
        text: It can be a string or a list of strings.
    Returns:
        Return a string or a list of strings without any punctuation.
    """
    punctuation = "!,.;:?、！，。；：？《》 "
    if isinstance(text, str):
        text = re.sub(r"[{}]+".format(punctuation), "", text).strip()
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = re.sub(r"[{}]+".format(punctuation), "", t).strip()
            result_text.append(t)
        return result_text
    else:
        raise Exception(f"Not support type {type(text)}")


def to_simple(text: str or List[str]):
    """Convert traditional Chinese to simplified Chinese.
    Args:
        text: It can be a string or a list of strings.
    Returns:
        Return a string or a list of strings converted to simplified Chinese.
    """
    if isinstance(text, str):
        text = convert(text, "zh-cn")
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = convert(t, "zh-cn")
            result_text.append(t)
        return result_text
    else:
        raise Exception(f"Not support type{type(text)}")


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="cif",
        help="""Decoding method.
        Supported values are:
          - cif
          - ar
          - cif-oracle
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="beam size for beam search decoding",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="whisper/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="large-v2",
        choices=["large-v2", "large-v3", "medium", "small", "tiny"],
        help="""The model name to use.
        """,
    )

    parser.add_argument(
        "--remove-whisper-encoder-input-length-restriction",
        type=str2bool,
        default=True,
        help="replace whisper encoder forward method to remove input length restriction",
    )

    parser.add_argument(
        "--custom-token-path",
        type=str,
        default="parawhisper/aishell_tokens_whisper.txt",
        help="The path to the custom dict.",
    )

    parser.add_argument(
        "--ctc-only",
        type=str2bool,
        default=False,
        help="Whether to use CTC only.",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "env_info": get_env_info(),
        }
    )
    return params

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

def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
) -> Dict[str, List[List[int]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: "beam-search"
        - value: A list of lists. Each sublist is a list of token IDs.
    Args:
        params:
            It is returned by :func:`get_params`.
        model:
            The neural model.
        batch:
            It is returned by :meth:`torch.utils.data.DataLoader.__iter__`.
    Returns:
        Return a dict, whose key may be "beam-search".
    """
    dtype = torch.float16
    device = torch.device("cuda")

    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device, dtype=dtype).transpose(1, 2)
    if not params.remove_whisper_encoder_input_length_restriction:
        T = 3000
        if feature.shape[2] < T:
            feature = torch.cat(
                [
                    feature,
                    torch.zeros(
                        feature.shape[0], feature.shape[1], T - feature.shape[2]
                    ).to(device, dtype=dtype),
                ],
                2,
            )

    supervisions = batch["supervisions"]
    feature_len = supervisions["num_frames"]
    feature_len = feature_len.to(device, dtype=dtype)


    texts = batch["supervisions"]["text"]
    # remove spaces in texts
    texts = [normalize_text_alimeeting(text) for text in texts]
    text_tokens_list = [
        model.tokenizer.encode(text)
        + [model.tokenizer.eot]
        for text in texts
    ]
    if 'oracle' in params.method:
        text_tokens_tensor = [
            torch.LongTensor(text_tokens) for text_tokens in text_tokens_list
        ]
        target_lengths = torch.LongTensor([len(tokens) for tokens in text_tokens_tensor])
    else:
        target_lengths = None
    
    if not params.ctc_only:
        if params.method == "ar":
            results = model.whisper_model.decode(feature, params.decoding_options)
            hyps = [result.text for result in results]
            pred_tokens = [result.tokens for result in results]
        # elif params.method == "diagnostic":
        #     from typing import Any
        #     from torch.nn.functional import pad as pad_tensor

        #     def _batch_tensors(tensors: List[torch.Tensor], pad_value: Any) -> torch.Tensor:
        #         padding_size = max(tensor.shape[0] for tensor in tensors)
        #         dims = len(tensors[0].shape)
        #         padded_tensors = []
        #         for tensor in tensors:
        #             padding = [0] * 2 * dims
        #             padding[-1] = padding_size - tensor.shape[0]
        #             padded_tensors.append(pad_tensor(tensor, padding, "constant", pad_value))
        #         return torch.stack([tensor for tensor in padded_tensors], dim=0)
        #     input_text_tokens_list = [
        #         [model.tokenizer.sot] + 
        #         model.tokenizer.encode(text)
        #         for text in texts
        #     ]
        #     # convert it to torch tensor
        #     input_text_tokens_list = [
        #         torch.LongTensor(text_tokens) for text_tokens in input_text_tokens_list
        #     ]
        #     prev_outputs_tokens = _batch_tensors(
        #         [tokens for tokens in input_text_tokens_list], pad_value=model.pad_id
        #     )
        #     hyps, pred_tokens = model.decode_diagnostic(feature, feature_len, prev_outputs_tokens, target_lengths)
        # elif params.method == "mask_predict_oracle":
        #     hyps, pred_tokens = model.decode_mask_predict(feature, feature_len, target_lengths)
        #     pred_tokens = [tokens[:tokens.index(50257)] if 50257 in tokens else tokens for tokens in pred_tokens]
        # elif params.method == "mask_predict_oracle_iterative":
        #     hyps, pred_tokens = model.decode_mask_predict_iterative(feature, feature_len, target_lengths)
        #     pred_tokens = [tokens[:tokens.index(50257)] if 50257 in tokens else tokens for tokens in pred_tokens]          
        else:
            assert 'cif' in params.method
            # hyps, pred_tokens = model.decode_cif_rescore(feature, feature_len, target_lengths)
            hyps, pred_tokens = model.decode(feature, feature_len, target_lengths)
            # remove all tokens after the first 50257 in the pred_tokens
            pred_tokens = [tokens[:tokens.index(50257)] if 50257 in tokens else tokens for tokens in pred_tokens]
    else:
        hyps, pred_tokens = model.ctc_decode(feature, feature_len)

    # hyps = [result.text for result in results]

    hyps = remove_punctuation(hyps)
    hyps = to_simple(hyps)
    hyps = [params.normalizer.normalize(hyp) for hyp in hyps]
    print(hyps)
    return {params.method: hyps}, pred_tokens, text_tokens_list


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
        dl:
            The dataloader.
        params:
            It is returned by :func:`get_params`.
        model:
            The neural model.
    Returns:
        Return a dict, whose key may be "beam-search".
    """

    results = []

    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    all_preds, all_labels, all_cut_ids = [], [], []
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict, pred_tokens, label_tokens = decode_one_batch(
            params=params,
            model=model,
            batch=batch,
        )
        all_preds.extend(pred_tokens)
        for label_token in label_tokens:
            all_labels.append(label_token[:-1])
        all_cut_ids.extend(cut_ids)

        for lm_scale, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_text = normalize_text_alimeeting(ref_text)
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words))

            results[lm_scale].extend(this_batch)

        num_cuts += len(batch["supervisions"]["text"])

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results, all_preds, all_labels, all_cut_ids


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):

    enable_log = True
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.exp_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        # results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        if enable_log:
            logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.exp_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        # we compute CER for aishell dataset.
        results_char = []
        for res in results:
            results_char.append((res[0], list("".join(res[1])), list("".join(res[2]))))
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results_char, enable_log=enable_log
            )
            test_set_wers[key] = wer

        if enable_log:
            logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.exp_dir / f"cer-summary-{test_set_name}-{params.suffix}.txt"
    with open(errs_info, "w") as f:
        print("settings\tCER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, CER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)

def save_token_results(
    params: AttributeDict,
    test_set_name: str,
    all_preds: List[List[int]],
    all_labels: List[List[int]],
    all_cut_ids: List[str],
):
    enable_log = True
    test_set_wers = dict()
    results_char = []
    for cut_id, pred, label in zip(all_cut_ids, all_preds, all_labels):
        # convert from int to str
        pred = [str(p) for p in pred]
        label = [str(l) for l in label]
        results_char.append((cut_id, label, pred))
    errs_filename = (
        params.exp_dir / f"errs-{test_set_name}-token-{params.method}-{params.suffix}.txt"
    )
    with open(errs_filename, "w") as f:
        wer = write_error_stats(
            f, f"{test_set_name}-token", results_char, enable_log=enable_log
        )
        test_set_wers["token"] = wer

    if enable_log:
        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.exp_dir / f"cer-summary-{test_set_name}-token-{params.method}-{params.suffix}.txt"
    # calculate the sentence level token num accuracy
    # e.g. len(pred) = 10, len(label) = 10, then the total_sentence_num += 1
    total_sentence_num = 0
    total_tolerance_num = 0
    for pred, label in zip(all_preds, all_labels):
        if len(pred) - len(label) > 1 or len(label) - len(pred) > 1:
            print(23333333333333333333333333333, len(pred), len(label))
        elif len(pred) - len(label) == 1 or len(label) - len(pred) == 1:
            total_tolerance_num += 1
        if len(pred) == len(label):
            total_sentence_num += 1
    sentence_accuracy = total_sentence_num / len(all_preds)
    tolerance_sentence_accuracy = (total_sentence_num + total_tolerance_num) / len(all_preds)
    logging.info(f"Total sentence num: {len(all_preds)}, sentence accuracy: {sentence_accuracy}")
    logging.info(f"Total sentence num: {len(all_preds)}, sentence accuracy with tolerance: {tolerance_sentence_accuracy}")

    with open(errs_info, "w") as f:
        print("settings\tCER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)
        print(f"Total sentence num: {len(all_preds)}, sentence accuracy: {sentence_accuracy}", file=f)

    s = "\nFor {}, CER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)



@torch.no_grad()
def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))
    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"
    setup_logger(
        f"{params.exp_dir}/log-{params.method}-beam{params.beam_size}/log-decode-{params.suffix}"
    )

    options = whisper.DecodingOptions(
        task="transcribe",
        language="zh",
        without_timestamps=True,
        beam_size=params.beam_size,
    )
    params.decoding_options = options
    params.cleaner = BasicTextNormalizer()
    params.normalizer = Normalizer()

    logging.info("Decoding started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logging.info(f"device: {device}")

    # if params.remove_whisper_encoder_input_length_restriction:
    replace_whisper_encoder_forward()
    replace_whisper_decoder_forward()
    model = whisper.load_model(params.model_name, "cpu")
    model = ParaWhisper(model, sampler=False)
    if params.epoch > 0:
        if params.avg > 1:
            start = params.epoch - params.avg
            assert start >= 1, start
            checkpoint = torch.load(
                f"{params.exp_dir}/epoch-{params.epoch}.pt", map_location="cpu"
            )
            if "model" not in checkpoint:
                # deepspeed converted checkpoint only contains model state_dict
                filenames = [
                    f"{params.exp_dir}/epoch-{epoch}.pt"
                    for epoch in range(start, params.epoch + 1)
                ]
                model.load_state_dict(average_checkpoints(filenames))
            else:
                filename_start = f"{params.exp_dir}/epoch-{start}.pt"
                filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
                logging.info(
                    f"Calculating the averaged model over epoch range from "
                    f"{start} (excluded) to {params.epoch}"
                )
                model.to(device)
                model.load_state_dict(
                    average_checkpoints_with_averaged_model(
                        filename_start=filename_start,
                        filename_end=filename_end,
                        device=device,
                    )
                )
            # save checkpoints
            filename = f"{params.exp_dir}/epoch-{params.epoch}-avg-{params.avg}.pt"
            torch.save(model.state_dict(), filename)
        else:
            checkpoint = torch.load(
                f"{params.exp_dir}/epoch-{params.epoch}.pt", map_location="cpu"
            )
            if "model" not in checkpoint:
                model.load_state_dict(checkpoint, strict=False)
            else:
                load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    model.to(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True

    data_module = AsrDataModule(args)
    multi_dataset = MultiDataset(args.manifest_dir, "", args.start_index, args.end_index)

    def remove_long_utt(c: Cut):
        # Keep only utterances with duration in 30 seconds
        #
        if c.duration > 30.0:
            # logging.warning(
            #    f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False
        return True

    test_sets_cuts = multi_dataset.test_cuts()

    test_sets = test_sets_cuts.keys()
    test_dls = [
        data_module.test_dataloaders(test_sets_cuts[cuts_name].filter(remove_long_utt))
        for cuts_name in test_sets
    ]

    for test_set, test_dl in zip(test_sets, test_dls):
        results_dict, all_preds, all_labels, all_cut_ids = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
        )
        save_results(params=params, test_set_name=test_set, results_dict=results_dict)
        save_token_results(params=params, test_set_name=test_set, all_preds=all_preds, all_labels=all_labels, all_cut_ids=all_cut_ids)

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
