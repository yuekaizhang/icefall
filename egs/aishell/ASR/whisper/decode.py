#!/usr/bin/env python3
# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo,
#                                            Fangjun Kuang,
#                                            Wei Kang)
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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import whisper
from whisper.normalizers import BasicTextNormalizer
import k2
import torch
import torch.nn as nn
from asr_datamodule import AishellAsrDataModule

#from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler
from icefall.checkpoint import average_checkpoints, load_checkpoint, average_checkpoints_with_averaged_model
from icefall.decode import (
    get_lattice,
    nbest_decoding,
    nbest_oracle,
    one_best_decoding,
    rescore_with_attention_decoder,
)
from icefall.env import get_env_info
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
    write_error_stats,
)
from zhconv import convert
from tn.chinese.normalizer import Normalizer
import re

def remove_punctuation(text: str or List[str]):
  # https://github.com/yeyupiaoling/Whisper-Finetune/blob/master/utils/data_utils.py
    punctuation = '!,.;:?、！，。；：？'
    if isinstance(text, str):
        text = re.sub(r'[{}]+'.format(punctuation), '', text).strip()
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = re.sub(r'[{}]+'.format(punctuation), '', t).strip()
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'不支持该类型{type(text)}')


# 将繁体中文总成简体中文
def to_simple(text: str or List[str]):
    if isinstance(text, str):
        text = convert(text, 'zh-cn')
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = convert(t, 'zh-cn')
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'不支持该类型{type(text)}')

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
        default="beam-search",
        help="""Decoding method.
        Supported values are:
          - beam-search
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
  
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            # parameters for conformer
            "subsampling_factor": 4,
            "feature_dim": 80,
            "nhead": 4,
            "attention_dim": 512,
            "num_encoder_layers": 12,
            "num_decoder_layers": 6,
            "vgg_frontend": False,
            "use_feat_batchnorm": True,
            # parameters for decoder
            "search_beam": 20,
            "output_beam": 7,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
            "env_info": get_env_info(),
        }
    )
    return params


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
) -> Dict[str, List[List[int]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if decoding method is 1best, the key is the string `no_rescore`.
               If attention rescoring is used, the key is the string
               `ngram_lm_scale_xxx_attention_scale_xxx`, where `xxx` is the
               value of `lm_scale` and `attention_scale`. An example key is
               `ngram_lm_scale_0.7_attention_scale_0.5`
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.method is "1best", it uses 1best decoding without LM rescoring.
        - params.method is "nbest", it uses nbest decoding without LM rescoring.
        - params.method is "attention-decoder", it uses attention rescoring.

      model:
        The neural model.
      HLG:
        The decoding graph. Used when params.method is NOT ctc-decoding.
      H:
        The ctc topo. Used only when params.method is ctc-decoding.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      lexicon:
        It contains the token symbol table and the word symbol table.
      sos_id:
        The token ID of the SOS.
      eos_id:
        The token ID of the EOS.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    dtype = torch.float16
    device = torch.device("cuda")

    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device, dtype=dtype).transpose(1, 2)
    # pad feature to T = 3000
    T = 3000
    if feature.shape[2] < T:
      feature = torch.cat([feature, torch.zeros(feature.shape[0], feature.shape[1], T - feature.shape[2]).to(device, dtype=dtype)], 2)
    print(feature.shape,23333)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_len = supervisions["num_frames"]
    feature_len = feature_len.to(device, dtype=dtype)
    results = model.decode(feature, params.decoding_options)
    hyps = [result.text for result in results]
    
    hyps = remove_punctuation(hyps)
    hyps = to_simple(hyps)

    hyps = [params.normalizer.normalize(hyp) for hyp in hyps]
    print(hyps, 233333333)

    key = "beam-search"

    return {key: hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      HLG:
        The decoding graph. Used when params.method is NOT ctc-decoding.
      H:
        The ctc topo. Used only when params.method is ctc-decoding.
      lexicon:
        It contains the token symbol table and the word symbol table.
      sos_id:
        The token ID for SOS.
      eos_id:
        The token ID for EOS.
    Returns:
      Return a dict, whose key may be "no-rescore" if the decoding method is
      1best or it may be "ngram_lm_scale_0.7_attention_scale_0.5" if attention
      rescoring is used. Its value is a list of tuples. Each tuple contains two
      elements: The first is the reference transcript, and the second is the
      predicted result.
    """
    results = []

    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            batch=batch,
        )

        for lm_scale, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words))

            results[lm_scale].extend(this_batch)

        num_cuts += len(batch["supervisions"]["text"])

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):

    enable_log = True
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = params.exp_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        if enable_log:
            logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.exp_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
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


@torch.no_grad()
def main():
    parser = get_parser()
    AishellAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))
    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"
    setup_logger(f"{params.exp_dir}/log-{params.method}-beam{params.beam_size}/log-decode-{params.suffix}")

    options = whisper.DecodingOptions(task="transcribe", language="zh", without_timestamps=True, beam_size=params.beam_size)
    params.decoding_options = options
    params.cleaner = BasicTextNormalizer()
    params.normalizer = Normalizer()
  
    logging.info("Decoding started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logging.info(f"device: {device}")

    model = whisper.load_model(params.model_name)
    if params.epoch > 0:
      if params.avg > 1:
        start = params.epoch - params.avg
        assert start >= 1, start
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
      else:
        checkpoint = torch.load(f"{params.exp_dir}/epoch-{params.epoch}.pt", map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        #load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    model.to(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")


    # we need cut ids to display recognition results.
    args.return_cuts = True
    aishell = AishellAsrDataModule(args)
    test_cuts = aishell.test_cuts()
    test_dl = aishell.test_dataloaders(test_cuts)

    test_sets = ["test"]
    test_dls = [test_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
        )

        save_results(params=params, test_set_name=test_set, results_dict=results_dict)

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()