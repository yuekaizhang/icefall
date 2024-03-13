#!/usr/bin/env python3
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang, Zengwei Yao)
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
import math
from typing import List

import k2
import kaldifeat
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

import sentencepiece as spm

from icefall.checkpoint import load_checkpoint
from icefall.utils import AttributeDict
from train import get_model

from beam_search import greedy_search_batch

"""
This script is a wrapper for the jit ASR model.
Usage:
  ./whisper/assistant_model_jit.py \
  --nn-model-filename ./zipformer/icefall-asr-aishell-zipformer-small-2023-10-24/exp/jit_script.pt \
  --tokens ./zipformer/icefall-asr-aishell-zipformer-small-2023-10-24/data/lang_char/tokens.txt \
  --decoding-method greedy_search \
  --sound-file ./zipformer/icefall-asr-aishell-zipformer-small-2023-10-24/test_wavs/BAC009S0764W0121.wav
"""
class AssistantASRModel:
    """
    This class is a wrapper for the jit ASR model.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        self.device = device
        #model = torch.jit.load(args.nn_model_filename)
        #model.eval()
        #model.to(device)
        #self.model = model
        # params = AttributeDict(
        #     {
        #         "use_transducer": True,
        #         "num_encoder_layers": [2, 2, 3, 4, 3, 2],
        #         "downsampling_factor": [1, 2, 4, 8, 4, 2],
        #         "feedforward_dim": [512, 768, 1024, 1536, 1024, 768],
        #         "num_heads": [4, 4, 4, 8, 4, 4],
        #         "encoder_dim": [192, 256, 384, 512, 384, 256],
        #         "query_head_dim": 32,
        #         "value_head_dim": 12,
        #         "pos_head_dim": 4,
        #         "pos_dim": 48,
        #         "encoder_unmasked_dim": [192, 192, 256, 256, 256, 192],
        #         "cnn_module_kernel": [31, 31, 15, 15, 15, 31],
        #         "decoder_dim": 512,
        #         "joiner_dim": 512,
        #         "causal": False,
        #         "chunk_size": [16, 32, 64, -1],
        #         "left_context_frames": [64, 128, 256, -1],
        #         "feature_dim": 80,
        #         "subsampling_factor": 4,  # not passed in, this is fixed.
        #     }
        # )
        params = AttributeDict(
            {
                "use_transducer": True,
                "use_ctc": False,
                "num_encoder_layers": "2,2,3,4,3,2",
                "downsampling_factor": "1,2,4,8,4,2",
                "feedforward_dim": "512,768,1024,1536,1024,768",
                "num_heads": "4,4,4,8,4,4",
                "encoder_dim": "192,256,384,512,384,256",
                "query_head_dim": "32",
                "value_head_dim": "12",
                "pos_head_dim": "4",
                "pos_dim": 48,
                "encoder_unmasked_dim": "192,192,256,256,256,192",
                "cnn_module_kernel": "31,31,15,15,15,31",
                "decoder_dim": 512,
                "joiner_dim": 512,
                "causal": False,
                "chunk_size": "16,32,64,-1",
                "left_context_frames": "64,128,256,-1",
                "feature_dim": 80,
                "subsampling_factor": 4,  # not passed in, this is fixed.
                "context_size": 2,
            }
        )

        sp = spm.SentencePieceProcessor()
        sp.load(args.tokens)
        self.sp = sp

        # <blk> and <unk> are defined in local/train_bpe_model.py
        params.blank_id = sp.piece_to_id("<blk>")
        params.unk_id = sp.piece_to_id("<unk>")
        params.vocab_size = sp.get_piece_size()
        model = get_model(params)
        load_checkpoint(args.nn_model_filename, model)

        self.model = model.to(device)

        # self.token_table = k2.SymbolTable.from_file(args.tokens)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )

        group.add_argument(
            "--nn-model-filename",
            type=str,
            required=True,
            help="Path to the torchscript model cpu_jit.pt",
        )

        group.add_argument(
            "--tokens",
            type=str,
            help="""Path to tokens.txt.""",
        )

        group.add_argument(
            "--decoding-method",
            type=str,
            default="greedy_search",
            help="""Possible values are:
            - greedy_search
            - modified_beam_search
            - fast_beam_search
            - fast_beam_search_nbest
            - fast_beam_search_nbest_oracle
            - fast_beam_search_nbest_LG
            If you use fast_beam_search_nbest_LG, you have to specify
            `--lang-dir`, which should contain `LG.pt`.
            """,
        )

        group.add_argument(
            "--sound-file",
            type=str,
            help="The input sound file(s) to transcribe. "
            "Supported formats are those supported by torchaudio.load(). "
            "For example, wav and flac are supported. "
            "The sample rate has to be 16kHz.",
        )

    def token_ids_to_words(self, token_ids: List[int]) -> str:
        text = ""
        for i in token_ids:
            text += self.token_table[i]
        return text.replace("▁", " ").strip()

    def decode(self, features: torch.Tensor, feature_lengths: torch.Tensor):
        encoder_out, encoder_out_lens = self.model.forward_encoder(
            features,
            feature_lengths,
        )
        if self.args.decoding_method == "greedy_search":
            tokens_ids = greedy_search_batch(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                blank_penalty=0.0,
            )
        else:
            raise ValueError(f"Unsupported decoding method: {self.args.decoding_method}")
        
        results = []
        # for hyp in token_ids:
        #     words = self.token_ids_to_words(hyp)
        #     results.append(words)
        for hyp in self.sp.decode(tokens_ids):
            results.append(hyp)
        return results

    # def greedy_search(
    #     self,
    #     encoder_out: torch.Tensor,
    #     encoder_out_lens: torch.Tensor,
    # ) -> List[List[int]]:
    #     """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    #     Args:
    #     model:
    #         The transducer model.
    #     encoder_out:
    #         A 3-D tensor of shape (N, T, C)
    #     encoder_out_lens:
    #         A 1-D tensor of shape (N,).
    #     Returns:
    #     Return the decoded results for each utterance.
    #     """
    #     assert encoder_out.ndim == 3
    #     assert encoder_out.size(0) >= 1, encoder_out.size(0)

    #     packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
    #         input=encoder_out,
    #         lengths=encoder_out_lens.cpu(),
    #         batch_first=True,
    #         enforce_sorted=False,
    #     )

    #     device = encoder_out.device
    #     blank_id = self.model.decoder.blank_id

    #     batch_size_list = packed_encoder_out.batch_sizes.tolist()
    #     N = encoder_out.size(0)

    #     assert torch.all(encoder_out_lens > 0), encoder_out_lens
    #     assert N == batch_size_list[0], (N, batch_size_list)

    #     context_size = self.model.decoder.context_size
    #     hyps = [[blank_id] * context_size for _ in range(N)]

    #     decoder_input = torch.tensor(
    #         hyps,
    #         device=device,
    #         dtype=torch.int64,
    #     )  # (N, context_size)

    #     decoder_out = self.model.decoder(
    #         decoder_input,
    #         need_pad=torch.tensor([False]),
    #     ).squeeze(1)

    #     offset = 0
    #     for batch_size in batch_size_list:
    #         start = offset
    #         end = offset + batch_size
    #         current_encoder_out = packed_encoder_out.data[start:end]
    #         current_encoder_out = current_encoder_out
    #         # current_encoder_out's shape: (batch_size, encoder_out_dim)
    #         offset = end

    #         decoder_out = decoder_out[:batch_size]

    #         logits = self.model.joiner(
    #             current_encoder_out,
    #             decoder_out,
    #         )
    #         # logits'shape (batch_size, vocab_size)

    #         assert logits.ndim == 2, logits.shape
    #         y = logits.argmax(dim=1).tolist()
    #         emitted = False
    #         for i, v in enumerate(y):
    #             if v != blank_id:
    #                 hyps[i].append(v)
    #                 emitted = True
    #         if emitted:
    #             # update decoder output
    #             decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
    #             decoder_input = torch.tensor(
    #                 decoder_input,
    #                 device=device,
    #                 dtype=torch.int64,
    #             )
    #             decoder_out = self.model.decoder(
    #                 decoder_input,
    #                 need_pad=torch.tensor([False]),
    #             )
    #             decoder_out = decoder_out.squeeze(1)

    #     sorted_ans = [h[context_size:] for h in hyps]
    #     ans = []
    #     unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    #     for i in range(N):
    #         ans.append(sorted_ans[unsorted_indices[i]])

    #     return ans

    # def decode(self, features: torch.Tensor, feature_lengths: torch.Tensor):
    #     encoder_out, encoder_out_lens = self.model.encoder(
    #         features=features,
    #         feature_lengths=feature_lengths,
    #     )
    #     if self.args.decoding_method == "greedy_search":
    #         token_ids = self.greedy_search(
    #             encoder_out=encoder_out,
    #             encoder_out_lens=encoder_out_lens,
    #         )
    #     else:
    #         raise ValueError(f"Unsupported decoding method: {self.args.decoding_method}")
        
    #     results = []
    #     # for hyp in token_ids:
    #     #     words = self.token_ids_to_words(hyp)
    #     #     results.append(words)
    #     for hyp in self.sp.decode(token_ids):
    #         results.append(hyp.split())
    #     return results

def read_sound_files(
    filenames: List[str], expected_sample_rate: float = 16000
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert (
            sample_rate == expected_sample_rate
        ), f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        # We use only the first channel
        ans.append(wave[0].contiguous())
    return ans

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    AssistantASRModel.add_arguments(parser)
    args = parser.parse_args()
    logging.info(vars(args))

    assistant = AssistantASRModel(args)

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = assistant.device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {args.sound_file}")
    waves = read_sound_files(
        filenames=[args.sound_file],
    )
    waves = [w.to(assistant.device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]

    features = pad_sequence(
        features,
        batch_first=True,
        padding_value=math.log(1e-10),
    )

    feature_lengths = torch.tensor(feature_lengths, device=assistant.device)

    hyps = assistant.decode(
        features=features.to(assistant.device),
        feature_lengths=feature_lengths,
    )

    logging.info(hyps)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
