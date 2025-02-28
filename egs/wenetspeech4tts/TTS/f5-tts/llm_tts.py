# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# https://github.com/SparkAudio/Spark-TTS/blob/main/cli/SparkTTS.py

import re
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMTTS:
    """
    LLM-TTS for text-to-speech generation.
    """

    def __init__(
        self,
        model_dir: Path,
        tokenizer_dir: Path,
        s3_tokenizer_name: str,
        device: torch.device,
    ):
        """
        Initializes the LLMTTS model with the provided configurations and device.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
        """
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation="flash_attention_2",
        )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        new_tokens = [f"<|s_{i}|>" for i in range(6561)] + [
            "<|SPEECH_GENERATION_START|>"
        ]
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.assistant_index = tokenizer.convert_tokens_to_ids("assistant")

    @torch.no_grad()
    def inference_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.

        Returns:
            torch.Tensor: Generated waveform as a tensor.
        """
        # Generate speech using the model
        generated_ids = self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            max_new_tokens=1024,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        results = []
        generated_ids = generated_ids.cpu().tolist()
        for i in range(len(generated_ids)):
            assistant_index = generated_ids[i].index(self.assistant_index)
            padding_index = len(generated_ids[i])
            result = generated_ids[i][assistant_index + 2 :]
            result = [token - 151665 for token in result]
            result = [token for token in result if token >= 0]
            results.append(result)
        return results
