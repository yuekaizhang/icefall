
import random
import re
# import whisper
from pathlib import Path

def load_chinese_chars(filename):
    chinese_chars = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Check if line contains Chinese characters
            if re.search('[\u4e00-\u9fff]', line):
                # Assume the Chinese character is always the first part before the space
                char = line.split()[0]
                chinese_chars.append(char)
    chinese_chars = chinese_chars[:2000]
    # tokenizer = whisper.tokenizer.get_tokenizer(
    #     True,
    #     num_languages=100,
    #     language="zh",
    #     task="transcribe",
    # )
    # for i in range(0, 2000):
    #     chinese_chars[i] = (tokenizer.encode(chinese_chars[i]), chinese_chars[i])
    return chinese_chars

def substitute_chinese_chars(text, chinese_chars):
    text = list(text)
    for i in range(len(text)):
        for j in range(len(chinese_chars)):
            if text[i] == chinese_chars[j][1]:
                text[i] = chinese_chars[j][0]
                break
    return text
    
if __name__ == "__main__":
    # token_path = '/mnt/samsung-t7/yuekai/asr/icefall/egs/speechio/ASR/zipformer/icefall-asr-multi-zh-hans-zipformer-ctc-2023-10-24/data/lang_bpe_2000/tokens.txt'
    token_path = '/mnt/samsung-t7/yuekai/asr/icefall/egs/speechio/ASR/zipformer/icefall-asr-aishell-zipformer-small-2023-10-24/data/lang_char/tokens.txt'
    chinese_chars = load_chinese_chars(token_path)
    # print(chinese_chars)
    print(substitute_chinese_chars('我是中国人', chinese_chars))

    # tokenizer = whisper.tokenizer.get_tokenizer(
    #     True,
    #     num_languages=100,
    #     language="zh",
    #     task="transcribe",
    # )
    # text = "禁天天气不对"
    # for char in text:
    #     token_id = tokenizer.encode(char)
    #     print(char, token_id)
    # token_ids = tokenizer.encode(text)
    # for token_id in token_ids:
    #     print(tokenizer.decode([token_id]), token_id)

    # with open("whisper_tokens.txt", 'w') as file:
    #     for i in range(0, 52000):
    #         symbol = tokenizer.decode([i])
    #         print(f"{symbol} {i}")
    #         file.write(f"{symbol} {i}\n")
    