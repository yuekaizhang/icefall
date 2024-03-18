
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

def generate_errors(texts, substitution_rate, chinese_chars):
    # select according to the substitution rate
    indices = random.sample(range(len(texts)), int(len(texts) * substitution_rate))
    for idx in indices:
        texts[idx] = substitute_chinese_chars(texts[idx], chinese_chars)
    return texts

def substitute_chinese_chars(text, chinese_chars, portion=0.15):
    # select portion of the chars, substitute with a random Chinese character
    chars = list(text)
    num_chars = len(chars)
    # at least substitute one character
    num_substitute = max(1, int(num_chars * portion))
    # generate random indices without same index
    indices = random.sample(range(num_chars), num_substitute)
    for idx in indices:
        chars[idx] = random.choice(chinese_chars)
    return ''.join(chars)


if __name__ == "__main__":
    # token_path = '/mnt/samsung-t7/yuekai/asr/icefall/egs/speechio/ASR/zipformer/icefall-asr-multi-zh-hans-zipformer-ctc-2023-10-24/data/lang_bpe_2000/tokens.txt'
    token_path = '/mnt/samsung-t7/yuekai/asr/icefall/egs/speechio/ASR/zipformer/icefall-asr-aishell-zipformer-small-2023-10-24/data/lang_char/tokens.txt'
    chinese_chars = load_chinese_chars(token_path)
    # print(chinese_chars)
    # print(substitute_chinese_chars('我是中国人', chinese_chars))
    # print(substitute_chinese_chars('今天天气不错我们一起去爬山', chinese_chars))
    test_texts = ['我是中国人', '今天天气不错我们一起去爬山']
    print(generate_errors(test_texts, 0.5, chinese_chars))


    

