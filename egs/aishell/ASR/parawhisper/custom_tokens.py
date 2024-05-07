
def load_new_tokens_dict(filename):
    char2dict, dict2char = {}, {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # one-to-one mapping
            char, token = line.split()
            token = int(token)
            char2dict[char] = token
            dict2char[token] = char
    return char2dict, dict2char
            

def encode_with_dict(text_str, char2dict):
    encoded = []
    for char in text_str:
        if char in char2dict:
            encoded.append(char2dict[char])
        else:
            encoded.append(char2dict["<|unk|>"])
    return encoded

def decode_with_dict(encoded, dict2char):
    decoded = ""
    for token in encoded:
        if token in dict2char:
            decoded += dict2char[token]
        else:
            decoded += "<|unk|>"
    return decoded

# make above into a class and test it
class CustomTokenizer:
    def __init__(self, token_path):
        self.char2dict, self.dict2char = load_new_tokens_dict(token_path)
        self.vocab_size = len(self.char2dict)
        self.eot = 2
        self.sot = 1
        self.pad = 0

    def encode(self, text_str):
        return encode_with_dict(text_str, self.char2dict)

    def decode(self, encoded):
        return decode_with_dict(encoded, self.dict2char)

if __name__ == "__main__":
    token_path = "./aishell_tokens_wenet.txt"
    char2dict, dict2char = load_new_tokens_dict(token_path)
    #print(char2dict, dict2char)
    test_str = "今天天气不错,我们一起去爬山吧"
    encoded = encode_with_dict(test_str, char2dict)
    #print(encoded)
    decoded = decode_with_dict(encoded, dict2char)
    #print(decoded)

    custom_tokenizer = CustomTokenizer(token_path)
    encoded = custom_tokenizer.encode(test_str)
    print(encoded)
    decoded = custom_tokenizer.decode(encoded)
    print(decoded)
