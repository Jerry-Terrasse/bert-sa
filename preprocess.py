import json
import sys
import unicodedata
from tqdm import tqdm

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def punc(text: str):
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1
    return ["".join(x) for x in output]

data = json.load(open(sys.argv[1], 'r'))
results = []
for r, q in tqdm(data):
    words = q.split(' ')
    q = ' '.join(sum((punc(w) for w in words), start=[]))
    results.append((r, q))
    
json.dump(results, open(sys.argv[2], 'w'))