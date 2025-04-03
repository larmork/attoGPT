from collections import Counter
import pdftotext # type: ignore

def get_stats(ids):
    pairs = zip(ids, ids[1:])
    counts = Counter(pairs)
    return counts

def print_sorted_stats(stats):
    sorted_stats = sorted(((v, k) for k, v in stats.items()), reverse=True)
    for count, pair in sorted_stats:
        print(f"Pair: {pair}, Count: {count}")

def read_in_chunks(file_path, chunk_size=1024*1024):
    with open(file_path, "r") as f:
        pdf = pdftotext.PDF(f)
        text = "\n\n".join(pdf) 
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def process_file(file_path):
    for chunk in read_in_chunks(file_path):
        tokens = chunk.encode("utf-8")
        tokens = list(map(int, tokens))
    return tokens

tokens = process_file(file_path)

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

vocab_size = 1000
num_merges = vocab_size - 256
ids = list(tokens)

merges = {}

for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"Merging pair {pair} into new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def process_file_chunks(file_path):
    encoded_chunks = []
    decoded_chunks = []

    for chunk in read_in_chunks(file_path):
        encoded_chunks.append(encode(chunk))

    encoded_data = [token for chunk in encoded_chunks for token in chunk]

    decoded_chunks.append(decode(encoded_data))

    final_decoded_text = ''.join(decoded_chunks)
    return final_decoded_text

file_path = "data/wikitext-103/wiki.valid.tokens"

final_decoded_text = process_file_chunks(file_path)

with open(file_path, "r") as f:
    original_text = f.read()

print(final_decoded_text == original_text)