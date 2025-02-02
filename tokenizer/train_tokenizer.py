import re
import unicodedata
import json
from collections import Counter

def normalize_text(text):
    """Normalize text by converting to lowercase, removing special characters, and extra spaces."""
    text = text.lower()  # Convert to lowercase
    text = unicodedata.normalize("NFKC", text)  # Normalize unicode characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def get_vocab(text):
    """Create vocabulary with character-level tokens."""
    words = text.split()
    vocab = Counter([" ".join(word) + " _" for word in words])  # Add space between characters and special end token
    return vocab

def get_stats(vocab):
    """Count frequency of adjacent character pairs in vocabulary."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merge the most frequent character pair into a new subword."""
    new_vocab = {}
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word, freq in vocab.items():
        new_word = pattern.sub("".join(pair), word)
        new_vocab[new_word] = freq
    return new_vocab

def train_bpe(text, num_merges=10):
    """Train Byte Pair Encoding (BPE) on input text."""
    vocab = get_vocab(text)
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
    return vocab, merges

def encode_text(text, merges):
    """Encode text into subword tokens using learned BPE merges."""
    words = text.split()
    tokens = [" ".join(word) + " _" for word in words]
    for pair in merges:
        pattern = re.compile(re.escape(" ".join(pair)))
        tokens = [pattern.sub("".join(pair), token) for token in tokens]
    return tokens

def decode_tokens(tokens, merges):
    """Decode subword tokens back to original text."""
    for pair in reversed(merges):  # Reverse merge order
        pattern = re.compile(re.escape("".join(pair)))
        tokens = [pattern.sub(" ".join(pair), token) for token in tokens]
    decoded_text = " ".join(tokens).replace(" _", "").strip()
    return decoded_text

def evaluate_tokenizer(text, merges):
    """Evaluate tokenizer performance."""
    words = text.split()
    original_token_count = len(words)
    encoded_tokens = encode_text(text, merges)
    compressed_token_count = len(encoded_tokens)
    compression_ratio = original_token_count / compressed_token_count if compressed_token_count > 0 else 0
    return {
        "Original Token Count": original_token_count,
        "Compressed Token Count": compressed_token_count,
        "Compression Ratio": compression_ratio
    }

def handle_oov_words(text, vocab):
    """Break unseen words into character-level tokens."""
    words = text.split()
    return [word if word in vocab else " ".join(word) for word in words]

def save_tokenizer(vocab, merges, filename="tokenizer.json"):
    """Save trained tokenizer to a file."""
    with open(filename, "w") as f:
        json.dump({"vocab": list(vocab.keys()), "merges": merges}, f)

def load_tokenizer(filename="tokenizer.json"):
    """Load tokenizer from a file."""
    with open(filename, "r") as f:
        data = json.load(f)
    return set(data["vocab"]), data["merges"]

# Sample Corpus
sample_text = """
Tokenization is the process of breaking down text into smaller units.
These units, called tokens, can be words, subwords, or even characters.
It's a fundamental step in Natural Language Processing (NLP).
"""

# Normalize the text
normalized_text = normalize_text(sample_text)

# Train BPE Tokenizer
bpe_vocab, bpe_merges = train_bpe(normalized_text, num_merges=10)

# Save and Load tokenizer
save_tokenizer(bpe_vocab, bpe_merges)
loaded_vocab, loaded_merges = load_tokenizer()

# Encode a sample sentence
encoded_tokens = encode_text("Tokenization is a process", loaded_merges)
print("Encoded Tokens:", encoded_tokens)

# Handle OOV words
oov_handled_text = handle_oov_words("unknownword example", loaded_vocab)
print("OOV Handled:", oov_handled_text)

# Decode back to text
decoded_text = decode_tokens(encoded_tokens, loaded_merges)
print("Decoded Text:", decoded_text)

# Evaluate tokenizer
evaluation_metrics = evaluate_tokenizer(normalized_text, loaded_merges)
print("Tokenizer Evaluation:", evaluation_metrics)
