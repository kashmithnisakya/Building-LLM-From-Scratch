# Tokenizer 

Building a tokenizer from scratch involves several key steps, including:

1. Text Normalization – Cleaning and preprocessing the text.
2. Tokenization Strategy – Choosing a tokenization approach (word-level, character-level, subword-level).
3. Building a Vocabulary – Creating a set of tokens.
4. Token Mapping – Assigning unique IDs to each token.
5. Encoding and Decoding – Converting text to token IDs and back.
6. Optimization – Using frequency-based or learned tokenization approaches.

## Step 1: Text Normalization

Let's start by implementing text normalization, which includes:

1. Converting text to lowercase
2. Removing special characters and punctuation
3. Replacing multiple spaces with a single space
4. Handling unicode normalization (optional)


## Step 2: Tokenization Strategy
Now that we have normalized the text, we need to decide on a tokenization approach. The most effective strategy for training an LLM is subword tokenization, such as Byte Pair Encoding (BPE), WordPiece, or Unigram.

Since BPE is widely used in LLMs (GPT, BERT, etc.), let’s implement it.

## Step 3: Implementing Byte Pair Encoding (BPE)
BPE is an adaptive tokenization technique that:

Starts with character-level tokens.
Merges the most frequent adjacent character pairs into subwords.
Continues merging until a vocabulary size limit is reached.

## Step 4: Encoding Text Using the Trained Tokenizer
Now that we have trained the BPE tokenizer, the next step is encoding text using the learned vocabulary.

Encoding Process:
Tokenize input text using character-level representation.
Apply learned merges from the trained BPE vocabulary to create subwords.
Convert tokens into their corresponding token IDs.

## Step 5: Decoding Tokens Back to Text
Now that we have implemented encoding, the final step is to create a decoder function that reconstructs the original text from BPE tokens.

Decoding Process
Start with encoded tokens.
Reverse the learned BPE merges in order to reconstruct words.
Remove special end markers (_) if necessary.
Convert tokenized text back to a readable format.

## Step 6: Evaluating the Tokenizer
To ensure our tokenizer works well, we need to:

Measure compression efficiency (i.e., how well BPE reduces the number of tokens).
Check if it handles unseen words gracefully.
Optimize performance (e.g., speed, memory usage).