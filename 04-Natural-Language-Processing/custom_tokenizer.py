import regex as re
import json
import os
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPT-2 pre-tokenization regex
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer trained from scratch.
    Implements the algorithm described in 'Neural Machine Translation of Rare Words with Subword Units'.
    Includes GPT-2 style pre-tokenization regex.
    """
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
        self.special_tokens: Dict[str, int] = {}
        self.inverse_vocab: Dict[bytes, int] = {}
        self.pattern = re.compile(GPT2_SPLIT_PATTERN)

    def _get_stats(self, ids: List[int], counts: Dict[Tuple[int, int], int] = None) -> Dict[Tuple[int, int], int]:
        """
        Count frequency of adjacent pairs in the token list.
        """
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """
        Replace all occurrences of 'pair' in 'ids' with 'idx'.
        """
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text: str, verbose: bool = True):
        """
        Train the BPE tokenizer on the provided text.
        Args:
            text: Large string of training data.
            verbose: Whether to show progress bar.
        """
        logger.info(f"Training BPE tokenizer. Target vocab size: {self.vocab_size}")
        
        # Pre-tokenize (split by regex)
        text_chunks = re.findall(self.pattern, text)
        
        # Convert to UTF-8 bytes and then to integers
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        
        # Flatten for initial stats calculation (approximation for speed)
        # In a real rigorous implementation, we'd handle boundaries carefully.
        # Here we flatten but re-chunking logic implies we merge within chunks.
        # For simplicity in this dense implementation, we'll process chunks iteratively or flatten.
        # Let's flatten to a single list of ints for the main merge loop to be standard BPE.
        # Note: GPT-2 merges *within* regex chunks, not across.
        # We will follow GPT-2 approach: list of lists.
        
        num_merges = self.vocab_size - 256
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for i in tqdm(range(num_merges), disable=not verbose, desc="Merging BPE Pairs"):
            stats = {}
            for chunk_ids in ids:
                self._get_stats(chunk_ids, stats)
                
            if not stats:
                logger.warning(f"No more pairs to merge. Stopping early at {256+i} tokens.")
                break
                
            # Find most frequent pair
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # Record merge
            self.merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
            # Apply merge to all chunks
            ids = [self._merge(chunk_ids, pair, idx) for chunk_ids in ids]
            
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        logger.info("Training complete.")

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs.
        """
        text_chunks = re.findall(self.pattern, text)
        ids = []
        
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                stats = self._get_stats(chunk_ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                
                if pair not in self.merges:
                    break # No more mergeable pairs
                
                idx = self.merges[pair]
                chunk_ids = self._merge(chunk_ids, pair, idx)
            ids.extend(chunk_ids)
            
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back to a string.
        """
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def save_model(self, path_prefix: str):
        """
        Save the tokenizer model (vocab and merges).
        """
        model_file = f"{path_prefix}.model"
        vocab_file = f"{path_prefix}.vocab"
        
        with open(model_file, 'wb') as f:
            pickle.dump(self.merges, f)
            
        with open(vocab_file, 'w', encoding='utf-8') as f:
            # Saving vocab as json for readability/interop
            # Convert bytes keys to latin-1 strings for JSON
            json_vocab = {k: v.decode('latin-1') for k, v in self.vocab.items()}
            json.dump(json_vocab, f, indent=2)
            
        logger.info(f"Model saved to {model_file} and {vocab_file}")

    def load_model(self, path_prefix: str):
        """
        Load a saved tokenizer model.
        """
        model_file = f"{path_prefix}.model"
        vocab_file = f"{path_prefix}.vocab"
        
        with open(model_file, 'rb') as f:
            self.merges = pickle.load(f)
            
        with open(vocab_file, 'r', encoding='utf-8') as f:
            json_vocab = json.load(f)
            self.vocab = {int(k): v.encode('latin-1') for k, v in json_vocab.items()}
            
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        logger.info(f"Model loaded from {model_file}")

class TokenizerPipeline:
    """
    Wrapper for preprocessing and tokenization pipelines.
    """
    def __init__(self, tokenizer: BPETokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, texts: List[str], padding: bool = True, truncation: bool = True) -> Dict[str, List[List[int]]]:
        batch_ids = []
        attention_masks = []
        
        for text in texts:
            ids = self.tokenizer.encode(text)
            
            if truncation and len(ids) > self.max_length:
                ids = ids[:self.max_length]
            
            mask = [1] * len(ids)
            
            if padding and len(ids) < self.max_length:
                pad_len = self.max_length - len(ids)
                # Assuming 0 is not a special pad token unless defined, but bytes 0 is null.
                # Usually we define a specific PAD token. For this raw implementation, we'll use 0.
                ids.extend([0] * pad_len)
                mask.extend([0] * pad_len)
                
            batch_ids.append(ids)
            attention_masks.append(mask)
            
        return {
            "input_ids": batch_ids,
            "attention_mask": attention_masks
        }

if __name__ == "__main__":
    # Test stub
    sample_text = """
    The Transformer is a deep learning architecture that relies on the parallel multi-head attention mechanism. 
    It is notable for requiring less training time than previous recurrent neural architectures, 
    such as long short-term memory (LSTM), and has been prevalently adopted for training large language models 
    on large (language) datasets, such as the Wikipedia Corpus and Common Crawl.
    """ * 10 # Make it longer
    
    tokenizer = BPETokenizer(vocab_size=300) # Small vocab for testing
    tokenizer.train(sample_text, verbose=True)
    
    encoded = tokenizer.encode("The Transformer is a deep learning architecture")
    logger.info(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    logger.info(f"Decoded: {decoded}")
    
    assert "Transformer" in decoded
    
    tokenizer.save_model("custom_bpe")
