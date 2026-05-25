from tqdm import tqdm
import pickle

class BPETokenizer:
    def __init__(self, vocab):
        """
        Initialize with existing vocabulary (stoi format)
        Args:
            vocab: Dictionary mapping tokens to indices (your existing stoi)
        """
        self.stoi = vocab.copy()
        self.itos = {v: k for k, v in self.stoi.items()}
        self.merge_rules = {}  # {(token1, token2): merged_token}
        self.bpe_vocab = set(self.stoi.keys())
        self.max_token_len = max(len(k) for k in self.stoi.keys())

        # Add individual characters if not present
        self._ensure_characters_in_vocab()

    def _ensure_characters_in_vocab(self):
        """Ensure all individual characters in vocab are present"""
        all_tokens = set(self.stoi.keys())
        all_chars = set(''.join([t for t in all_tokens if not t.startswith('[')]))
        for c in all_chars:
            if c not in self.stoi:
                idx = len(self.stoi)
                self.stoi[c] = idx
                self.itos[idx] = c
                self.bpe_vocab.add(c)

        self.max_token_len = max(len(k) for k in self.stoi.keys())

    def _tokenize(self, text):
        """Tokenize SMILES using existing vocab (same as SmilesTokenizer)"""
        tokens = []
        i = 0
        length = len(text)

        while i < length:
            # Handle bracketed expressions
            if text[i] == "[":
                j = i + 1
                while j < length and text[j] != "]":
                    j += 1
                if j < length:
                    token = text[i:j+1]
                    if token in self.stoi:
                        tokens.append(token)
                        i = j + 1
                        continue
                    else:
                        i += 1
                        continue

            # Try 2-character token
            if i + 1 < length and text[i:i+2] in self.stoi:
                tokens.append(text[i:i+2])
                i += 2
                continue

            # Try 1-character token
            if text[i] in self.stoi:
                tokens.append(text[i])
            else:
                # Fallback to character-level tokenization
                tokens.append(text[i])

            i += 1

        return tokens

    def train(self, corpus, num_merges=100):
        """
        Train BPE merges on a SMILES corpus
        Args:
            corpus: List of SMILES strings
            num_merges: Number of BPE merge operations to perform
        """
        word_freqs = defaultdict(int)
        for text in tqdm(corpus, desc="Building word frequencies"):
            tokens = self._tokenize(text)
            word = ''.join(tokens)  # Reconstruct tokenized string
            word_freqs[word] = word_freqs.get(word, 0) + 1

        vocab = self._get_initial_pairs(word_freqs)

        for i in tqdm(range(num_merges), desc="Training BPE"):
            if not vocab:
                break

            # Get most frequent pair
            best_pair = max(vocab, key=vocab.get)
            new_token = ''.join(best_pair)

            # Add new token to vocab
            if new_token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[new_token] = idx
                self.itos[idx] = new_token
                self.bpe_vocab.add(new_token)
                self.max_token_len = max(self.max_token_len, len(new_token))

            # Record the merge
            self.merge_rules[best_pair] = new_token

            # Update vocab
            vocab = self._update_vocab(best_pair, new_token, word_freqs)

    def _get_initial_pairs(self, word_freqs):
        """Get frequency counts of adjacent token pairs"""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = self._tokenize(word)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pairs[pair] += freq
        return pairs

    def _update_vocab(self, pair, new_token, word_freqs):
        """Update vocab after a merge"""
        new_pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = self._tokenize(word)
            i = 0
            new_tokens = []

            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            # Rebuild word and update frequencies
            new_word = ''.join(new_tokens)
            word_freqs[new_word] = word_freqs.get(new_word, 0) + freq

            # Count new pairs
            for j in range(len(new_tokens) - 1):
                new_pair = (new_tokens[j], new_tokens[j+1])
                new_pairs[new_pair] += freq

        return new_pairs

    # def encode(self, text, max_len=-100, add_special_tokens=True):
    #     """Tokenize and apply BPE merges"""
    #     tokens = self._tokenize(text)

    #     # Apply BPE merges greedily
    #     changed = True
    #     while changed:
    #         changed = False
    #         new_tokens = []
    #         i = 0
    #         while i < len(tokens):
    #             if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) in self.merge_rules:
    #                 new_tokens.append(self.merge_rules[(tokens[i], tokens[i+1])])
    #                 i += 2
    #                 changed = True
    #             else:
    #                 new_tokens.append(tokens[i])
    #                 i += 1
    #         tokens = new_tokens

    #     # Convert to indices
    #     if add_special_tokens:
    #         token_ids = [self.stoi['<sos>']] + [self.stoi[t] for t in tokens if t in self.stoi] + [self.stoi['<eos>']]
    #     else:
    #         token_ids = [self.stoi[t] for t in tokens if t in self.stoi]

    #     # Padding
    #     if max_len != -100:
    #         while len(token_ids) < max_len:
    #             token_ids.append(self.stoi['<pad>'])

    #     return token_ids

    def encode(self, text, max_len=-100, add_special_tokens=True):
        """Tokenize and apply BPE merges in a single pass (greedy left-to-right)"""
        tokens = self._tokenize(text)

        i = 0
        while i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merge_rules:
                # Replace the pair with the merged token
                tokens[i:i+2] = [self.merge_rules[pair]]
            else:
                i += 1

        # Convert to indices
        if add_special_tokens:
            token_ids = [self.stoi['<sos>']] + [self.stoi[t] for t in tokens if t in self.stoi] + [self.stoi['<eos>'], self.stoi['<eos>'], self.stoi['<eos>']]
        else:
            token_ids = [self.stoi[t] for t in tokens if t in self.stoi]

        # Padding
        if max_len != -100:
            token_ids = token_ids[:max_len]
            while len(token_ids) < max_len:
                token_ids.append(self.stoi['<pad>'])
        #print('encoding')
        return token_ids

    # def decode(self, token_ids, remove_padding=False):
    #     """Convert token ids back to string"""
    #     if isinstance(token_ids, torch.Tensor):
    #         token_ids = token_ids.tolist()

    #     tokens = [self.itos[t] for t in token_ids]
    #     if remove_padding and self.stoi['<pad>'] in tokens:
    #         tokens = tokens[:tokens.index(self.stoi['<pad>'])]

    #     return ''.join(tokens)

    def decode(self, token_ids, remove_padding=False, remove_special_tokens=False, return_clean_ids=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if not remove_padding:
            result = [self.itos[tok_id] for tok_id in token_ids]
        else:
            result = []
            clean_ids =[]
            for tok_id in token_ids:
                if tok_id == self.stoi['<eos>']:
                    result.append(self.itos[tok_id])
                    clean_ids.append(tok_id)
                    break
                result.append(self.itos[tok_id])
                clean_ids.append(tok_id)

        if remove_special_tokens:
            result = result[1:-1]
            clean_ids = clean_ids[1:-1]
            return ''.join(result), clean_ids

        return ''.join(result)
        
    def save(self, path):
        """Save tokenizer to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'stoi': self.stoi,
                'itos': self.itos,
                'merge_rules': self.merge_rules,
            }, f)

    @classmethod
    def load(cls, path):
        """Load tokenizer from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tokenizer = cls(vocab=data['stoi'])
        tokenizer.merge_rules = data['merge_rules']
        return tokenizer