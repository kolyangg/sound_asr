import re
from string import ascii_lowercase
from collections import defaultdict
import multiprocessing
from typing import List, Tuple, Optional
import torch
import kenlm
from transformers import AutoTokenizer
from pyctcdecode import build_ctcdecoder
import numpy as np
import os

from typing import Union

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CTCTextEncoder:
    def __init__(self, alphabet=None, arpa_path=None, binary_path=None, unigram_path=None,
                 use_bpe=False, pretrained_tokenizer="bert-base-uncased", 
                 lm_weight=0.5, beam_size=100, **kwargs):
        """Initialize encoder with explicit blank token handling"""
        # Define blank token explicitly
        self.BLANK_TOKEN = "" 
        self.use_bpe = use_bpe
        self.beam_size = beam_size
        self.lm_weight = lm_weight
        self.arpa_path = arpa_path
        self.binary_path = binary_path
        
        print('CTC Text Encoder:')
        print('use_bpe:', use_bpe)
        print('lm_weight:', lm_weight)
        print('beam_size:', beam_size)
        
        self.EMPTY_TOK = "" # ADDED
        
        # Load unigrams if provided
        self.unigrams = None
        if unigram_path and os.path.exists(unigram_path):
            print(f"Loading unigrams from: {unigram_path}")
            with open(unigram_path) as f:
                self.unigrams = [t.lower() for t in f.read().strip().split("\n")]
            print(f"Loaded {len(self.unigrams)} unigrams")

        # Initialize vocabulary with explicit blank handling
        if use_bpe:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
            # Use explicit blank token instead of pad token
            self.vocab = list(self.tokenizer.vocab.keys())
            # Ensure blank token is first in vocabulary
            if self.BLANK_TOKEN in self.vocab:
                self.vocab.remove(self.BLANK_TOKEN)
            self.vocab.insert(0, self.BLANK_TOKEN)
            print(f"Loaded BPE vocabulary of size: {len(self.vocab)}")
        else:
            if alphabet is None:
                if self.unigrams:
                    alphabet_set = set(char for word in self.unigrams for char in word)
                    alphabet_set.add(" ")
                    alphabet = sorted(list(alphabet_set))
                else:
                    alphabet = list(ascii_lowercase + " ")
            
            self.alphabet = alphabet
            # Insert blank token at beginning of vocabulary
            self.vocab = [self.BLANK_TOKEN] + list(self.alphabet)

        # Create index mappings
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.blank_index = self.char2ind[self.BLANK_TOKEN]  # Should be 0
        
        print(f"\nVocabulary Info:")
        print(f"Size: {len(self.vocab)}")
        print(f"Blank token index: {self.blank_index}")
        
        # Initialize language model components
        self._initialize_language_model()

    def ctc_decode(self, logits: torch.Tensor) -> str:
        """
        Improved CTC decoding with better blank and repeat handling
        """
        # Get the most likely token at each timestep
        predictions = torch.argmax(logits, dim=-1)
        
        decoded = []
        # Previous real token (excluding blanks)
        last_real_token = None
        # Previous token including blanks
        last_token = None
        
        for t in range(predictions.size(0)):
            token_idx = predictions[t].item()
            
            # Skip if current token is blank
            if token_idx == self.blank_index:
                continue
                
            # Skip if token is repeated and not a space
            current_char = self.ind2char[token_idx]
            if token_idx == last_token and current_char != " ":
                continue
                
            # Handle repeated real tokens (excluding blanks)
            if token_idx != self.blank_index:
                if token_idx != last_real_token:
                    decoded.append(current_char)
                    last_real_token = token_idx
            
            last_token = token_idx
        
        if self.use_bpe:
            text = "".join(decoded)
            return self.tokenizer.clean_up_tokenization(text)
        else:
            return "".join(decoded)

    def _initialize_language_model(self):
        """Initialize language model with explicit blank token handling"""
        self.lm = None
        self.decoder = None
        
        model_path = self.binary_path if self.binary_path else self.arpa_path
        if not model_path or not os.path.exists(model_path):
            return
            
        try:
            self.lm = kenlm.Model(model_path)
            print(f"Loaded {'binary' if self.binary_path else 'ARPA'} language model")
            
            
            
            
            # # Get vocabulary without blank token
            # labels = [c for c in self.vocab if c != self.BLANK_TOKEN]
            
            # Ensure blank token is at index 0
            labels = [self.BLANK_TOKEN] + [c for c in self.vocab if c != self.BLANK_TOKEN]
            
            # Ensure blank token is included at index 0
            assert self.BLANK_TOKEN == self.vocab[0], "Blank token must be at index 0"
            print(f"Decoder labels initialized with blank at index 0: {labels[:10]}")
            
            decoder_config = {
                "labels": labels,
                "kenlm_model_path": model_path,
                "alpha": self.lm_weight, # LM weight
                "beta": 0.1, # Word insertion penalty
                "unk_score_offset": -10.0,
                # "blank_token": self.BLANK_TOKEN,  # Explicitly set blank token
            }
            
            if self.unigrams:
                decoder_config["unigrams"] = self.unigrams
            
            self.decoder = build_ctcdecoder(**decoder_config)
            print("Successfully initialized language model and decoder")
            
        except Exception as e:
            print(f"Warning: Failed to initialize decoder: {str(e)}")
            self.decoder = None
            
    
    def encode(self, text) -> torch.Tensor:
        """Encode text with BPE or char-level encoding"""
        text = self.normalize_text(text)
        
        if self.use_bpe:
            try:
                # Use tokenizer's encode method with appropriate settings
                tokens = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                )["input_ids"]
                return torch.tensor(tokens).unsqueeze(0)
            except Exception as e:
                raise Exception(f"BPE encoding error: {str(e)}")
        else:
            # Char-level encoding remains the same
            try:
                return torch.tensor([self.char2ind[char] for char in text]).unsqueeze(0)
            except KeyError as e:
                unknown_chars = set([char for char in text if char not in self.char2ind])
                raise Exception(f"Unknown chars: '{' '.join(unknown_chars)}'")
    



    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text while preserving spaces"""
        # Convert to lowercase
        text = text.lower()
        # Replace multiple spaces with single space
        text = ' '.join(text.split())
        # Remove all characters except lowercase letters and spaces
        text = re.sub(r'[^a-z ]', '', text)
        return text


    def get_vocab_size(self):
        """Return vocabulary size for model configuration"""
        return len(self.vocab)
    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert isinstance(item, int)
        return self.ind2char[item]
    
    def decode(self, indices) -> str:
        """Decode indices to text"""
        if self.use_bpe:
            # Filter out padding tokens
            valid_indices = [
                int(idx) for idx in indices 
                if int(idx) != self.char2ind[self.EMPTY_TOK]
            ]
            try:
                return self.tokenizer.decode(valid_indices)
            except:
                return " ".join([self.ind2char[idx] for idx in valid_indices])
        else:
            return "".join([self.ind2char[int(ind)] for ind in indices if int(ind) != 0])

    def ctc_decode(self, inds) -> str:
        """CTC decoding with proper BPE handling"""
        decoded = []
        # last_char_ind = self.char2ind[self.EMPTY_TOK]
        
        # last_char_ind = self.EMPTY_TOK # old version
        last_char_ind = self.blank_index  # Use correct blank index (new version)
        
        for ind in inds:
            if last_char_ind == ind:
                continue
            # if ind != self.EMPTY_TOK: # old version
            if ind != self.blank_index: # new version
                decoded.append(self.ind2char[ind])
            last_char_ind = ind
            
        if self.use_bpe:
            text = " ".join(decoded)
            return self.tokenizer.clean_up_tokenization(text)
        else:
            return "".join(decoded)

    
    def ctc_beam_search(self, probs, beam_size: int = 100,
                       use_lm: bool = False, debug: bool = False) -> List[Tuple[str, float]]:
        """
        Beam search with enhanced BPE support
        """
        beam_size = self.beam_size
        debug = False
        # debug = True
        
        if use_lm and self.decoder is not None:
            try:
                # Convert to numpy if needed
                if isinstance(probs, torch.Tensor):
                    probs = probs.cpu().numpy()
                
                # Use pyctcdecode's beam search
                beams = self.decoder.decode_beams(
                    probs,
                    beam_prune_logp=-10.0,
                    token_min_logp=-5.0,
                    hotwords=[],
                    hotword_weight=10.0,
                )
                
                # Format beams and clean up BPE tokens
                formatted_beams = []
                for beam in beams[:beam_size]:
                    text = beam[0]  # Get the text
                    acoustic_score = beam[3]
                    lm_score = beam[4]
                    
                    # Clean up BPE tokens if using BPE
                    if self.use_bpe:
                        # Remove special tokens and clean up
                        text = self.tokenizer.clean_up_tokenization(text)
                    
                    # Combine scores with weighting
                    combined_score = (1 - self.lm_weight) * acoustic_score + self.lm_weight * lm_score
                    
                    # Apply length normalization
                    text_len = max(1, len(text.split()))
                    normalized_score = combined_score / text_len
                    
                    formatted_beams.append((text, normalized_score))
                
                if debug:
                    print("\nFormatted beam results with BPE:")
                    for text, score in formatted_beams[:3]:
                        print(f"Text: '{text}', Score: {score:.4f}")
                
                if formatted_beams:
                    return sorted(formatted_beams, key=lambda x: -x[1])
                else:
                    print("No valid beams found, falling back to standard beam search")
                    return self._standard_beam_search(probs, beam_size, debug)
                    
            except Exception as e:
                print(f"Beam search with LM failed: {str(e)}, falling back to standard beam search")
                return self._standard_beam_search(probs, beam_size, debug)
        else:
            return self._standard_beam_search(probs, beam_size, debug)
    
    
    def _standard_beam_search(self, probs, beam_size: int = 10, debug: bool = False) -> List[Tuple[str, float]]:
        """Original beam search implementation with improved debugging"""
        
        beam_size = self.beam_size
        
        # Convert input to torch tensor if needed
        if isinstance(probs, np.ndarray):
            probs = torch.from_numpy(probs)

        # Ensure probs is on CPU
        if probs.device != torch.device('cpu'):
            probs = probs.cpu()

        # Initialize beam with empty string
        dp = {("", self.EMPTY_TOK): 0.0}  # Using log probs

        # Convert to log probabilities
        log_probs = torch.log(probs + 1e-8)
        
        if debug:
            print("\nStarting beam search with beam size:", beam_size)
        
        for t, prob in enumerate(log_probs):
            new_dp = defaultdict(lambda: float('-inf'))
            
            # Get top-k tokens for this timestep
            top_k = torch.topk(prob, k=min(beam_size, len(prob)))
            
            if debug and t < 2:  # Print first two timesteps
                print(f"\nTimestep {t}:")
                print("Top tokens:", [(self.ind2char[idx.item()], val.item()) 
                                    for val, idx in zip(top_k.values, top_k.indices)])
            
            # Only expand using top-k tokens
            for val, ind in zip(top_k.values, top_k.indices):
                curr_char = self.ind2char[ind.item()]
                next_token_log_prob = val.item()
                
                for (prefix, last_char), log_prob in dp.items():
                    # Skip repeated characters (except spaces)
                    if last_char == curr_char and curr_char != " ":
                        new_prefix = prefix
                    else:
                        if curr_char != self.EMPTY_TOK:
                            # Handle spaces better
                            if curr_char == " " and prefix.endswith(" "):
                                continue
                            new_prefix = prefix + curr_char
                        else:
                            new_prefix = prefix
                    
                    # Update score
                    new_log_prob = log_prob + next_token_log_prob
                    key = (new_prefix, curr_char)
                    new_dp[key] = max(new_dp[key], new_log_prob)
            
            # Normalize scores
            if len(new_dp) > 0:
                max_score = max(score for _, score in new_dp.items())
                new_dp = {key: score - max_score for key, score in new_dp.items()}
            
            # Truncate beams
            dp = dict(sorted(new_dp.items(), key=lambda x: -x[1])[:beam_size])
            
            if debug and t < 2:  # Print beam state for first two timesteps
                print("\nCurrent beam:")
                for (text, last_char), score in list(dp.items())[:3]:
                    print(f"Text: '{text}', Last: '{last_char}', Score: {score:.4f}")
        
        # Format final results
        final_beams = []
        for (text, _), score in dp.items():
            # Clean up text
            if self.use_bpe:
                text = self.tokenizer.clean_up_tokenization(text)
            else:
                text = ' '.join(text.split())
                
            if not text.strip():  # Skip empty results
                continue
                
            # Length normalization
            text_len = max(1, len(text.split()))
            normalized_score = score / text_len
            
            final_beams.append((text, normalized_score))
            
        # Sort and ensure we have results
        final_beams.sort(key=lambda x: -x[1])
        if not final_beams:
            final_beams = [("", float('-inf'))]
        
        return final_beams[:beam_size]
        
    def test_language_model(self):
        """Debug function to verify LM functionality"""
        print("\nTesting Language Model...")
        
        # Check if LM is loaded
        if self.lm is None:
            print("Error: Language model is not loaded!")
            return
        
        # Test sentences with expected relative scores
        test_sentences = [
            "this is a good sentence",  # Should get good score
            "this is also a good sentence",  # Should be similar
            "thiss iss nott aa goodd sentencee",  # Should get worse score
            "random word salad box cat",  # Should get bad score
            "the cat sat on the mat",  # Should get good score
            "",  # Edge case
            "a",  # Edge case
        ]
        
        print("\nTesting individual sentences:")
        for sentence in test_sentences:
            score = self.score_with_lm(sentence)
            print(f"\nText: '{sentence}'")
            print(f"LM Score: {score:.4f}")
        
        # Test word completions
        test_prefixes = [
            "the quick brown",  # should favor "fox"
            "how are",  # should favor "you"
            "thank",  # should favor "you"
            "nice to",  # should favor "meet"
        ]
        
        print("\nTesting word completions:")
        for prefix in test_prefixes:
            print(f"\nPrefix: '{prefix}'")
            completions = [
                prefix + " " + word for word in ["you", "fox", "cat", "xyz", "meet"]
            ]
            scores = [(completion, self.score_with_lm(completion)) 
                    for completion in completions]
            scores.sort(key=lambda x: x[1], reverse=True)
            print("Top completions by score:")
            for completion, score in scores[:3]:
                print(f"  '{completion}': {score:.4f}")
    
    def score_with_lm(self, text: str) -> float:
        """
        Improved LM scoring using proper n-gram context
        """
        if self.lm is None:
            return 0.0
        
        if not text or len(text.strip()) == 0:
            return float('-inf')
        
        text = text.lower().strip()
        return self.lm.score(text, bos=True, eos=True)  # Add beginning/end sentence tokens


    def _basic_ctc_decode(self, logits: np.ndarray, sequence_length: int) -> List[str]:
        """Basic CTC decoding without LM"""
        # Get argmax indices across vocabulary dimension (last dim)
        argmax_indices = np.argmax(logits, axis=-1)
        
        # For 1D input, handle single index
        if len(argmax_indices.shape) == 0:
            argmax_indices = np.array([argmax_indices])
            
        # Convert single sequence to batch if needed
        if len(argmax_indices.shape) == 1:
            argmax_indices = np.expand_dims(argmax_indices, axis=0)
            
        # Decode each sequence
        predictions = []
        for sequence in argmax_indices:
            decoded = []
            last_idx = None
            
            # Use only up to sequence_length
            for idx in sequence[:sequence_length]:
                if idx != 0 and idx != last_idx:  # Skip blanks and repeats
                    decoded.append(self.ind2char[idx])
                last_idx = idx
                
            text = "".join(decoded)
            if self.use_bpe:
                text = self.tokenizer.clean_up_tokenization(text)
            predictions.append(text)
            
        return predictions
    
    

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize input text"""
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def test_decoder(self, sample_text: str = "test decoder functionality"):
        """Test the decoder setup"""
        print("\nTesting decoder configuration...")
        
        # Test basic encoding/decoding
        encoded = self.encode(sample_text)
        decoded = self.decode(encoded[0])
        print(f"Original text: {sample_text}")
        print(f"Basic decode: {decoded}")
        
        # Test with random logits
        sequence_length = 50
        vocab_size = len(self)
        fake_logits = torch.randn(1, sequence_length, vocab_size)
        fake_length = torch.tensor([sequence_length])
        
        # Test pyctcdecode
        if self.decoder is not None:
            print("\nTesting pyctcdecode integration...")
            decoded_with_lm = self.ctc_decode(fake_logits, fake_length)
            print(f"Decoded with LM: {decoded_with_lm[0]}")
            
            # Test beam search parameters
            print(f"\nBeam width: {self.beam_width}")
            print(f"LM weight: {self.lm_weight}")
        else:
            print("\nNo language model loaded - using basic CTC decoding")
            basic_decoded = self._basic_ctc_decode(fake_logits, fake_length)
            print(f"Basic CTC decoded: {basic_decoded[0]}")
            
    def test_language_model(self):
        """Debug function to verify LM functionality"""
        print("\nTesting Language Model...")
        
        # Check if LM is loaded
        if self.lm is None:
            print("Error: Language model is not loaded!")
            return
        
        # Test sentences with expected relative scores
        test_sentences = [
            "this is a good sentence",  # Should get good score
            "this is also a good sentence",  # Should be similar
            "thiss iss nott aa goodd sentencee",  # Should get worse score
            "random word salad box cat",  # Should get bad score
            "the cat sat on the mat",  # Should get good score
            "",  # Edge case
            "a",  # Edge case
        ]
        
        print("\nTesting individual sentences:")
        for sentence in test_sentences:
            score = self.score_with_lm(sentence)
            print(f"\nText: '{sentence}'")
            print(f"LM Score: {score:.4f}")
        
        # Test word completions
        test_prefixes = [
            "the quick brown",  # should favor "fox"
            "how are",  # should favor "you"
            "thank",  # should favor "you"
            "nice to",  # should favor "meet"
        ]
        
        print("\nTesting word completions:")
        for prefix in test_prefixes:
            print(f"\nPrefix: '{prefix}'")
            completions = [
                prefix + " " + word for word in ["you", "fox", "cat", "xyz", "meet"]
            ]
            scores = [(completion, self.score_with_lm(completion)) 
                    for completion in completions]
            scores.sort(key=lambda x: x[1], reverse=True)
            print("Top completions by score:")
            for completion, score in scores[:3]:
                print(f"  '{completion}': {score:.4f}")

    def test_kenlm_directly(self):
        """Test direct KenLM scoring without any normalization"""
        if self.lm is None:
            print("Error: Language model is not loaded!")
            return
            
        print("\nDirect KenLM Scoring Test...")
        print(f"LM Info:")
        print(f"Order: {self.lm.order}")
        
        test_phrases = [
            "this is a good sentence",
            "thiss iss nott aa goodd sentencee",
            "the quick brown fox",
            "the quick brown cat",
            "how are you",
            "how are fox",
            "random word salad box cat",
            "the cat sat on the mat"
        ]
        
        for phrase in test_phrases:
            # Get raw full score
            full_score = self.lm.score(phrase)
            
            # Get individual word scores
            state = kenlm.State()
            self.lm.BeginSentenceWrite(state)
            word_scores = []
            
            for word in phrase.split():
                out_state = kenlm.State()
                score = self.lm.BaseScore(state, word, out_state)
                word_scores.append((word, score))
                state = out_state
                
            print(f"\nPhrase: '{phrase}'")
            print(f"Full score: {full_score:.4f}")
            print("Word scores:")
            for word, score in word_scores:
                print(f"  {word}: {score:.4f}")
            
            # Also print total and average scores
            total_score = sum(score for _, score in word_scores)
            avg_score = total_score / len(word_scores)
            print(f"Total score: {total_score:.4f}")
            print(f"Average score: {avg_score:.4f}")

    def score_with_lm(self, text: str) -> float:
        """
        Score text using language model, handling edge cases
        """
        if self.lm is None:
            return 0.0
        
        if not text or len(text.strip()) == 0:
            return float('-inf')
        
        text = text.lower().strip()
        return self.lm.score(text, bos=True, eos=True)