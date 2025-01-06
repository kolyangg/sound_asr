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
    # def __init__(self, alphabet=None, arpa_path=None, binary_path=None, unigram_path=None,
    #              use_bpe=True, pretrained_tokenizer="bert-base-uncased", 
    #              lm_weight=0.5, beam_width=100, **kwargs):
    #     """
    #     Enhanced CTCTextEncoder with unigram support
        
    #     Args:
    #         alphabet: Base character set (if not using BPE)
    #         arpa_path: Path to ARPA language model file
    #         binary_path: Path to binary language model file
    #         unigram_path: Path to unigram vocabulary file (e.g., librispeech-vocab.txt)
    #         use_bpe: Whether to use BPE tokenization
    #         pretrained_tokenizer: HuggingFace tokenizer to use if use_bpe=True
    #         lm_weight: Weight for language model scoring
    #         beam_width: Beam width for beam search decoding
    #     """
    #     self.EMPTY_TOK = ""
    #     self.use_bpe = use_bpe
    #     self.beam_width = beam_width
    #     self.lm_weight = lm_weight
    #     self.arpa_path = arpa_path
    #     self.binary_path = binary_path
        
    #     # Load unigrams if provided
    #     self.unigram_list = None
    #     if unigram_path and os.path.exists(unigram_path):
    #         try:
    #             with open(unigram_path) as f:
    #                 self.unigram_list = [t.lower() for t in f.read().strip().split("\n")]
    #             print(f"Loaded {len(self.unigram_list)} unigrams from {unigram_path}")
    #         except Exception as e:
    #             print(f"Failed to load unigrams from {unigram_path}: {str(e)}")
        
    #     # Initialize vocabulary based on configuration
    #     if use_bpe:
    #         # BPE tokenization
    #         self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    #         self.vocab = [self.EMPTY_TOK] + list(self.tokenizer.vocab.keys())
    #     else:
    #         # Character-level with potential unigram enhancement
    #         if alphabet is None:
    #             if self.unigram_list:
    #                 # Build alphabet from unigrams
    #                 alphabet_set = set()
    #                 for word in self.unigram_list:
    #                     alphabet_set.update(word)
    #                 alphabet = sorted(list(alphabet_set))
    #                 print(f"Built alphabet from unigrams: {alphabet}")
    #             else:
    #                 # Default to ascii lowercase + space
    #                 alphabet = list(ascii_lowercase + " ")
    #         self.alphabet = alphabet
    #         self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

    #     # Create index mappings
    #     self.ind2char = dict(enumerate(self.vocab))
    #     self.char2ind = {v: k for k, v in self.ind2char.items()}
        
    #     # Print vocabulary info
    #     print(f"\nVocabulary Info:")
    #     print(f"Size: {len(self.vocab)}")
    #     print(f"First few items: {self.vocab[:10]}")
    #     print(f"Last few items: {self.vocab[-10:]}")
        
    #     # Initialize language model
    #     self.lm = None
    #     self.decoder = None
        
    #     # Try binary model first, then ARPA
    #     model_path = None
    #     if self.binary_path and os.path.exists(self.binary_path):
    #         try:
    #             self.lm = kenlm.Model(self.binary_path)
    #             model_path = self.binary_path
    #             print("\nLoaded binary language model")
    #         except Exception as e:
    #             print(f"Failed to load binary model: {str(e)}")
        
    #     if self.lm is None and self.arpa_path and os.path.exists(self.arpa_path):
    #         try:
    #             self.lm = kenlm.Model(self.arpa_path)
    #             model_path = self.arpa_path
    #             print("Loaded ARPA language model")
    #         except Exception as e:
    #             print(f"Failed to load ARPA model: {str(e)}")
        
    #     # Initialize pyctcdecode decoder
    #     if model_path:
    #         try:
    #             labels = [c for c in self.vocab if c != self.EMPTY_TOK]
    #             self.decoder = build_ctcdecoder(
    #                 labels,
    #                 kenlm_model_path=model_path,
    #                 unigrams=self.unigram_list,  # Add unigrams to decoder
    #                 alpha=lm_weight,  # LM weight
    #                 beta=0.1,  # word insertion bonus
    #                 unk_score_offset=-10.0,  # Penalty for unknown tokens
    #             )
    #             print("Successfully initialized pyctcdecode decoder")
    #         except Exception as e:
    #             print(f"Warning: Failed to initialize decoder: {str(e)}")
    #             self.decoder = None
    
    # def __init__(self, alphabet=None, arpa_path=None, binary_path=None, unigram_path=None,
    #              use_bpe=True, pretrained_tokenizer="bert-base-uncased", 
    #              lm_weight=0.5, beam_width=100, **kwargs):
    #     """Initialize encoder with proper space handling"""
    #     self.EMPTY_TOK = ""
    #     self.use_bpe = use_bpe
    #     self.beam_width = beam_width
    #     self.lm_weight = lm_weight
    #     self.arpa_path = arpa_path
    #     self.binary_path = binary_path
        
    #     # Load unigrams if provided
    #     self.unigram_list = None
    #     if unigram_path and os.path.exists(unigram_path):
    #         try:
    #             with open(unigram_path) as f:
    #                 self.unigram_list = [t.lower() for t in f.read().strip().split("\n")]
    #             print(f"Loaded {len(self.unigram_list)} unigrams from {unigram_path}")
    #         except Exception as e:
    #             print(f"Failed to load unigrams from {unigram_path}: {str(e)}")
        
    #     # Initialize vocabulary
    #     if use_bpe:
    #         # BPE tokenization
    #         self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    #         self.vocab = [self.EMPTY_TOK] + list(self.tokenizer.vocab.keys())
    #     else:
    #         # Character-level with potential unigram enhancement
    #         if alphabet is None:
    #             if self.unigram_list:
    #                 # Build alphabet from unigrams
    #                 alphabet_set = set()
    #                 for word in self.unigram_list:
    #                     alphabet_set.update(word)
    #                 # Ensure space is in alphabet
    #                 alphabet_set.add(" ")
    #                 alphabet = sorted(list(alphabet_set))
    #                 print(f"Built alphabet from unigrams: {alphabet}")
    #             else:
    #                 # Default to ascii lowercase + space
    #                 alphabet = list(ascii_lowercase + " ")
            
    #         # Ensure space is in alphabet if not already
    #         if " " not in alphabet:
    #             alphabet = list(alphabet) + [" "]
            
    #         self.alphabet = alphabet
    #         self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

    #     # Create index mappings
    #     self.ind2char = dict(enumerate(self.vocab))
    #     self.char2ind = {v: k for k, v in self.ind2char.items()}
        
    #     # Print vocabulary info
    #     print(f"\nVocabulary Info:")
    #     print(f"Size: {len(self.vocab)}")
    #     print(f"First few items: {self.vocab[:10]}")
    #     if not self.use_bpe:
    #         print(f"Alphabet: {self.alphabet}")
    #         print(f"Space character index: {self.char2ind.get(' ', 'Not found')}")
        
    #     # Initialize language model components
    #     self._initialize_language_model()
    
    # def encode(self, text) -> torch.Tensor:
    #     """Encode text with proper error handling"""
    #     text = self.normalize_text(text)
        
    #     if self.use_bpe:
    #         try:
    #             tokens = self.tokenizer.encode(text, add_special_tokens=False)
    #             return torch.tensor(tokens).unsqueeze(0)
    #         except Exception as e:
    #             raise Exception(f"BPE encoding error: {str(e)}")
    #     else:
    #         try:
    #             indices = []
    #             for char in text:
    #                 if char not in self.char2ind:
    #                     print(f"Warning: Character '{char}' not in vocabulary, skipping")
    #                     continue
    #                 indices.append(self.char2ind[char])
    #             if not indices:
    #                 raise Exception(f"No valid characters found in text: '{text}'")
    #             return torch.tensor(indices).unsqueeze(0)
    #         except Exception as e:
    #             print(f"Encoding failed for text: '{text}'")
    #             print(f"Vocabulary: {self.alphabet}")
    #             print(f"char2ind: {self.char2ind}")
    #             raise e

    # WORKING VERSION!!!
    # def __init__(self, alphabet=None, arpa_path=None, binary_path=None, unigram_path=None,
    #              use_bpe=True, pretrained_tokenizer="bert-base-uncased", 
    #              lm_weight=0.5, beam_width=100, **kwargs):
    #     """Initialize encoder with corrected BPE handling"""
    #     self.EMPTY_TOK = ""
    #     self.use_bpe = use_bpe
    #     self.beam_width = beam_width
    #     self.lm_weight = lm_weight
    #     self.arpa_path = arpa_path
    #     self.binary_path = binary_path
        
    #     # Initialize vocabulary
    #     if use_bpe:
    #         # BPE tokenization
    #         self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    #         # Don't add EMPTY_TOK for BPE - use tokenizer's pad token instead
    #         self.EMPTY_TOK = self.tokenizer.pad_token
    #         self.vocab = list(self.tokenizer.vocab.keys())  # No need to add EMPTY_TOK
    #         print(f"Loaded BPE vocabulary of size: {len(self.vocab)}")
    #     else:
    #         # Character-level setup remains the same
    #         if alphabet is None:
    #             if unigram_path and os.path.exists(unigram_path):
    #                 # Build alphabet from unigrams
    #                 print(f"Loading unigrams from: {unigram_path}")
    #                 alphabet_set = set()
    #                 with open(unigram_path) as f:
    #                     unigrams = [t.lower() for t in f.read().strip().split("\n")]
    #                 for word in unigrams:
    #                     alphabet_set.update(word)
    #                 alphabet_set.add(" ")
    #                 alphabet = sorted(list(alphabet_set))
    #             else:
    #                 alphabet = list(ascii_lowercase + " ")
            
    #         self.alphabet = alphabet
    #         self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

    #     # Create index mappings
    #     self.ind2char = dict(enumerate(self.vocab))
    #     self.char2ind = {v: k for k, v in self.ind2char.items()}
        
    #     print(f"\nVocabulary Info:")
    #     print(f"Size: {len(self.vocab)}")
    #     print(f"EMPTY_TOK index: {self.char2ind[self.EMPTY_TOK]}")
    #     if not use_bpe:
    #         print(f"First few chars: {self.vocab[:10]}")
        
    #     # Initialize language model components
    #     self._initialize_language_model()
    
    
    # ADDING SMTH FOR UNIGRAMS!!!
    
    def __init__(self, alphabet=None, arpa_path=None, binary_path=None, unigram_path=None,
                 use_bpe=True, pretrained_tokenizer="bert-base-uncased", 
                 lm_weight=0.5, beam_width=100, **kwargs):
        """Initialize encoder with corrected BPE handling"""
        self.EMPTY_TOK = ""
        self.use_bpe = use_bpe
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.arpa_path = arpa_path
        self.binary_path = binary_path
        
               
        # Load unigrams if provided
        self.unigrams = None
        if unigram_path and os.path.exists(unigram_path):
            print(f"Loading unigrams from: {unigram_path}")
            with open(unigram_path) as f:
                # self.unigrams = [t for t in f.read().strip().split("\n")]
                self.unigrams = [t.lower() for t in f.read().strip().split("\n")]
            print(f"Loaded {len(self.unigrams)} unigrams")

        # # Initialize vocabulary
        # if use_bpe:
        #     # BPE tokenization
        #     self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
        #     self.EMPTY_TOK = self.tokenizer.pad_token
        #     self.vocab = list(self.tokenizer.vocab.keys())
        #     print(f"Loaded BPE vocabulary of size: {len(self.vocab)}")
        # else:
        #     # Character-level setup
        #     if alphabet is None:
        #         if self.unigrams:
        #             # Build alphabet from unigrams
        #             alphabet_set = set()
        #             for word in self.unigrams:
        #                 alphabet_set.update(word)
        #             alphabet_set.add(" ")
        #             # alphabet_set.add(ascii_lowercase)
        #             alphabet = sorted(list(alphabet_set))
        #         else:
        #             alphabet = list(ascii_lowercase + " ")
            
        #     self.alphabet = alphabet
        #     self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

         ### NEW VERSION TO FIX SPACES ERRROR!!! ###
        # Initialize vocabulary
        if use_bpe:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
            self.EMPTY_TOK = self.tokenizer.pad_token or "[PAD]"
            self.vocab = list(self.tokenizer.vocab.keys())

            # Ensure the blank token is explicitly added if not in vocab
            if self.EMPTY_TOK not in self.vocab:
                self.vocab.append(self.EMPTY_TOK)
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
            self.vocab = list(self.alphabet)

        # Ensure EMPTY_TOK is explicitly added to the end
        if self.EMPTY_TOK not in self.vocab:
            self.vocab.append(self.EMPTY_TOK)
            print(f"Explicitly added EMPTY_TOK '{self.EMPTY_TOK}' at the end of vocab.")
                ### NEW VERSION TO FIX SPACES ERRROR!!! ###
                
        # Create index mappings
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.blank_index = self.char2ind[self.EMPTY_TOK]
        print(f"Final blank token index: {self.blank_index}")
        
        print(f"\nVocabulary Info:")
        print(f"Size: {len(self.vocab)}")
        print(f"EMPTY_TOK index: {self.char2ind[self.EMPTY_TOK]}")
        if not use_bpe:
            print(f"First few chars: {self.vocab[:10]}")
        
        # Initialize language model components
        self._initialize_language_model()
            
    
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

    # def _initialize_language_model(self):
    #     """Initialize language model and decoder"""
    #     self.lm = None
    #     self.decoder = None
        
    #     # Try binary model first, then ARPA
    #     model_path = None
    #     if self.binary_path and os.path.exists(self.binary_path):
    #         try:
    #             self.lm = kenlm.Model(self.binary_path)
    #             model_path = self.binary_path
    #             print("\nLoaded binary language model")
    #         except Exception as e:
    #             print(f"Failed to load binary model: {str(e)}")
        
    #     if self.lm is None and self.arpa_path and os.path.exists(self.arpa_path):
    #         try:
    #             self.lm = kenlm.Model(self.arpa_path)
    #             model_path = self.arpa_path
    #             print("Loaded ARPA language model")
    #         except Exception as e:
    #             print(f"Failed to load ARPA model: {str(e)}")
        
    #     if model_path:
    #         try:
    #             labels = [c for c in self.vocab if c != self.EMPTY_TOK]
    #             self.decoder = build_ctcdecoder(
    #                 labels,
    #                 kenlm_model_path=model_path,
    #                 unigrams=self.unigram_list,
    #                 alpha=self.lm_weight,
    #                 beta=0.1,
    #             )
    #             print("Successfully initialized pyctcdecode decoder")
    #         except Exception as e:
    #             print(f"Warning: Failed to initialize decoder: {str(e)}")

    def get_vocab_size(self):
        """Return vocabulary size for model configuration"""
        return len(self.vocab)
    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert isinstance(item, int)
        return self.ind2char[item]

    # def encode(self, text) -> torch.Tensor:
    #     """Encode text using either BPE or char-level encoding"""
    #     text = self.normalize_text(text)
        
    #     if self.use_bpe:
    #         tokens = self.tokenizer.encode(text, add_special_tokens=False)
    #         return torch.tensor(tokens).unsqueeze(0)
    #     else:
    #         return torch.tensor([self.char2ind[char] for char in text]).unsqueeze(0)

    # def decode(self, indices) -> str:
    #     """Basic decoding without CTC/LM"""
    #     if self.use_bpe:
    #         valid_indices = [int(idx) for idx in indices if int(idx) != 0]
    #         try:
    #             return self.tokenizer.decode(valid_indices)
    #         except:
    #             return " ".join([self.ind2char[idx] for idx in valid_indices])
    #     else:
    #         return "".join([self.ind2char[int(ind)] for ind in indices if int(ind) != 0])

    
    # def ctc_decode(self, logits: Union[torch.Tensor, np.ndarray], 
    #                logits_length: Optional[Union[torch.Tensor, np.ndarray]] = None) -> List[str]:
    #     """
    #     Decode using pyctcdecode with support for 1D input
        
    #     Args:
    #         logits: Log probabilities with shape:
    #                - (vocab_size,) for single timestep
    #                - (sequence_length, vocab_size) for single sequence
    #                - (batch_size, sequence_length, vocab_size) for batch
    #         logits_length: Optional sequence length
    #     """
    #     # Convert torch tensor to numpy if needed
    #     if isinstance(logits, torch.Tensor):
    #         logits_np = logits.cpu().numpy()
    #     else:
    #         logits_np = logits
            
    #     # Handle 1D input (single timestep)
    #     if len(logits_np.shape) == 1:
    #         # Reshape to (1, 1, vocab_size)
    #         logits_np = np.expand_dims(np.expand_dims(logits_np, axis=0), axis=0)
    #         sequence_length = 1
            
    #     # Handle 2D input (single sequence)
    #     elif len(logits_np.shape) == 2:
    #         # Reshape to (1, sequence_length, vocab_size)
    #         logits_np = np.expand_dims(logits_np, axis=0)
    #         sequence_length = logits_np.shape[1]
            
    #     # 3D input (batch)
    #     else:
    #         sequence_length = logits_np.shape[1]
            
    #     # Get actual sequence length if provided
    #     if logits_length is not None:
    #         if isinstance(logits_length, torch.Tensor):
    #             sequence_length = logits_length.cpu().item()
    #         else:
    #             sequence_length = logits_length
                
    #     # Use pyctcdecode if available
    #     if self.decoder is not None:
    #         try:
    #             with multiprocessing.get_context("fork").Pool() as pool:
    #                 # Get the actual sequence to decode
    #                 logits_seq = logits_np[0, :sequence_length]
    #                 decoded_batch = self.decoder.decode_batch(
    #                     pool,
    #                     [logits_seq],
    #                     beam_width=self.beam_width
    #                 )
    #             return decoded_batch
    #         except Exception as e:
    #             print(f"Beam search failed: {str(e)}, falling back to basic decoding")
    #             return self._basic_ctc_decode(logits_np, sequence_length)
        
    #     # Fallback to basic decoding
    #     return self._basic_ctc_decode(logits_np, sequence_length)

    # def ctc_decode(self, inds) -> str:
    #     """
    #     CTC decoding - works with both BPE and char-level
    #     Maintains original successful implementation
    #     """
    #     decoded = []
    #     last_char_ind = self.EMPTY_TOK
        
    #     for ind in inds:
    #         if last_char_ind == ind:
    #             continue
    #         if ind != self.EMPTY_TOK:
    #             decoded.append(self.ind2char[ind])
    #         last_char_ind = ind
            
    #     if self.use_bpe:
    #         # Join BPE tokens and post-process
    #         text = " ".join(decoded)
    #         return self.tokenizer.clean_up_tokenization(text)
    #     else:
    #         # Original char-level joining
    #         return "".join(decoded)
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

    
    # WORKING VERSION!!!
    # def _initialize_language_model(self):
    #     """Initialize language model and decoder with correct vocabulary"""
    #     self.lm = None
    #     self.decoder = None
        
    #     model_path = self.binary_path if self.binary_path else self.arpa_path
    #     if not model_path or not os.path.exists(model_path):
    #         return
            
    #     try:
    #         self.lm = kenlm.Model(model_path)
    #         # For pyctcdecode, use actual vocabulary without padding token
    #         labels = [token for token in self.vocab if token != self.EMPTY_TOK]
    #         self.decoder = build_ctcdecoder(
    #             labels,
    #             kenlm_model_path=model_path,
    #             alpha=self.lm_weight,
    #             beta=0.1,
    #         )
    #         print("Successfully initialized language model and decoder")
    #     except Exception as e:
    #         print(f"Failed to initialize LM/decoder: {str(e)}")
            
            
    # TRY ADDING UNIGRAMS!!!
    def _initialize_language_model(self):
        """Initialize language model and decoder with proper unigram handling"""
        self.lm = None
        self.decoder = None
        
        model_path = self.binary_path if self.binary_path else self.arpa_path
        if not model_path or not os.path.exists(model_path):
            return
            
        try:
            self.lm = kenlm.Model(model_path)
            print(f"Loaded {'binary' if self.binary_path else 'ARPA'} language model")
            
            # Get vocabulary without EMPTY_TOK
            labels = [c for c in self.vocab if c != self.EMPTY_TOK]
            
            # For character-level, we use a lower alpha (LM weight) since we're using char-level probabilities
            if not self.use_bpe:
                self.lm_weight = min(self.lm_weight, 0.3)  # Reduce LM influence for char-level
            
            decoder_config = {
                "labels": labels,
                "kenlm_model_path": model_path,
                "alpha": self.lm_weight,  # LM weight
                "beta": 0.1,  # word insertion bonus
                "unk_score_offset": -10.0,  # Penalty for unknown tokens
            }
            
            # Only add unigrams if we have them and they're appropriate for our model type
            if hasattr(self, 'unigrams') and self.unigrams:
                decoder_config["unigrams"] = self.unigrams
            
            self.decoder = build_ctcdecoder(**decoder_config)
            
            print("Successfully initialized language model and decoder")
            if hasattr(self, 'unigrams'):
                print(f"Decoder initialized with {len(self.unigrams)} unigrams")
                
        except Exception as e:
            print(f"Warning: Failed to initialize decoder: {str(e)}")
            self.decoder = None
    
    # def ctc_beam_search(self, probs, beam_size: int = 10,
    #                    use_lm: bool = False, debug: bool = False) -> List[Tuple[str, float]]:
    #     """
    #     Beam search with optional LM integration using pyctcdecode
    #     """
    #     if use_lm and self.decoder is not None:
    #         try:
    #             # Convert to numpy if needed
    #             if isinstance(probs, torch.Tensor):
    #                 probs = probs.cpu().numpy()
                
    #             # Use pyctcdecode's beam search
    #             beams = self.decoder.decode_beams(
    #                 probs,
    #                 beam_prune_logp=-10.0,
    #                 token_min_logp=-5.0,
    #                 hotwords=[],
    #                 hotword_weight=10.0,
    #             )
                
    #             # Format beams based on actual structure
    #             # Beam format is (text, kenlm_state, token_history, acoustic_score, language_model_score)
    #             formatted_beams = []
    #             for beam in beams[:beam_size]:
    #                 text = beam[0]  # Get the text
    #                 acoustic_score = beam[3]  # Get acoustic score
    #                 lm_score = beam[4]  # Get language model score
                    
    #                 # Combine scores with weighting
    #                 combined_score = (1 - self.lm_weight) * acoustic_score + self.lm_weight * lm_score
                    
    #                 # Apply length normalization
    #                 text_len = max(1, len(text.split()))
    #                 normalized_score = combined_score / text_len
                    
    #                 formatted_beams.append((text, normalized_score))
                
    #             if debug:
    #                 print("\nFormatted beam results:")
    #                 for text, score in formatted_beams[:3]:
    #                     print(f"Text: '{text}', Score: {score:.4f}")
                    
    #                 # Also print detailed scores for first beam
    #                 if formatted_beams:
    #                     first_beam = beams[0]
    #                     print("\nDetailed scores for top beam:")
    #                     print(f"Text: '{first_beam[0]}'")
    #                     print(f"Acoustic score: {first_beam[3]:.4f}")
    #                     print(f"LM score: {first_beam[4]:.4f}")
                
    #             if formatted_beams:
    #                 return sorted(formatted_beams, key=lambda x: -x[1])  # Sort by score descending
    #             else:
    #                 print("No valid beams found, falling back to standard beam search")
    #                 return self._standard_beam_search(probs, beam_size, debug)
                    
    #         except Exception as e:
    #             print(f"Beam search with LM failed: {str(e)}, falling back to standard beam search")
    #             return self._standard_beam_search(probs, beam_size, debug)
    #     else:
    #         return self._standard_beam_search(probs, beam_size, debug)

    def ctc_beam_search(self, probs, beam_size: int = 10,
                       use_lm: bool = False, debug: bool = False) -> List[Tuple[str, float]]:
        """
        Beam search with enhanced BPE support
        """
        debug = False
        
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