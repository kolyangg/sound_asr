import re
from string import ascii_lowercase
from collections import defaultdict
import torch
import kenlm
from transformers import AutoTokenizer
from pyctcdecode import build_ctcdecoder
import numpy as np
import os

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


from typing import List, Tuple, Optional, Union

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CTCTextEncoder:
    # def __init__(
    #     self,
    #     alphabet: Optional[List[str]] = None,
    #     arpa_path: Optional[str] = None,
    #     binary_path: Optional[str] = None,
    #     unigram_path: Optional[str] = None,
    #     use_bpe: bool = False,
    #     pretrained_tokenizer: str = "bert-base-uncased",
    #     lm_weight: float = 0.5,
    #     beam_size: int = 100,
    #     blank_token: str = "<blank>",
    #     **kwargs
    # ):
    #     self.use_bpe = use_bpe
    #     self.beam_size = beam_size
    #     self.lm_weight = lm_weight
    #     self.arpa_path = arpa_path
    #     self.binary_path = binary_path
    #     self.blank_token = blank_token  # Unique blank token

    #     self.printed_samples = 0
    #     self.max_printed_samples = 5
    #     print('CTC Text Encoder:')
    #     print('use_bpe:', use_bpe)
    #     print('lm_weight:', lm_weight)
    #     print('beam_size:', beam_size)
    #     print('binary_path:', binary_path)

    #     # Load unigrams if provided
    #     self.unigrams = None
    #     if unigram_path and os.path.exists(unigram_path):
    #         print(f"Loading unigrams from: {unigram_path}")
    #         with open(unigram_path, 'r', encoding='utf-8') as f:
    #             self.unigrams = [line.strip().lower() for line in f if line.strip()]
    #         print(f"Loaded {len(self.unigrams)} unigrams")

    #     # Initialize vocabulary
    #     if self.use_bpe:
    #         self._initialize_bpe_vocab(pretrained_tokenizer)
    #     else:
    #         self._initialize_char_vocab(alphabet)

    #     # Ensure <blank> is at index 0
    #     assert self.vocab[0] == self.blank_token, "Blank token must be at index 0 in vocab."

    #     # Create index mappings
    #     self.ind2char = dict(enumerate(self.vocab))
    #     self.char2ind = {v: k for k, v in self.ind2char.items()}
    #     self.blank_index = self.char2ind[self.blank_token]

    #     print(f"\nVocabulary Info:")
    #     print(f"Size: {len(self.vocab)}")
    #     print("Full Vocabulary (up to first 50 tokens):", self.vocab[:50])
    #     print(f"Blank token: {self.blank_token}, Blank index: {self.blank_index}")

    #     print("Sample ind2char mappings:", {k: self.ind2char[k] for k in list(self.ind2char.keys())[:10]})
    #     print("Sample char2ind mappings:", {k: self.char2ind[k] for k in list(self.char2ind.keys())[:10]})

    #     self._initialize_language_model()

    # def __init__(self, *args, **kwargs):
    #     # Instead of custom code or BPE initialization:
    #     self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    #     # The Wav2Vec2CTCTokenizer automatically has <pad>, <unk>, and <s> tokens 
    #     # and uses <pad> as CTC blank. You can inspect its vocabulary:
    #     print("Wav2Vec2 vocab:", self.processor.tokenizer.get_vocab())
    #     self.vocab = self.processor.tokenizer.get_vocab()
    #     # The blank token is typically set as <pad> for Wav2Vec2CTCTokenizer
    #     self.blank_token = self.processor.tokenizer.pad_token
    #     self.ind2char = {v: k for k, v in self.processor.tokenizer.get_vocab().items()}
    #     self.char2ind = {k: v for k, v in self.processor.tokenizer.get_vocab().items()}
    #     self.blank_index = self.char2ind[self.blank_token]
    #     print("Blank token:", self.blank_token, "Blank index:", self.blank_index)

    #     self.beam_size = 10
    #     self._initialize_language_model()

    import os
from typing import List, Optional, Union
import torch
from transformers import Wav2Vec2CTCTokenizer
from ctcdecode import CTCBeamDecoder

class CTCTextEncoder:
    def __init__(
        self,
        use_wav2vec2: bool = True,
        pretrained_model_name: str = "facebook/wav2vec2-base-960h",
        lm_path: Optional[str] = None,  # Path to KenLM binary
        lm_weight: float = 0.5,
        beam_size: int = 100,
        blank_token: str = "<pad>",
        unk_token: str = "<unk>",
        max_printed_samples: int = 2,
        **kwargs
    ):
        """
        Initialize encoder with Wav2Vec2 tokenizer and beam search decoder.

        Args:
            use_wav2vec2 (bool): Whether to use Wav2Vec2 tokenizer.
            pretrained_model_name (str): Pretrained model name/path.
            lm_path (Optional[str]): Path to KenLM language model binary file.
            lm_weight (float): Language model weight for beam search.
            beam_size (int): Beam size for decoding.
            blank_token (str): Token to be used as the CTC blank token.
            unk_token (str): Token to be used for unknown tokens.
            max_printed_samples (int): Max samples to print debug info per epoch.
            **kwargs: Additional arguments.
        """
        self.use_wav2vec2 = use_wav2vec2
        self.beam_size = beam_size
        self.lm_weight = lm_weight
        self.blank_token = blank_token  # Typically <pad>
        self.unk_token = unk_token
        self.printed_samples = 0
        self.max_printed_samples = max_printed_samples

        if self.use_wav2vec2:
            self._initialize_wav2vec2_tokenizer(pretrained_model_name)
        else:
            # Initialize other tokenizer, e.g., BPE or char-based
            raise NotImplementedError("Only Wav2Vec2 tokenizer is implemented in this example.")

        # Initialize decoder
        if lm_path:
            if not os.path.exists(lm_path):
                raise FileNotFoundError(f"Language model file not found at {lm_path}")
            self._initialize_decoder(lm_path)
        else:
            # Initialize decoder without language model
            self.decoder = CTCBeamDecoder(
                vocabulary=self.vocab,
                beam_width=self.beam_size,
                blank_id=self.blank_index,
                num_processes=4,  # Adjust based on your CPU cores
                log_probs_input=True,
                max_symbols_per_step=30,
                cutoff_prob=1.0,
                cutoff_top_n=40,
                alpha=self.lm_weight,  # Language model weight
                beta=1.0,  # Language model word bonus
                num_results=1,
            )
            print("Initialized CTCBeamDecoder without language model.")

    def _initialize_wav2vec2_tokenizer(self, pretrained_model_name: str):
        """
        Initialize Wav2Vec2 tokenizer and create mappings.

        Args:
            pretrained_model_name (str): Pretrained model name/path.
        """
        # Load Wav2Vec2CTCTokenizer
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name)

        # Check if blank token is <pad>
        if self.blank_token not in self.tokenizer.get_vocab():
            print(f"Blank token '{self.blank_token}' not found in tokenizer vocabulary, adding it.")
            self.tokenizer.add_tokens([self.blank_token])
        else:
            print(f"'{self.blank_token}' already exists in the tokenizer's vocabulary.")

        # Ensure that <pad> is set as the blank token
        # Wav2Vec2CTCTokenizer uses <pad> as CTC blank by default
        assert self.tokenizer.pad_token == self.blank_token, "Blank token must be <pad> in Wav2Vec2CTCTokenizer."

        # Build index mappings
        self.vocab = list(self.tokenizer.get_vocab().keys())
        self.ind2char = {i: tok for i, tok in enumerate(self.vocab)}
        self.char2ind = {tok: i for i, tok in enumerate(self.vocab)}
        self.blank_index = self.char2ind[self.blank_token]

        # Handle unknown tokens
        if self.unk_token not in self.tokenizer.get_vocab():
            print(f"Unknown token '{self.unk_token}' not found in tokenizer vocabulary, adding it.")
            self.tokenizer.add_tokens([self.unk_token])
            self.vocab.append(self.unk_token)
            self.ind2char[len(self.vocab)-1] = self.unk_token
            self.char2ind[self.unk_token] = len(self.vocab)-1
        else:
            print(f"'{self.unk_token}' already exists in the tokenizer's vocabulary.")

        # Debug prints
        print("\nVocabulary Info:")
        print(f"Size: {len(self.vocab)}")
        print("Full Vocabulary (up to first 50 tokens):", self.vocab[:50])
        print(f"Blank token: {self.blank_token}, Blank index: {self.blank_index}")
        print("Sample ind2char mappings:", {k: self.ind2char[k] for k in list(self.ind2char.keys())[:10]})
        print("Sample char2ind mappings:", {k: self.char2ind[k] for k in list(self.char2ind.keys())[:10]})

    def _initialize_decoder(self, lm_path: str):
        """
        Initialize the beam search decoder with a language model.

        Args:
            lm_path (str): Path to KenLM language model binary file.
        """
        self.decoder = CTCBeamDecoder(
            vocabulary=self.vocab,
            beam_width=self.beam_size,
            blank_id=self.blank_index,
            num_processes=4,  # Adjust based on your CPU cores
            log_probs_input=True,
            max_symbols_per_step=30,
            cutoff_prob=1.0,
            cutoff_top_n=40,
            alpha=self.lm_weight,  # Language model weight
            beta=1.0,  # Language model word bonus
            num_results=1,
            # Path to the language model binary
            # You might need to specify the `language_model_path` parameter based on `ctcdecode` version
            # e.g., language_model_path=lm_path
        )
        print(f"Initialized CTCBeamDecoder with language model from {lm_path}.")

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text into token indices.

        Args:
            text (str): Input text to encode.

        Returns:
            torch.Tensor: Encoded tensor of shape [1, seq_len].
        """
        if self.printed_samples < self.max_printed_samples:
            original_text = text
            text = self.normalize_text(text)
            print(f"\nEncoding text:")
            print(f" Original: '{original_text}'")
            print(f" Normalized: '{text}'")
        
        # Tokenize and encode
        encoded = self.tokenizer(text, return_tensors="pt", padding=False, truncation=False)
        input_ids = encoded.input_ids  # Shape: [1, seq_len]

        if self.use_wav2vec2 and self.printed_samples < self.max_printed_samples:
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            print(f" Tokens (Wav2Vec2): {tokens}")
            print(f" Token indices: {input_ids.tolist()[0]}")
            self.printed_samples += 1

        return input_ids

    def decode(self, logits: torch.Tensor) -> List[str]:
        """
        Decode logits using beam search decoder.

        Args:
            logits (torch.Tensor): Logits output from the model of shape [batch_size, time_steps, vocab_size].

        Returns:
            List[str]: List of decoded strings for each item in the batch.
        """
        if not hasattr(self, 'decoder'):
            raise AttributeError("Decoder has not been initialized.")

        # Convert logits to log probabilities if they aren't already
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

        # Decode using beam search
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(log_probs)

        decoded_predictions = []
        for beam in beam_results:
            decoded_text = self.decoder.decode_beam(beam[0])
            decoded_predictions.append(decoded_text)

        return decoded_predictions

    def decode_beam(self, beam: List[int]) -> str:
        """
        Convert a single beam of indices to text.

        Args:
            beam (List[int]): List of token indices from the beam search.

        Returns:
            str: Decoded text string.
        """
        # Remove duplicates and blanks
        decoded = []
        last_token = self.blank_index

        for token_idx in beam:
            if token_idx == self.blank_index:
                continue
            if token_idx != last_token:
                decoded.append(token_idx)
                last_token = token_idx

        # Map indices to tokens
        try:
            tokens = [self.ind2char[idx] for idx in decoded]
            if self.printed_samples < self.max_printed_samples:
                print(f" Decoded tokens: {tokens}")
        except KeyError as e:
            raise ValueError(f"Invalid token index encountered during decoding: {e}")

        # Convert tokens to string
        text = self.tokenizer.convert_tokens_to_string(tokens)
        if self.printed_samples < self.max_printed_samples:
            print(f" Decoded text before cleanup: '{text}'")
        
        # Clean up tokenization artifacts
        text = self.tokenizer.clean_up_tokenization(text)

        # Remove blank token if any
        text = text.replace(self.blank_token, "").strip()
        text = ' '.join(text.split())

        if self.printed_samples < self.max_printed_samples:
            print(f" Decoded text after cleanup: '{text}'")
            self.printed_samples += 1

        return text

    def normalize_text(self, text: str) -> str:
        """
        Normalize the input text. Modify this as per your requirements.

        Args:
            text (str): Text to normalize.

        Returns:
            str: Normalized text.
        """
        # Example normalization: lowercase and remove extra spaces
        return ' '.join(text.lower().strip().split())


    # def _initialize_bpe_vocab(self, pretrained_tokenizer: str):
    #     """Initialize vocabulary using BPE tokens from a pretrained tokenizer."""
    #     self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    #     # Add the unique blank token to the tokenizer's vocabulary if not present
    #     if self.blank_token not in self.tokenizer.get_vocab():
    #         self.tokenizer.add_tokens([self.blank_token])
    #         print(f"Added '{self.blank_token}' to the tokenizer's vocabulary.")
    #     else:
    #         print(f"'{self.blank_token}' already exists in the tokenizer's vocabulary.")

    #     # Exclude all other special tokens from the BPE vocabulary
    #     special_tokens = set(self.tokenizer.all_special_tokens)
    #     self.bpe_tokens = [
    #         tok for tok in self.tokenizer.get_vocab().keys() if tok not in special_tokens
    #     ]

    #     # Define CTC's vocabulary: [blank] + BPE tokens
    #     self.vocab = [self.blank_token] + self.bpe_tokens
    #     print(f"Loaded BPE vocabulary of size: {len(self.vocab)}")

    def decode(self, indices: List[int]) -> str:
        return self.decode_indices(indices)

    def normalize_text(self, text: str) -> str:
        """
        Normalize the input text. Modify this as per your requirements.

        Args:
            text (str): Text to normalize.

        Returns:
            str: Normalized text.
        """
        # Example normalization: lowercase and remove extra spaces
        return ' '.join(text.lower().strip().split())



    def _initialize_bpe_vocab(self, pretrained_tokenizer: str):
        """Initialize vocabulary using BPE tokens from a pretrained tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

        # Add the unique blank token to the tokenizer's vocabulary if not present
        if self.blank_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.blank_token])
            print(f"Added '{self.blank_token}' to the tokenizer's vocabulary.")
        else:
            print(f"'{self.blank_token}' already exists in the tokenizer's vocabulary.")

        # Exclude all other special tokens
        special_tokens = set(self.tokenizer.all_special_tokens)
        self.bpe_tokens = [
            tok for tok in self.tokenizer.get_vocab().keys()
            if tok not in special_tokens
        ]

        # Filter out [unused*] tokens
        self.bpe_tokens = [tok for tok in self.bpe_tokens if not tok.startswith("[unused")]

        # Define CTC's vocabulary: [blank] + BPE tokens
        self.vocab = [self.blank_token] + self.bpe_tokens
        print(f"Loaded BPE vocabulary of size: {len(self.vocab)}")

        
    
    def _initialize_char_vocab(self, alphabet: Optional[List[str]]):
        """Initialize vocabulary using a traditional character set."""
        if alphabet is None:
            if self.unigrams:
                # Build alphabet from unigrams
                alphabet_set = set(char for word in self.unigrams for char in word)
                alphabet_set.add(" ")
                alphabet = sorted(list(alphabet_set))
            else:
                # Default to lowercase letters and space
                alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        # Insert blank token at the beginning of the vocabulary
        self.vocab = [self.blank_token] + list(self.alphabet)
        print(f"Loaded character vocabulary of size: {len(self.vocab)}")

    def _initialize_language_model(self):
        """Initialize language model with explicit blank token handling."""
        self.lm = None
        self.decoder = None

        model_path = self.binary_path if self.binary_path else self.arpa_path
        print('model_path: ', model_path )
        if not model_path or not os.path.exists(model_path):
            print("No language model path provided or file does not exist.")
            return

        try:
            self.lm = kenlm.Model(model_path)
            print(f"Loaded {'binary' if self.binary_path else 'ARPA'} language model.")

            # Ensure blank token is at index 0
            labels = [self.blank_token] + [c for c in self.vocab if c != self.blank_token]

            # Initialize CTC decoder with pyctcdecode
            decoder_config = {
                "labels": labels,
                "kenlm_model_path": model_path,
                "alpha": self.lm_weight,  # LM weight
                "beta": 0.1,               # Word insertion penalty
                "unk_score_offset": -10.0, # UNK score offset
            }

            if self.unigrams:
                decoder_config["unigrams"] = self.unigrams

            self.decoder = build_ctcdecoder(**decoder_config)
            print("Successfully initialized language model and decoder.")

        except Exception as e:
            print(f"Warning: Failed to initialize decoder: {str(e)}")
            self.decoder = None

    # def encode(self, text: str) -> torch.Tensor:
    #     """
    #     Encode text with BPE or character-level encoding.

    #     Args:
    #         text (str): Input text to encode.

    #     Returns:
    #         torch.Tensor: Encoded tensor of shape [1, seq_len].
    #     """
        
    #     debug = False

    #     if self.printed_samples < self.max_printed_samples:
    #         original_text = text
    #         text = self.normalize_text(text)
    #         if debug:
    #             print(f"samples: {str(self.printed_samples)}")
    #             print(f"\nEncoding text:")
    #             print(f" Original: '{original_text}'")
    #             print(f" Normalized: '{text}'")

    #     if self.use_bpe:
    #         try:
    #             # Tokenize the text into BPE tokens without adding special tokens
    #             tokens = self.tokenizer.tokenize(text, add_special_tokens=False)
    #             if debug and self.printed_samples < self.max_printed_samples:
    #                 print(f" Tokens (BPE): {tokens}")
    #             # Map tokens to CTC indices
    #             token_indices = [self.char2ind[token] for token in tokens]
    #             if debug and self.printed_samples < self.max_printed_samples:
    #                 print(f" Token indices: {token_indices}")
    #             self.printed_samples += 1
    #             return torch.tensor(token_indices).unsqueeze(0)  # Shape: [1, seq_len]
    #         except KeyError as e:
    #             unknown_tokens = set([token for token in tokens if token not in self.char2ind])
    #             raise Exception(f"Unknown tokens: '{' '.join(unknown_tokens)}'")
    #         except Exception as e:
    #             raise Exception(f"BPE encoding error: {str(e)}")
    #     else:
    #         # Character-level encoding remainds the same
    #         if debug and self.printed_samples < self.max_printed_samples:
    #             chars = list(text)
    #             print(f" Chars: {chars}")
    #             char_indices = [self.char2ind.get(c, None) for c in chars]
    #             print(f" Char indices: {char_indices}")
    #             self.printed_samples += 1
                    
    #         try:
    #             return torch.tensor([self.char2ind[char] for char in text]).unsqueeze(0)
    #         except KeyError as e:
    #             unknown_chars = set([char for char in text if char not in self.char2ind])
    #             raise Exception(f"Unknown chars: '{' '.join(unknown_chars)}'")

    def encode(self, text: str) -> torch.Tensor:
        encoded = self.processor.tokenizer(text, return_tensors="pt", padding=False, truncation=False)
        # `input_ids` will contain the encoded tensor
        return encoded.input_ids

    
    def decode(self, indices: List[int]) -> str:
        """
        Decode indices to text.

        Args:
            indices (List[int]): List of CTC indices to decode.

        Returns:
            str: Decoded text.
        """
        if self.use_bpe:
            # Filter out blank tokens and ensure indices are within range
            valid_indices = [
                idx for idx in indices
                if idx != self.blank_index and 0 < idx < len(self.ind2char)
            ]
            try:
                # Map CTC indices back to tokens
                tokens = [self.ind2char[idx] for idx in valid_indices]
                # Convert tokens to string using tokenizer's method
                return self.tokenizer.convert_tokens_to_string(tokens)
            except KeyError as e:
                # Handle any unexpected indices gracefully
                return " ".join([self.ind2char[idx] for idx in valid_indices if idx in self.ind2char])
        else:
            # Character-level decoding
            return "".join([self.ind2char[idx] for idx in indices if idx != self.blank_index])

    # def decode_indices(self, indices: Union[torch.Tensor, List[int], np.ndarray]) -> str:
    #     """
    #     Decode a sequence of indices into text without performing argmax.

    #     Args:
    #         indices (Union[torch.Tensor, List[int], np.ndarray]): Indices to decode.

    #     Returns:
    #         str: Decoded text.
    #     """
        
    #     debug = True
    #     # print(f"Decoding indices: {indices}")
        
    #     # Convert input to list if it's a Tensor or numpy array
    #     if isinstance(indices, torch.Tensor):
    #         indices = indices.tolist()
    #     elif isinstance(indices, np.ndarray):
    #         indices = indices.tolist()
    #     elif not isinstance(indices, list):
    #         raise TypeError("decode_indices expects a list, torch.Tensor, or numpy.ndarray.")

    #     decoded = []
    #     last_token = self.blank_index

    #     for token_idx in indices:
    #         if token_idx == self.blank_index:
    #             continue  # Skip blank tokens
    #         if token_idx != last_token:
    #             decoded.append(token_idx)
    #             last_token = token_idx

    #     if self.use_bpe:
    #         # Convert list of token indices to string
    #         try:
    #             tokens = [self.ind2char[idx] for idx in decoded]
    #         except KeyError as e:
    #             raise ValueError(f"Invalid token index encountered during BPE decoding: {e}")
            
    #         # Convert tokens to string using tokenizer
    #         text = self.tokenizer.convert_tokens_to_string(tokens)
    #         if debug and self.printed_samples < self.max_printed_samples:
    #             print(f" Decoded text before final cleanup: '{text}'")
            
    #         # Clean up tokenization artifacts
    #         text = self.tokenizer.clean_up_tokenization(text)
            
    #         # Remove <blank>
    #         text = text.replace("<blank>", "")
    #         if debug and self.printed_samples < self.max_printed_samples:
    #             print(f" Decoded text after cleanup: '{text}'")
    #         self.printed_samples += 1 
    #         return text
    #     else:
    #         # Character-level decoding
    #         try:
    #             characters = [self.ind2char[idx] for idx in decoded]
    #         except KeyError as e:
    #             raise ValueError(f"Invalid character index encountered during decoding: {e}")
            
    #         return "".join(characters)

    # def decode_indices(self, indices: Union[torch.Tensor, List[int], np.ndarray]) -> str:
    #     import re
    #     debug = True

    #     if isinstance(indices, torch.Tensor):
    #         indices = indices.tolist()
    #     elif isinstance(indices, np.ndarray):
    #         indices = indices.tolist()
    #     elif not isinstance(indices, list):
    #         raise TypeError("decode_indices expects a list, torch.Tensor, or numpy.ndarray.")

    #     decoded = []
    #     last_token = self.blank_index

    #     for token_idx in indices:
    #         if token_idx == self.blank_index:
    #             continue  # Skip blank tokens
    #         if token_idx != last_token:
    #             decoded.append(token_idx)
    #             last_token = token_idx

    #     if self.use_bpe:
    #         try:
    #             tokens = [self.ind2char[idx] for idx in decoded]
    #         except KeyError as e:
    #             raise ValueError(f"Invalid token index encountered during BPE decoding: {e}")

    #         text = self.tokenizer.convert_tokens_to_string(tokens)
    #         if debug and self.printed_samples < 2:
    #             print(f" Decoded tokens: {tokens}")
    #             print(f" Decoded text before cleanup: '{text}'")

    #         text = self.tokenizer.clean_up_tokenization(text)
    #         text = text.replace("<blank>", "").strip()
    #         text = ' '.join(text.split())

    #         if debug and self.printed_samples < 2:
    #             print(f" Decoded text after cleanup: '{text}'")
    #         self.printed_samples += 1
    #         return text
    #     else:
    #         try:
    #             characters = [self.ind2char[idx] for idx in decoded]
    #         except KeyError as e:
    #             raise ValueError(f"Invalid character index encountered during decoding: {e}")

    #         text = "".join(characters)
    #         text = text.replace("<blank>", "").strip()
    #         text = ' '.join(text.split())
    #         if debug and self.printed_samples < 2:
    #             print(f" Decoded text after cleanup: '{text}'")
    #             self.printed_samples += 1
    #         return text

    def decode_indices(self, indices: Union[torch.Tensor, List[int], np.ndarray]) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.squeeze().tolist()
        return self.processor.tokenizer.decode(indices, skip_special_tokens=True).strip()

            
        
    # def ctc_decode(self, logits: Union[torch.Tensor, List[int], np.ndarray]) -> str:
    #     """
    #     Perform CTC decoding with proper handling for BPE and different input types.

    #     Args:
    #         logits (Union[torch.Tensor, List[int], np.ndarray]): Raw logits from the model or already-decoded indices.

    #     Returns:
    #         str: Decoded text.
    #     """
    #     # Convert input to torch.Tensor if it's not already
    #     if isinstance(logits, np.ndarray):
    #         logits = torch.from_numpy(logits)
    #     elif isinstance(logits, list):
    #         logits = torch.tensor(logits)
        
    #     # Determine input type based on tensor dimensions
    #     if logits.dim() == 3:
    #         # Assume logits shape is [batch_size, time_steps, vocab_size]
    #         # Perform argmax to get predicted indices
    #         predictions = torch.argmax(logits, dim=-1)[0].tolist()  # Shape: [time_steps]
    #         print("\n[DEBUG] Raw predicted indices:", predictions[:50])  # first 50 steps

    #         ### DEBUG PART ###
    #         topk = 3
    #         # Extract top-k predictions for up to first 5 timesteps
    #         probs = torch.softmax(logits, dim=-1)
    #         topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)

    #         for t in range(min(5, logits.size(1))):
    #             top_tokens = [self.ind2char[idx.item()] for idx in topk_indices[0, t]]
    #             top_values = topk_probs[0, t].tolist()
    #             print(f"[DEBUG] Time step {t}: Top-{topk} tokens = {top_tokens}, Probs = {top_values}")
    #         ### DEBUG PART ###

    #         decoded = []
    #         last_token = self.blank_index

    #         for token_idx in predictions:
    #             if token_idx == self.blank_index:
    #                 continue  # Skip blank tokens
    #             if token_idx != last_token:
    #                 decoded.append(token_idx)
    #                 last_token = token_idx

    #         if self.use_bpe:
    #             # Convert list of token indices to string
    #             try:
    #                 tokens = [self.ind2char[idx] for idx in decoded]
    #             except KeyError as e:
    #                 raise ValueError(f"Invalid token index encountered during BPE decoding: {e}")
                
    #             # Convert tokens to string using tokenizer
    #             text = self.tokenizer.convert_tokens_to_string(tokens)
    #             # Clean up tokenization artifacts
    #             text = self.tokenizer.clean_up_tokenization(text)
                
    #             # Remove <blank>
    #             text = text.replace("<blank>", "")
    #             return text
    #         else:
    #             # Character-level decoding
    #             try:
    #                 characters = [self.ind2char[idx] for idx in decoded]
    #             except KeyError as e:
    #                 raise ValueError(f"Invalid character index encountered during decoding: {e}")
                
    #             return "".join(characters)

    #     elif logits.dim() == 1:
    #         # Assume logits is a sequence of already-decoded indices
    #         return self.decode_indices(logits)
    #     else:
    #         raise ValueError(f"Unsupported logits shape: {logits.shape}. Expected 1D or 3D tensor.")

    def ctc_decode(self, logits: Union[torch.Tensor, List[int], np.ndarray]) -> str:
        import re

        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        elif isinstance(logits, list):
            logits = torch.tensor(logits)

        if logits.dim() == 3:
            # Print raw predictions and top-k tokens for a few timesteps
            predictions = torch.argmax(logits, dim=-1)[0].tolist()  # [time_steps]
            print("\n[DEBUG] Raw predicted indices:", predictions[:50])  # first 50 steps

            topk = 3
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
            for t in range(min(5, logits.size(1))):
                top_tokens = [self.ind2char[idx.item()] for idx in topk_indices[0, t]]
                top_values = topk_probs[0, t].tolist()
                print(f"[DEBUG] Time step {t}: Top-{topk} tokens = {top_tokens}, Probs = {top_values}")

            decoded = []
            last_token = self.blank_index
            for token_idx in predictions:
                if token_idx == self.blank_index:
                    continue
                if token_idx != last_token:
                    decoded.append(token_idx)
                    last_token = token_idx

            if self.use_bpe:
                try:
                    tokens = [self.ind2char[idx] for idx in decoded]
                    print(f"Decoded tokens: {tokens}")  # Debug print
                except KeyError as e:
                    raise ValueError(f"Invalid token index encountered during BPE decoding: {e}")

                text = self.tokenizer.convert_tokens_to_string(tokens)
                print(f"Decoded text before cleanup: '{text}'")  # Debug print

                text = self.tokenizer.clean_up_tokenization(text)
                text = text.replace("<blank>", "")
                text = ' '.join(text.split())
                print(f"Decoded text after cleanup: '{text}'")  # Debug print

                self.printed_samples += 1
                return text
            else:
                try:
                    characters = [self.ind2char[idx] for idx in decoded]
                    print(f"Decoded characters: {characters}")  # Debug print
                except KeyError as e:
                    raise ValueError(f"Invalid character index encountered during decoding: {e}")

                text = "".join(characters)
                text = text.replace("<blank>", "").strip()
                text = ' '.join(text.split())
                print(f"Decoded text after cleanup: '{text}'")  # Debug print
                self.printed_samples += 1
                return text

        elif logits.dim() == 1:
            # Already have predicted indices, just decode
            return self.decode_indices(logits)
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}. Expected 1D or 3D tensor.")



    def __len__(self):
        return len(self.vocab)


####### BEAM SEARCH AND OTHER PART ########
    
    def ctc_beam_search(self, probs, beam_size: int = 40,
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

    ### Updated to fix <blank> token problem
    # def ctc_beam_search(self, probs, beam_size: int = 40,
    #                 use_lm: bool = False, debug: bool = False) -> List[Tuple[str, float]]:
    #     """
    #     Beam search with enhanced BPE support
    #     """
    #     beam_size = self.beam_size
    #     debug = False
    #     # debug = True

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

    #             # Format beams and clean up BPE tokens
    #             formatted_beams = []
    #             for beam in beams[:beam_size]:
    #                 text = beam[0]  # Get the text
    #                 acoustic_score = beam[3]
    #                 lm_score = beam[4]

    #                 # Clean up BPE tokens if using BPE
    #                 if self.use_bpe:
    #                     text = self.tokenizer.clean_up_tokenization(text)

    #                 # Combine scores with weighting
    #                 combined_score = (1 - self.lm_weight) * acoustic_score + self.lm_weight * lm_score

    #                 # Apply length normalization
    #                 text_len = max(1, len(text.split()))
    #                 normalized_score = combined_score / text_len

    #                 formatted_beams.append((text, normalized_score))

    #             if debug:
    #                 print("\nFormatted beam results with BPE:")
    #                 for text, score in formatted_beams[:3]:
    #                     print(f"Text: '{text}', Score: {score:.4f}")

    #             if formatted_beams:
    #                 # Sort beams
    #                 sorted_beams = sorted(formatted_beams, key=lambda x: -x[1])
                    
    #                 # Remove <blank> from the final text of each beam at the very end
    #                 final_results = []
    #                 for txt, sc in sorted_beams:
    #                     # Remove <blank> right before returning
    #                     txt = txt.replace("<blank>", "")
    #                     # Also consider trimming spaces if needed
    #                     txt = ' '.join(txt.split())
    #                     final_results.append((txt, sc))

    #                 return final_results
    #             else:
    #                 print("No valid beams found, falling back to standard beam search")
    #                 return self._standard_beam_search(probs, beam_size, debug)

    #         except Exception as e:
    #             print(f"Beam search with LM failed: {str(e)}, falling back to standard beam search")
    #             return self._standard_beam_search(probs, beam_size, debug)
    #     else:
    #         return self._standard_beam_search(probs, beam_size, debug)

    
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
            
            if debug and t < self.max_printed_samples:  # Print first two timesteps
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