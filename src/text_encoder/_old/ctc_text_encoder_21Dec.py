import re
from collections import defaultdict
import torch
import kenlm
from transformers import Wav2Vec2Processor, AutoProcessor
from pyctcdecode import build_ctcdecoder
import numpy as np
import os

from typing import List, Tuple, Optional, Union

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CTCTextEncoder:
    def __init__(
        self,
        arpa_path: Optional[str] = None,
        binary_path: Optional[str] = None,
        unigram_path: Optional[str] = None,
        pretrained_tokenizer: str = "facebook/wav2vec2-base-960h",
        lm_weight: float = 0.5,
        beam_size: int = 100,
        blank_token: str = "[pad]",  # Blank token as <pad> for Wav2Vec2
        unk_token: str = "[unk]",     # UNK token
        **kwargs
    ):
        """
        Initialize encoder with Wav2Vec2 tokenizer and beam search decoder.

        Changes:
        - Ensure normalization is strictly lowercase with only [a-z ].
        """
        self.beam_size = beam_size
        self.lm_weight = lm_weight
        self.arpa_path = arpa_path
        self.binary_path = binary_path
        self.blank_token = blank_token
        self.unk_token = unk_token
        self.printed_samples = 0
        self.max_printed_samples = 5
        print('CTC Text Encoder:')
        print('pretrained_tokenizer:', pretrained_tokenizer)
        print('lm_weight:', lm_weight)
        print('beam_size:', beam_size)
        print('binary_path:', binary_path)

        # unigram_path = None

        # Load unigrams if provided
        
        self.unigrams = None
        
        # if unigram_path and os.path.exists(unigram_path):
        #     print(f"Loading unigrams from: {unigram_path}")
        #     with open(unigram_path, 'r', encoding='utf-8') as f:
        #         self.unigrams = [line.strip().lower() for line in f if line.strip()]
        #     print(f"Loaded {len(self.unigrams)} unigrams")
        
        if unigram_path and os.path.exists(unigram_path):
            print(f"Loading unigrams from: {unigram_path}")
            with open(unigram_path, 'r', encoding='utf-8') as f:
                self.unigrams = [line.strip().lower() for line in f if line.strip()]
            print(f"Loaded {len(self.unigrams)} unigrams")


        self._initialize_wav2vec2_tokenizer(pretrained_tokenizer)

        # Create index mappings
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.blank_index = self.char2ind[self.blank_token]

        print(f"\nVocabulary Info:")
        print(f"Size: {len(self.vocab)}")
        print("Full Vocabulary (up to first 50 tokens):", self.vocab[:50])
        print(f"Blank token: {self.blank_token}, Blank index: {self.blank_index}")

        print("Sample ind2char mappings:", {k: self.ind2char[k] for k in list(self.ind2char.keys())[:10]})
        print("Sample char2ind mappings:", {k: self.char2ind[k] for k in list(self.char2ind.keys())[:10]})

        self._initialize_language_model()

    # def _initialize_wav2vec2_tokenizer(self, pretrained_tokenizer: str):
    #     """Initialize vocabulary using Wav2Vec2 tokenizer."""
    #     self.processor = Wav2Vec2Processor.from_pretrained(pretrained_tokenizer)

    #     # Add the unique blank token if not present
    #     if self.blank_token not in self.processor.tokenizer.get_vocab():
    #         self.processor.tokenizer.add_tokens([self.blank_token])
    #         print(f"Added '{self.blank_token}' to the tokenizer's vocabulary.")
    #     else:
    #         print(f"'{self.blank_token}' already exists in the tokenizer's vocabulary.")

    #     # Add the UNK token if not present
    #     if self.unk_token not in self.processor.tokenizer.get_vocab():
    #         self.processor.tokenizer.add_tokens([self.unk_token])
    #         print(f"Added '{self.unk_token}' to the tokenizer's vocabulary.")
    #     else:
    #         print(f"'{self.unk_token}' already exists in the tokenizer's vocabulary.")

    #     # Get vocab, convert to lowercase and replace '|' with ' '
    #     original_vocab = list(self.processor.tokenizer.get_vocab().keys())
    #     self.vocab = [x.lower() for x in original_vocab]
    #     self.vocab = [t.replace('|', ' ') for t in self.vocab]

    #     # Debug: Print a few tokens after modification
    #     print("Modified Vocabulary (first 20 tokens):", self.vocab[:20])
    

    def _initialize_wav2vec2_tokenizer(self, pretrained_tokenizer: str):
        """Initialize vocabulary using Wav2Vec2 tokenizer."""
        # Initialize Wav2Vec2Processor
        # self.processor = Wav2Vec2Processor.from_pretrained(pretrained_tokenizer)

        

        # # Add the UNK token if not present
        # if self.unk_token not in self.processor.tokenizer.get_vocab():
        #     self.processor.tokenizer.add_tokens([self.unk_token])
        #     print(f"Added '{self.unk_token}' to the tokenizer's vocabulary.")
        # else:
        #     print(f"'{self.unk_token}' already exists in the tokenizer's vocabulary.")

        # # Get vocab without altering the case
        # self.vocab = list(self.processor.tokenizer.get_vocab().keys())

        # vocab_dict = processor.tokenizer.get_vocab()
        # sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        # labels_adj = list(sorted_vocab_dict.keys())

        # self.vocab = [t.replace('|', ' ') for t in self.vocab]

        # # Debugging: Inspect the tokenizer's vocabulary casing
        # print("\n--- Tokenizer Vocabulary Inspection ---")
        # sample_size = 100  # Adjust as needed
        # sample_tokens = self.vocab[:sample_size]
        # print(f"First {sample_size} tokens in vocabulary:")
        # print(sample_tokens)

        # # Save the full tokenizer vocabulary to a file for comparison
        # with open("tokenizer_vocab.txt", "w", encoding="utf-8") as f:
        #     for token in self.vocab:
        #         f.write(f"{token}\n")
        # print("Full tokenizer vocabulary saved to 'tokenizer_vocab.txt'.")
        # print("----------------------------------------\n")

        self.processor = AutoProcessor.from_pretrained("hf-test/xls-r-300m-sv")

        vocab_dict = self.processor.tokenizer.get_vocab()
        sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

        self.labels = list(sorted_vocab_dict.keys())

        # Get vocab, convert to lowercase and replace '|' with ' '
        original_vocab = list(self.processor.tokenizer.get_vocab().keys())
        # self.vocab = [x.lower() for x in original_vocab]
        self.vocab = sorted_vocab_dict
        self.vocab = [t.replace('|', ' ') for t in self.vocab]

        # Add the unique blank token to the tokenizer's vocabulary if not present
        if self.blank_token not in self.processor.tokenizer.get_vocab():
            self.processor.tokenizer.add_tokens([self.blank_token])
            print(f"Added '{self.blank_token}' to the tokenizer's vocabulary.")
        else:
            print(f"'{self.blank_token}' already exists in the tokenizer's vocabulary.")

        # Debug: Print a few tokens after modification
        print("Modified Vocabulary (first 20 tokens):", self.vocab[:20])


    def _initialize_language_model(self):
        """Initialize language model with explicit blank token handling."""
        self.lm = None
        self.decoder = None

        model_path = self.binary_path if self.binary_path else self.arpa_path
        print('model_path: ', model_path)
        if not model_path or not os.path.exists(model_path):
            print("No language model path provided or file does not exist.")
            return

        try:
            self.lm = kenlm.Model(model_path)
            print(f"Loaded {'binary' if self.binary_path else 'ARPA'} language model.")

            # labels = [self.blank_token] + [c for c in self.vocab if c != self.blank_token]

            decoder_config = {
                "labels": self.labels,
                "kenlm_model_path": model_path,
                "alpha": self.lm_weight,
                "beta": 0.1,
                "unk_score_offset": -10.0,
            }

            if self.unigrams:
                print("\n--- Unigrams List ---")
                # Save the unigrams to a file
                with open("unigrams_list.txt", "w", encoding="utf-8") as f:
                    for unigram in self.unigrams:
                        f.write(f"{unigram}\n")
                print(f"Unigrams list saved to 'unigrams_list.txt'. Total unigrams: {len(self.unigrams)}")
                print("----------------------\n")
                decoder_config["unigrams"] = self.unigrams

            self.decoder = build_ctcdecoder(**decoder_config)
            print("Successfully initialized language model and decoder.")

        except Exception as e:
            print(f"Warning: Failed to initialize decoder: {str(e)}")
            self.decoder = None


    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text with Wav2Vec2 tokenizer.
        """
        debug = False

        if self.printed_samples < self.max_printed_samples:
            original_text = text
            text = self.normalize_text(text)
            if debug:
                print(f"samples: {str(self.printed_samples)}")
                print(f"\nEncoding text:")
                print(f" Original: '{original_text}'")
                print(f" Normalized: '{text}'")
                for ch in text:
                    print(ch, ord(ch))


        try:
            encoded = self.processor.tokenizer(text, return_tensors="pt", padding=False, truncation=False)
            token_indices = encoded.input_ids[0].tolist()
            if self.printed_samples < self.max_printed_samples:
                # Convert indices to tokens from self.vocab
                tokens = [self.vocab[idx] if 0 <= idx < len(self.vocab) else "<invalid>" for idx in token_indices]
                # print(f" Tokens (lowercased and '|'->' '): {tokens}")
                # print(f" Token indices: {token_indices}")
                self.printed_samples += 1
            return torch.tensor(token_indices).unsqueeze(0)
        except KeyError as e:
            unknown_tokens = set([token for token in text.split() if token not in self.char2ind])
            raise Exception(f"Unknown tokens: '{' '.join(unknown_tokens)}'")
        except Exception as e:
            raise Exception(f"Encoding error: {str(e)}")

    # latest ver
    # def decode(self, indices: List[int]) -> str:
    #     """
    #     Decode indices to text using beam search decoder if available.
    #     """
    #     if self.decoder:
    #         decoded_text = self.decoder.decode(indices)
    #         # convert to lower case
    #         decoded_text = decoded_text.lower()
    #         return decoded_text
    #     else:
    #         decoded_text = self.decode_simple(indices)
    #         # convert to lower case
    #         decoded_text = decoded_text.lower()
    #         return
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode indices to text using beam search decoder if available.
        """
        if self.decoder:
            decoded_text = self.decoder.decode(indices)
            # Convert to lower case
            decoded_text = decoded_text.lower()
            return decoded_text
        else:
            decoded_text = self.decode_simple(indices)
            # Convert to lower case
            decoded_text = decoded_text.lower()
            return decoded_text  # Ensure the decoded text is returned

    # latest ver
    # def decode_simple(self, indices: List[int]) -> str:
    #     """
    #     Simple CTC decoding without language model.
    #     """
    #     valid_indices = [
    #         idx for idx in indices
    #         if idx != self.blank_index and 0 <= idx < len(self.ind2char)
    #     ]
    #     try:
    #         tokens = [self.ind2char[idx] for idx in valid_indices]
    #         text = " ".join(tokens).strip().lower()
    #         return self.processor.tokenizer.clean_up_tokenization(text)
    #     except KeyError as e:
    #         return " ".join([self.ind2char[idx] for idx in valid_indices if idx in self.ind2char])

    def decode_simple(self, indices: List[int]) -> str:
        """
        Simple CTC decoding without language model.
        Collapses consecutive duplicate tokens and removes blanks.
        """
        decoded_chars = []
        previous_idx = None

        for idx in indices:
            if idx == self.blank_index:
                previous_idx = idx
                continue  # Skip blank tokens
            if idx == previous_idx:
                continue  # Skip duplicate tokens
            if 0 <= idx < len(self.ind2char):
                char = self.ind2char[idx]
                decoded_chars.append(char)
            previous_idx = idx

        # Join characters without spaces and convert to lowercase
        text = "".join(decoded_chars).strip().lower()

        # Clean up tokenization using the tokenizer's method
        return self.processor.tokenizer.clean_up_tokenization(text)


    def decode_logits(self, logits: Union[torch.Tensor, List[List[float]], np.ndarray]) -> str:
        """
        Decode logits using the decoder if available, otherwise use greedy decoding.
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        elif isinstance(logits, list):
            logits = np.array(logits)
        elif not isinstance(logits, np.ndarray):
            raise TypeError("logits must be a torch.Tensor, list of lists, or numpy.ndarray")

        if logits.ndim == 3:
            logits = logits[0]

        if logits.ndim != 2:
            raise ValueError(f"Logits should be 2D (time_steps, vocab_size), got {logits.ndim}D")

        if self.decoder:
            decoded_text = self.decoder.decode(logits)
            return decoded_text
        else:
            predicted_indices = np.argmax(logits, axis=-1).tolist()
            return self.decode_simple(predicted_indices)

    def decode_indices(self, indices: Union[torch.Tensor, List[int], np.ndarray]) -> str:
        """
        Decode token indices to text using simple decoding (no LM).
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.squeeze().tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()
        elif not isinstance(indices, list):
            raise TypeError("decode_indices expects a list, torch.Tensor, or numpy.ndarray.")

        return self.decode_simple(indices)


    # latest ver
    def ctc_decode(self, logits: Union[torch.Tensor, List[int], np.ndarray]) -> str:
        """
        Perform CTC decoding on logits.
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        elif isinstance(logits, list):
            logits = torch.tensor(logits)

        if logits.dim() == 3:
            decoded_text = self.decode_logits(logits)
            return decoded_text
        elif logits.dim() == 2:
            decoded_text = self.decode_logits(logits)
            return decoded_text
        elif logits.dim() == 1:
            decoded_text = self.decode_indices(logits)
            return decoded_text
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}. Expected 1D, 2D, or 3D.")

    def ctc_beam_search(self, probs, beam_size: int = 40,
                       use_lm: bool = False, debug: bool = False) -> List[Tuple[str, float]]:
        """
        Beam search with Wav2Vec2 support.
        """
        beam_size = self.beam_size
        debug = False

        if use_lm and self.decoder is not None:
            try:
                if isinstance(probs, torch.Tensor):
                    probs = probs.cpu().numpy()
                elif isinstance(probs, list):
                    probs = np.array(probs)
                elif isinstance(probs, np.ndarray):
                    pass
                else:
                    raise TypeError("probs must be a torch.Tensor, list, or numpy.ndarray")

                beams = self.decoder.decode_beams(
                    probs,
                    beam_prune_logp=-10.0,
                    token_min_logp=-5.0,
                    hotwords=[],
                    hotword_weight=10.0,
                )

                formatted_beams = []
                for beam in beams[:beam_size]:
                    text = beam[0]
                    acoustic_score = beam[3]
                    lm_score = beam[4]

                    text = self.processor.tokenizer.clean_up_tokenization(text)
                    text = text.lower().strip()

                    combined_score = (1 - self.lm_weight) * acoustic_score + self.lm_weight * lm_score
                    text_len = max(1, len(text.split()))
                    normalized_score = combined_score / text_len

                    formatted_beams.append((text, normalized_score))

                if debug:
                    print("\nFormatted beam results with Wav2Vec2:")
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

        if isinstance(probs, np.ndarray):
            probs = torch.from_numpy(probs)

        if probs.device != torch.device('cpu'):
            probs = probs.cpu()

        dp = {("", self.blank_token): 0.0}
        log_probs = torch.log(probs + 1e-8)

        if debug:
            print("\nStarting beam search with beam size:", beam_size)

        for t, prob in enumerate(log_probs):
            new_dp = defaultdict(lambda: float('-inf'))
            top_k = torch.topk(prob, k=min(beam_size, len(prob)))

            if debug and t < self.max_printed_samples:
                print(f"\nTimestep {t}:")
                print("Top tokens:", [(self.ind2char[idx.item()], val.item()) 
                                    for val, idx in zip(top_k.values, top_k.indices)])

            for val, ind in zip(top_k.values, top_k.indices):
                curr_char = self.ind2char[ind.item()]
                next_token_log_prob = val.item()

                for (prefix, last_char), log_prob in dp.items():
                    if last_char == curr_char and curr_char != " ":
                        new_prefix = prefix
                    else:
                        if curr_char != self.blank_token:
                            if curr_char == " " and prefix.endswith(" "):
                                continue
                            new_prefix = prefix + curr_char
                        else:
                            new_prefix = prefix

                    new_log_prob = log_prob + next_token_log_prob
                    key = (new_prefix, curr_char)
                    new_dp[key] = max(new_dp[key], new_log_prob)

            if len(new_dp) > 0:
                max_score = max(score for _, score in new_dp.items())
                new_dp = {key: score - max_score for key, score in new_dp.items()}

            dp = dict(sorted(new_dp.items(), key=lambda x: -x[1])[:beam_size])

            if debug and t < 2:
                print("\nCurrent beam:")
                for (text, last_char), score in list(dp.items())[:3]:
                    print(f"Text: '{text}', Last: '{last_char}', Score: {score:.4f}")

        final_beams = []
        for (text, _), score in dp.items():
            text = self.processor.tokenizer.clean_up_tokenization(text)
            text = text.lower().strip()
            text_len = max(1, len(text.split()))
            normalized_score = score / text_len
            final_beams.append((text, normalized_score))

        final_beams.sort(key=lambda x: -x[1])
        if not final_beams:
            final_beams = [("", float('-inf'))]

        return final_beams[:beam_size]

    def test_language_model(self):
        """Debug function to verify LM functionality"""
        print("\nTesting Language Model...")

        if self.lm is None:
            print("Error: Language model is not loaded!")
            return

        test_sentences = [
            "this is a good sentence",
            "this is also a good sentence",
            "thiss iss nott aa goodd sentencee",
            "random word salad box cat",
            "the cat sat on the mat",
            "",
            "a",
        ]

        print("\nTesting individual sentences:")
        for sentence in test_sentences:
            score = self.score_with_lm(sentence)
            print(f"\nText: '{sentence}'")
            print(f"LM Score: {score:.4f}")

        test_prefixes = [
            "the quick brown",
            "how are",
            "thank",
            "nice to",
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
        Score text using language model, handling edge cases
        """
        if self.lm is None:
            return 0.0

        if not text or len(text.strip()) == 0:
            return float('-inf')

        text = text.lower().strip()
        return self.lm.score(text, bos=True, eos=True)

    def _basic_ctc_decode(self, logits: np.ndarray, sequence_length: int) -> List[str]:
        """Basic CTC decoding without LM"""
        argmax_indices = np.argmax(logits, axis=-1)

        if len(argmax_indices.shape) == 0:
            argmax_indices = np.array([argmax_indices])

        if len(argmax_indices.shape) == 1:
            argmax_indices = np.expand_dims(argmax_indices, axis=0)

        predictions = []
        for sequence in argmax_indices:
            decoded = []
            last_idx = None

            for idx in sequence[:sequence_length]:
                if idx != self.blank_index and idx != last_idx:
                    decoded.append(self.ind2char[idx])
                last_idx = idx

            text = "".join(decoded)
            if hasattr(self, 'processor'):
                text = self.processor.tokenizer.clean_up_tokenization(text)
            predictions.append(text)

        return predictions

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize input text"""
        # text = text.lower()
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        # text = re.sub(r"[^A-Z ]", "", text)
        return text


    def test_decoder(self, sample_text: str = "test decoder functionality"):
        """Test the decoder setup"""
        print("\nTesting decoder configuration...")

        encoded = self.encode(sample_text)
        decoded = self.decode(encoded[0].tolist())
        print(f"Original text: {sample_text}")
        print(f"Basic decode: {decoded}")

        sequence_length = 50
        vocab_size = len(self)
        fake_logits = torch.randn(1, sequence_length, vocab_size)
        fake_length = torch.tensor([sequence_length])

        if self.decoder is not None:
            print("\nTesting pyctcdecode integration...")
            decoded_with_lm = self.ctc_decode(fake_logits)
            print(f"Decoded with LM: {decoded_with_lm}")

            print(f"\nBeam width: {self.beam_size}")
            print(f"LM weight: {self.lm_weight}")
        else:
            print("\nNo language model loaded - using basic CTC decoding")
            basic_decoded = self._basic_ctc_decode(fake_logits.numpy(), fake_length)
            print(f"Basic CTC decoded: {basic_decoded[0]}")

    def __len__(self):
        return len(self.vocab)
