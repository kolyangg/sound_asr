import re
from collections import defaultdict
import torch
import kenlm
from pyctcdecode import build_ctcdecoder
import numpy as np
import os
from typing import List, Tuple, Optional, Union

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CTCTextEncoder:
    def __init__(
        self,
        arpa_path: Optional[str] = None,
        binary_path: Optional[str] = "4-gram_lc_correct.bin",
        unigram_path: Optional[str] = "librispeech-vocab.txt",
        pretrained_tokenizer: str = "sentencepiece_model/librispeech_unigram_model.model",
        lm_weight: float = 0.5,
        beam_size: int = 100,
        use_lm: bool = False,
        use_bpe: bool = False,
        blank_token: str = "<pad>",
        unk_token: str = "<unk>",
        **kwargs
    ):
        """
        Initialize encoder with conditional subword (SentencePiece) or char-based vocab.
        """

        # Debug switch
        self.debug = False  # Set to True if you want verbose prints

        if self.debug:
            print("[DEBUG] Initializing CTCTextEncoder...")
            print(f"  -> arpa_path={arpa_path}, binary_path={binary_path}, unigram_path={unigram_path}")
            print(f"  -> pretrained_tokenizer={pretrained_tokenizer}")
            print(f"  -> lm_weight={lm_weight}, beam_size={beam_size}")
            print(f"  -> use_lm={use_lm}, use_bpe={use_bpe}")
            print(f"  -> blank_token={blank_token}, unk_token={unk_token}")

        self.beam_size = beam_size
        self.lm_weight = lm_weight
        self.arpa_path = arpa_path
        self.binary_path = binary_path
        self.blank_token = blank_token
        self.unk_token = unk_token
        self.use_lm = use_lm
        self.use_bpe = use_bpe

        # Load unigrams if provided
        self.unigrams = None
        if unigram_path and os.path.exists(unigram_path):
            if self.debug:
                print(f"[DEBUG] Loading unigrams from {unigram_path}")
            with open(unigram_path, 'r', encoding='utf-8') as f:
                self.unigrams = [line.strip().lower() for line in f if line.strip()]
            if self.debug:
                print(f"[DEBUG] Loaded {len(self.unigrams)} unigrams")

        # Initialize vocabulary/tokenizer
        self._initialize_vocabulary(pretrained_tokenizer)

        # Create index mappings
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.blank_index = self.char2ind.get(self.blank_token, None)

        if self.debug:
            print("\n[DEBUG] Vocabulary Info:")
            print(f"  -> vocab size: {len(self.vocab)}")
            print(f"  -> blank_token='{self.blank_token}' => index={self.blank_index}")
            print("  -> sample vocab:", self.vocab[:30])

            sample_inds = list(self.ind2char.keys())[:10]
            print("  -> Sample ind2char:", {k: self.ind2char[k] for k in sample_inds})

            sample_chars = list(self.char2ind.keys())[:10]
            print("  -> Sample char2ind:", {k: self.char2ind[k] for k in sample_chars})

        # Optionally load LM
        if self.use_lm:
            self._initialize_language_model()
        else:
            if self.debug:
                print("[DEBUG] No LM usage => self.decoder=None")
            self.lm = None
            self.decoder = None

        if self.debug:
            print("[DEBUG] CTCTextEncoder initialized.\n")

    def _initialize_vocabulary(self, pretrained_tokenizer: str):
        """
        If use_bpe=True, load a SentencePiece model from `pretrained_tokenizer` path.
        Otherwise, use a basic character-based vocab.
        """
        if self.use_bpe:
            if self.debug:
                print("[DEBUG] use_bpe=True => Loading SentencePiece model.")
            import sentencepiece as spm

            if not os.path.exists(pretrained_tokenizer):
                raise FileNotFoundError(f"SentencePiece model not found at: {pretrained_tokenizer}")

            self.sp = spm.SentencePieceProcessor()
            self.sp.load(pretrained_tokenizer)

            vocab_size = self.sp.get_piece_size()
            if self.debug:
                print(f"[DEBUG] Loaded SP model => vocab_size={vocab_size}")
                print(f"[DEBUG] sp.IdToPiece(0)={self.sp.id_to_piece(0)} (often <unk>)")

            self.labels = [self.sp.id_to_piece(i) for i in range(vocab_size)]
            self.vocab = self.labels  # len(self.vocab) = vocab_size
        else:
            # Character-based => do NOT TOUCH
            if self.debug:
                print("[DEBUG] Initializing character-based vocabulary without using tokenizer.")
            self.vocab = [
                'a','b','c','d','e','f','g','h','i','j',
                'k','l','m','n','o','p','q','r','s','t',
                'u','v','w','x','y','z',' '
            ]
            self.vocab += [self.blank_token, self.unk_token]
            self.labels = self.vocab
            self.sp = None

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text. If bpe => subword, else => char-based.
        """
        if self.use_bpe and hasattr(self, 'sp') and self.sp is not None:
            normalized_text = self.normalize_text(text)
            token_ids = self.sp.encode(text, out_type=int)
            if self.debug:
                print(f"[DEBUG-bpe] encode => text='{text}' => token_ids={token_ids}")
            unknown_count = sum(1 for tid in token_ids if self.sp.id_to_piece(tid) == "<unk>")
            if unknown_count > 0 and self.debug:
                print(f"[DEBUG-bpe] <unk> count={unknown_count} => coverage issues possible.")
            return torch.tensor([token_ids], dtype=torch.long)
        else:
            # char-based => do not touch
            normalized_text = self.normalize_text(text)
            token_indices = [self.char2ind.get(char, self.char2ind.get(self.unk_token))
                             for char in normalized_text]
            return torch.tensor(token_indices).unsqueeze(0)

    def decode_simple(self, indices: List[int]) -> str:
        """
        Simple CTC decode => collapse repeats, remove blank.
        If bpe => subword decode after collapsing, else => char-based.
        """
        if self.use_bpe and hasattr(self, 'sp') and self.sp is not None:
            collapsed = []
            prev_idx = None
            for idx in indices:
                if idx == self.blank_index:
                    prev_idx = idx
                    continue
                if idx != prev_idx:
                    collapsed.append(idx)
                prev_idx = idx

            if self.debug:
                print(f"[DEBUG-bpe] decode_simple => collapsed={collapsed}")

            text = self.sp.decode(collapsed)

            if self.debug:
                print(f"[DEBUG-bpe] decode_simple => text='{text}' before removing placeholders")

            # Convert "▁" => space
            text = text.replace("▁", " ")

            # Remove any leftover placeholders: "⁇", "??", " ⁇" etc
            placeholders = ["⁇", "??", " ⁇"]
            for p in placeholders:
                if p in text:
                    if self.debug:
                        print(f"[DEBUG-bpe] Removing placeholder '{p}' from text => '{text}'")
                    text = text.replace(p, "")

            # Merge repeated spaces
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        else:
            # char-based => do not touch
            decoded_chars = []
            prev_idx = None
            for idx in indices:
                if idx == self.blank_index:
                    prev_idx = idx
                    continue
                if idx == prev_idx:
                    continue
                if 0 <= idx < len(self.ind2char):
                    decoded_chars.append(self.ind2char[idx])
                prev_idx = idx
            text = "".join(decoded_chars).strip().lower()
            return text

    def decode(self, indices: List[int]) -> str:
        """
        If LM => decoder.decode(...), else => decode_simple
        """
        if self.decoder:
            text = self.decoder.decode(indices).lower()
            # Convert "▁" => space
            text = text.replace("▁", " ")
            # Remove placeholders
            placeholders = ["⁇", "??", " ⁇"]
            for p in placeholders:
                if p in text:
                    if self.debug:
                        print(f"[DEBUG-lm] Removing placeholder '{p}' from text => '{text}'")
                    text = text.replace(p, "")
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        else:
            return self.decode_simple(indices).lower()

    def decode_logits(self, logits: Union[torch.Tensor, List[List[float]], np.ndarray]) -> str:
        """
        If LM => decoder.decode(logits), else => greedy => decode_simple
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        elif isinstance(logits, list):
            logits = np.array(logits)

        if logits.ndim == 3:
            logits = logits[0]
        if logits.ndim != 2:
            raise ValueError(f"Logits should be 2D, got shape {logits.shape}")

        if self.debug:
            print(f"[DEBUG] decode_logits => shape={logits.shape}, use_lm={self.use_lm}, use_bpe={self.use_bpe}, lm_weight={self.lm_weight}")

        if self.decoder:
            text = self.decoder.decode(logits).lower()
            # Replace "▁" => space
            text = text.replace("▁", " ")
            # Remove placeholders
            placeholders = ["⁇", "??", " ⁇"]
            for p in placeholders:
                if p in text:
                    if self.debug:
                        print(f"[DEBUG-lm] decode_logits => removing placeholder '{p}' => '{text}'")
                    text = text.replace(p, "")
            # Merge repeated spaces
            text = re.sub(r'\s+', ' ', text).strip()
            if self.debug:
                print(f"[DEBUG-lm] decode_logits => partial='{text[:60]}...'")
            return text
        else:
            predicted_indices = np.argmax(logits, axis=-1).tolist()
            return self.decode_simple(predicted_indices)

    def decode_indices(self, indices: Union[torch.Tensor, List[int], np.ndarray]) -> str:
        """
        Direct decode => no LM
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.squeeze().tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()
        return self.decode_simple(indices)

    def ctc_decode(self, logits: Union[torch.Tensor, List[int], np.ndarray]) -> str:
        """
        CTC decode from logits or token indices.
        If bpe => subword approach, else => char-based approach
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        elif isinstance(logits, list):
            logits = torch.tensor(logits)

        if self.debug:
            print(f"[DEBUG] ctc_decode => shape={logits.shape}, use_bpe={self.use_bpe}")

        if self.use_bpe:
            # subword path
            if logits.dim() == 3:
                return self.decode_logits(logits)
            elif logits.dim() == 2:
                return self.decode_logits(logits)
            elif logits.dim() == 1:
                return self.decode_indices(logits)
            else:
                raise ValueError(f"Unsupported logits shape: {logits.shape}.")
        else:
            # char-based => do NOT TOUCH
            if logits.dim() == 3:
                return self.decode_logits(logits)
            elif logits.dim() == 2:
                return self.decode_logits(logits)
            elif logits.dim() == 1:
                return self.decode_indices(logits)
            else:
                raise ValueError(f"Unsupported logits shape: {logits.shape}.")

    def _initialize_language_model(self):
        if self.debug:
            print("[DEBUG] _initialize_language_model => start")

        self.lm = None
        self.decoder = None

        model_path = self.binary_path if self.binary_path else self.arpa_path
        if self.debug:
            print(f"[DEBUG] model_path => {model_path}")
        if not model_path or not os.path.exists(model_path):
            if self.debug:
                print("[DEBUG] No valid LM path => skipping LM init.")
            return

        try:
            self.lm = kenlm.Model(model_path)
            if self.debug:
                print(f"[DEBUG] KenLM model loaded from: {model_path}")
                print(f"[DEBUG-lm] Building decoder_config with alpha={self.lm_weight}, beta=0.1")

            decoder_config = {
                "labels": self.labels if self.use_bpe else self.vocab,
                "kenlm_model_path": model_path,
                "alpha": self.lm_weight,
                "beta": 0.1,
                "unk_score_offset": -10.0,
            }

            if self.unigrams:
                if self.debug:
                    print("[DEBUG] Found unigrams => adding them to decoder_config.")
                with open("unigrams_list.txt", "w", encoding="utf-8") as f:
                    for unigram in self.unigrams:
                        f.write(f"{unigram}\n")
                decoder_config["unigrams"] = self.unigrams

            self.decoder = build_ctcdecoder(**decoder_config)
            if self.debug:
                print("[DEBUG] LM-based decoder successfully initialized.")
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] LM init failed => {str(e)}")
            self.decoder = None

    def test_language_model(self):
        if self.debug:
            print("[DEBUG] test_language_model =>")

        if self.lm is None:
            if self.debug:
                print("[DEBUG] No LM loaded.")
            return

        sample_sents = ["this is a test", "hello world", "aaaa bbbb cccc dddd"]
        for s in sample_sents:
            score = self.lm.score(s, bos=True, eos=True)
            if self.debug:
                print(f"[DEBUG-lm] LM Score for '{s}' => {score:.4f}")

    def score_with_lm(self, text: str) -> float:
        if self.lm is None:
            return 0.0
        return self.lm.score(text.lower().strip(), bos=True, eos=True)

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Basic normalization => lower + remove non-alpha + space
        """
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def _basic_ctc_decode(self, logits: np.ndarray, sequence_length: int) -> List[str]:
        if self.debug:
            print("[DEBUG] _basic_ctc_decode => start")

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
            if self.use_bpe and hasattr(self, 'sp') and self.sp:
                # Could decode with sp if you want
                pass
            predictions.append(text)
        return predictions

    def test_decoder(self, sample_text: str = "test decoder functionality"):
        if self.debug:
            print(f"[DEBUG] test_decoder => '{sample_text}'")

        encoded = self.encode(sample_text)
        decoded = self.decode(encoded[0].tolist())
        if self.debug:
            print(f"[DEBUG] Original: '{sample_text}' => Decoded: '{decoded}'")

        seq_length = 50
        vocab_size = len(self)
        fake_logits = torch.randn(1, seq_length, vocab_size)

        if self.decoder is not None:
            if self.debug:
                print("[DEBUG] Testing pyctcdecode on fake logits => ctc_decode()")
            out = self.ctc_decode(fake_logits)
            if self.debug:
                print(f"[DEBUG] Decoded with LM => '{out}'")
        else:
            if self.debug:
                print("[DEBUG] No LM => calling _basic_ctc_decode (char-based) on fake logits")
            basic_dec = self._basic_ctc_decode(fake_logits.numpy(), seq_length)
            if self.debug:
                print(f"[DEBUG] Basic ctc decoded => '{basic_dec[0]}'")

    def __len__(self):
        return len(self.vocab)

    def ctc_beam_search(self, probs, beam_size=None, use_lm=False, debug=False) -> List[Tuple[str, float]]:
        """
        Beam search with optional LM. If LM => uses pyctcdecode, else => naive beam.
        """
        beam_size = beam_size or self.beam_size
        if self.use_lm and self.decoder is not None:
            try:
                if isinstance(probs, torch.Tensor):
                    probs = probs.cpu().numpy()
                elif isinstance(probs, list):
                    probs = np.array(probs)

                if self.debug:
                    print(f"[DEBUG] ctc_beam_search => shape={probs.shape}, use_bpe={self.use_bpe}, lm_weight={self.lm_weight}")
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

                    # If BPE => remove "▁"
                    if self.use_bpe and hasattr(self, 'sp') and self.sp is not None:
                        text = text.replace("▁", " ")

                    # Remove leftover placeholders
                    placeholders = ["⁇", "??", " ⁇"]
                    for p in placeholders:
                        if p in text:
                            if self.debug:
                                print(f"[DEBUG-lm] Removing placeholder '{p}' => text='{text}'")
                            text = text.replace(p, "")

                    # Merge repeated spaces
                    text = re.sub(r'\s+', ' ', text).strip()

                    combined_score = (1 - self.lm_weight)*acoustic_score + self.lm_weight*lm_score
                    text_len = max(1, len(text.split())) if self.use_bpe else max(1, len(text))
                    norm_score = combined_score / text_len
                    formatted_beams.append((text.lower(), norm_score))

                if not formatted_beams:
                    if self.debug:
                        print("[DEBUG] No valid beams => fallback to standard beam search")
                    return self._standard_beam_search(probs, debug)

                # Sort by descending normalized score
                return sorted(formatted_beams, key=lambda x: -x[1])

            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] LM decode_beams failed => {e}, fallback to standard beam search")
                return self._standard_beam_search(probs, debug)
        else:
            return self._standard_beam_search(probs, debug)

    def _standard_beam_search(self, probs, debug=False) -> List[Tuple[str, float]]:
        """
        Naive beam search w/o LM => prefix expansions.
        """
        beam_size = self.beam_size
        if isinstance(probs, np.ndarray):
            probs = torch.from_numpy(probs)
        if probs.device != torch.device('cpu'):
            probs = probs.cpu()

        dp = {("", self.blank_token): 0.0}
        log_probs = torch.log(probs + 1e-8)

        for t, prob in enumerate(log_probs):
            new_dp = defaultdict(lambda: float('-inf'))
            top_k = torch.topk(prob, k=min(beam_size, len(prob)))

            if self.debug and t < 3:
                print(f"[DEBUG] _standard_beam_search => t={t}, top_k.indices={top_k.indices.tolist()}")

            for val, ind in zip(top_k.values, top_k.indices):
                curr_char = self.ind2char[ind.item()]
                next_lp = val.item()

                for (prefix, last_char), old_lp in dp.items():
                    if last_char == curr_char and curr_char != " ":
                        new_prefix = prefix
                    else:
                        if curr_char != self.blank_token:
                            if curr_char == " " and prefix.endswith(" "):
                                continue
                            new_prefix = prefix + curr_char
                        else:
                            new_prefix = prefix

                    new_score = old_lp + next_lp
                    key = (new_prefix, curr_char)
                    # Keep only max
                    new_dp[key] = max(new_dp[key], new_score)

            if len(new_dp) > 0:
                max_score = max(new_dp.values())
                new_dp = {k: v - max_score for k, v in new_dp.items()}

            dp = dict(sorted(new_dp.items(), key=lambda x: -x[1])[:beam_size])

        final_beams = []
        for (text, _), score in dp.items():
            # If BPE => remove placeholders
            if self.use_bpe and hasattr(self, 'sp') and self.sp is not None:
                text = text.replace("▁", " ")
            placeholders = ["⁇", "??", " ⁇"]
            for p in placeholders:
                if p in text:
                    if self.debug:
                        print(f"[DEBUG] _standard_beam_search => removing '{p}' => text='{text}'")
                    text = text.replace(p, "")
            text = text.lower().strip()
            text_len = max(1, len(text.split())) if self.use_bpe else max(1, len(text))
            norm_score = score / text_len
            final_beams.append((text, norm_score))

        # Sort by descending normalized score
        final_beams.sort(key=lambda x: -x[1])
        if not final_beams:
            final_beams = [("", float('-inf'))]
        return final_beams[:beam_size]





# import re
# from collections import defaultdict
# import torch
# import kenlm
# from pyctcdecode import build_ctcdecoder
# import numpy as np
# import os
# from typing import List, Tuple, Optional, Union

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# class CTCTextEncoder:
#     def __init__(
#         self,
#         arpa_path: Optional[str] = None,
#         binary_path: Optional[str] = "4-gram_lc_correct.bin",
#         unigram_path: Optional[str] = "librispeech-vocab.txt",
#         pretrained_tokenizer: str = "sentencepiece_model/librispeech_unigram_model.model",
#         lm_weight: float = 0.5,
#         beam_size: int = 100,
#         use_lm: bool = False,
#         use_bpe: bool = False,
#         blank_token: str = "<pad>",
#         unk_token: str = "<unk>",
#         **kwargs
#     ):
#         """
#         Initialize encoder with conditional subword (SentencePiece) or char-based vocab.
#         """

#         # Debug switch
#         self.debug = False  # Set True for verbose prints

#         if self.debug:
#             print("[DEBUG] Initializing CTCTextEncoder...")
#             print(f"  -> arpa_path={arpa_path}, binary_path={binary_path}, unigram_path={unigram_path}")
#             print(f"  -> pretrained_tokenizer={pretrained_tokenizer}")
#             print(f"  -> lm_weight={lm_weight}, beam_size={beam_size}")
#             print(f"  -> use_lm={use_lm}, use_bpe={use_bpe}")
#             print(f"  -> blank_token={blank_token}, unk_token={unk_token}")

#         self.beam_size = beam_size
#         self.lm_weight = lm_weight
#         self.arpa_path = arpa_path
#         self.binary_path = binary_path
#         self.blank_token = blank_token
#         self.unk_token = unk_token
#         self.use_lm = use_lm
#         self.use_bpe = use_bpe

#         # Load unigrams if provided
#         self.unigrams = None
#         if unigram_path and os.path.exists(unigram_path):
#             if self.debug:
#                 print(f"[DEBUG] Loading unigrams from {unigram_path}")
#             with open(unigram_path, 'r', encoding='utf-8') as f:
#                 self.unigrams = [line.strip().lower() for line in f if line.strip()]
#             if self.debug:
#                 print(f"[DEBUG] Loaded {len(self.unigrams)} unigrams")

#         # Initialize vocabulary/tokenizer
#         self._initialize_vocabulary(pretrained_tokenizer)

#         # Create index mappings
#         self.ind2char = dict(enumerate(self.vocab))
#         self.char2ind = {v: k for k, v in self.ind2char.items()}
#         self.blank_index = self.char2ind.get(self.blank_token, None)

#         if self.debug:
#             print("\n[DEBUG] Vocabulary Info:")
#             print(f"  -> vocab size: {len(self.vocab)}")
#             print(f"  -> blank_token='{self.blank_token}' => index={self.blank_index}")
#             print("  -> sample vocab:", self.vocab[:30])

#             sample_inds = list(self.ind2char.keys())[:10]
#             print("  -> Sample ind2char:", {k: self.ind2char[k] for k in sample_inds})

#             sample_chars = list(self.char2ind.keys())[:10]
#             print("  -> Sample char2ind:", {k: self.char2ind[k] for k in sample_chars})

#         # Optionally load LM
#         if self.use_lm:
#             self._initialize_language_model()
#             self.test_language_model()
#         else:
#             if self.debug:
#                 print("[DEBUG] No LM usage. self.decoder=None")
#             self.lm = None
#             self.decoder = None

#         if self.debug:
#             print("[DEBUG] CTCTextEncoder initialized.\n")

#     def _initialize_vocabulary(self, pretrained_tokenizer: str):
#         """
#         If use_bpe=True, load a SentencePiece model from `pretrained_tokenizer` path.
#         Otherwise, use a basic character-based vocab.
#         """
#         if self.use_bpe:
#             if self.debug:
#                 print("[DEBUG] use_bpe=True => Loading SentencePiece model.")
#             import sentencepiece as spm

#             if not os.path.exists(pretrained_tokenizer):
#                 raise FileNotFoundError(
#                     f"SentencePiece model not found at: {pretrained_tokenizer}"
#                 )

#             self.sp = spm.SentencePieceProcessor()
#             self.sp.load(pretrained_tokenizer)

#             vocab_size = self.sp.get_piece_size()
#             if self.debug:
#                 print(f"[DEBUG] Loaded SP model => vocab_size={vocab_size}")
#                 print(f"[DEBUG] sp.IdToPiece(0)={self.sp.id_to_piece(0)} (often <unk>)")

#             self.labels = [self.sp.id_to_piece(i) for i in range(vocab_size)]
#             self.vocab = self.labels  # len(self.vocab) = vocab_size
#         else:
#             # char-based => do not touch
#             if self.debug:
#                 print("[DEBUG] Initializing character-based vocabulary without using tokenizer.")
#             self.vocab = [
#                 'a','b','c','d','e','f','g','h','i','j',
#                 'k','l','m','n','o','p','q','r','s','t',
#                 'u','v','w','x','y','z',' '
#             ]
#             self.vocab += [self.blank_token, self.unk_token]
#             self.labels = self.vocab
#             self.sp = None

#     def encode(self, text: str) -> torch.Tensor:
#         """
#         Encode text. If bpe => subword, else => char-based.
#         """
#         if self.use_bpe and hasattr(self, 'sp') and self.sp is not None:
#             token_ids = self.sp.encode(text, out_type=int)
#             if self.debug:
#                 print(f"[DEBUG-bpe] encode => text='{text}' => token_ids={token_ids}")
#             unknown_count = sum(1 for tid in token_ids if self.sp.id_to_piece(tid) == "<unk>")
#             if unknown_count > 0 and self.debug:
#                 print(f"[DEBUG-bpe] <unk> count={unknown_count}, might indicate coverage issues.")
#             return torch.tensor([token_ids], dtype=torch.long)
#         else:
#             # char-based => do not touch
#             normalized_text = self.normalize_text(text)
#             token_indices = [
#                 self.char2ind.get(char, self.char2ind.get(self.unk_token))
#                 for char in normalized_text
#             ]
#             return torch.tensor(token_indices).unsqueeze(0)

#     def decode_simple(self, indices: List[int]) -> str:
#         """
#         Simple CTC decode => collapses repeats, removes blank.
#         If bpe => subword decode after collapsing, else => char-based.
#         """
#         if self.use_bpe and hasattr(self, 'sp') and self.sp is not None:
#             collapsed = []
#             prev_idx = None
#             for idx in indices:
#                 if idx == self.blank_index:
#                     prev_idx = idx
#                     continue
#                 if idx != prev_idx:
#                     collapsed.append(idx)
#                 prev_idx = idx

#             if self.debug:
#                 print(f"[DEBUG-bpe] decode_simple => collapsed={collapsed}")

#             text = self.sp.decode(collapsed)

#             if self.debug:
#                 print(f"[DEBUG-bpe] decode_simple => text='{text}' before filtering placeholders")

#             # Convert "▁" => space
#             text = text.replace("▁", " ")

#             # Remove "⁇" or "??" placeholders
#             # e.g. if "??" is leftover from unknown tokens
#             if "⁇" in text:
#                 if self.debug:
#                     print(f"[DEBUG-bpe] Found '⁇' => removing them from text.")
#                 text = text.replace("⁇", "")
#             # If your placeholders are "??"
#             if "??" in text:
#                 if self.debug:
#                     print(f"[DEBUG-bpe] Found '??' => removing them from text.")
#                 text = text.replace("??", "")

#             # Merge repeated spaces
#             text = re.sub(r'\s+', ' ', text).strip()

#             return text
#         else:
#             # char-based => do not touch
#             decoded_chars = []
#             prev_idx = None
#             for idx in indices:
#                 if idx == self.blank_index:
#                     prev_idx = idx
#                     continue
#                 if idx == prev_idx:
#                     continue
#                 if 0 <= idx < len(self.ind2char):
#                     decoded_chars.append(self.ind2char[idx])
#                 prev_idx = idx
#             text = "".join(decoded_chars).strip().lower()
#             return text

#     def decode(self, indices: List[int]) -> str:
#         """
#         If LM => self.decoder.decode(...), else => decode_simple
#         """
#         if self.decoder:
#             decoded_text = self.decoder.decode(indices)
#             decoded_text = decoded_text.lower().replace("▁", " ")
#             # Remove "??" placeholders
#             if "??" in decoded_text:
#                 if self.debug:
#                     print(f"[DEBUG-lm] Found '??' in LM decode => removing them.")
#                 decoded_text = decoded_text.replace("??", "")
#             # Merge spaces
#             decoded_text = re.sub(r'\s+', ' ', decoded_text).strip()
#             return decoded_text
#         else:
#             return self.decode_simple(indices).lower()

#     def decode_logits(self, logits: Union[torch.Tensor, List[List[float]], np.ndarray]) -> str:
#         """
#         If LM => self.decoder.decode(logits), else => greedy => decode_simple
#         """
#         if isinstance(logits, torch.Tensor):
#             logits = logits.cpu().numpy()
#         elif isinstance(logits, list):
#             logits = np.array(logits)

#         if logits.ndim == 3:
#             logits = logits[0]
#         if logits.ndim != 2:
#             raise ValueError(f"Logits should be 2D, got shape {logits.shape}")

#         if self.debug:
#             print(f"[DEBUG] decode_logits => shape={logits.shape}, use_lm={self.use_lm}, use_bpe={self.use_bpe}")

#         if self.decoder:
#             text = self.decoder.decode(logits)
#             text = text.lower().replace("▁", " ")
#             # Remove "??"
#             if "??" in text:
#                 if self.debug:
#                     print(f"[DEBUG-lm] decode_logits => found '??', removing them.")
#                 text = text.replace("??", "")
#             text = re.sub(r'\s+', ' ', text).strip()
#             if self.debug:
#                 print(f"[DEBUG-lm] decode_logits => partial='{text[:50]}...'")
#             return text
#         else:
#             predicted_indices = np.argmax(logits, axis=-1).tolist()
#             return self.decode_simple(predicted_indices)

#     def decode_indices(self, indices: Union[torch.Tensor, List[int], np.ndarray]) -> str:
#         """
#         Direct decode => no LM
#         """
#         if isinstance(indices, torch.Tensor):
#             indices = indices.squeeze().tolist()
#         elif isinstance(indices, np.ndarray):
#             indices = indices.tolist()
#         return self.decode_simple(indices)

#     def ctc_decode(self, logits: Union[torch.Tensor, List[int], np.ndarray]) -> str:
#         """
#         CTC decode from logits or token indices.
#         If bpe => subword approach, else => char-based approach
#         """
#         if isinstance(logits, np.ndarray):
#             logits = torch.from_numpy(logits)
#         elif isinstance(logits, list):
#             logits = torch.tensor(logits)

#         if self.debug:
#             print(f"[DEBUG] ctc_decode => shape={logits.shape}, use_bpe={self.use_bpe}")

#         if self.use_bpe:
#             # subword path
#             if logits.dim() == 3:
#                 return self.decode_logits(logits)
#             elif logits.dim() == 2:
#                 return self.decode_logits(logits)
#             elif logits.dim() == 1:
#                 return self.decode_indices(logits)
#             else:
#                 raise ValueError(f"Unsupported logits shape: {logits.shape}.")
#         else:
#             # char-based => do NOT TOUCH
#             if logits.dim() == 3:
#                 return self.decode_logits(logits)
#             elif logits.dim() == 2:
#                 return self.decode_logits(logits)
#             elif logits.dim() == 1:
#                 return self.decode_indices(logits)
#             else:
#                 raise ValueError(f"Unsupported logits shape: {logits.shape}.")

#     def _initialize_language_model(self):
#         if self.debug:
#             print("[DEBUG] _initialize_language_model => start")
#         self.lm = None
#         self.decoder = None

#         model_path = self.binary_path if self.binary_path else self.arpa_path
#         if self.debug:
#             print(f"[DEBUG] model_path => {model_path}")
#         if not model_path or not os.path.exists(model_path):
#             if self.debug:
#                 print("[DEBUG] No valid LM path => skipping LM init.")
#             return

#         try:
#             self.lm = kenlm.Model(model_path)
#             if self.debug:
#                 print("[DEBUG] KenLM model loaded from:", model_path)

#             decoder_config = {
#                 "labels": self.labels if self.use_bpe else self.vocab,
#                 "kenlm_model_path": model_path,
#                 "alpha": self.lm_weight,
#                 "beta": 0.1,
#                 "unk_score_offset": -10.0,
#             }

#             if self.unigrams:
#                 if self.debug:
#                     print("[DEBUG] Found unigrams => adding to decoder_config.")
#                 with open("unigrams_list.txt", "w", encoding="utf-8") as f:
#                     for unigram in self.unigrams:
#                         f.write(f"{unigram}\n")
#                 decoder_config["unigrams"] = self.unigrams

#             self.decoder = build_ctcdecoder(**decoder_config)
#             if self.debug:
#                 print("[DEBUG] LM-based decoder successfully initialized.")
#         except Exception as e:
#             if self.debug:
#                 print(f"[DEBUG] LM init failed => {str(e)}")
#             self.decoder = None

#     # def test_language_model(self):
#     #     if self.debug:
#     #         print("[DEBUG] test_language_model =>")
#     #     if self.lm is None:
#     #         if self.debug:
#     #             print("[DEBUG] No LM loaded.")
#     #         return

#     #     sample_sents = ["this is a test", "hello world", "aaaa bbbb cccc dddd"]
#     #     for s in sample_sents:
#     #         score = self.lm.score(s, bos=True, eos=True)
#     #         if self.debug:
#     #             print(f"LM Score for '{s}' => {score:.4f}")

#     def test_language_model(self):
#         """Debug function to verify LM functionality"""
#         print("\nTesting Language Model...")

#         if self.lm is None:
#             print("Error: Language model is not loaded!")
#             return

#         test_sentences = [
#             "this is a good sentence",
#             "this is also a good sentence",
#             "thiss iss nott aa goodd sentencee",
#             "random word salad box cat",
#             "the cat sat on the mat",
#             "",
#             "a",
#         ]

#         print("\nTesting individual sentences:")
#         for sentence in test_sentences:
#             score = self.score_with_lm(sentence)
#             print(f"\nText: '{sentence}'")
#             print(f"LM Score: {score:.4f}")

#         test_prefixes = [
#             "the quick brown",
#             "how are",
#             "thank",
#             "nice to",
#         ]

#         print("\nTesting word completions:")
#         for prefix in test_prefixes:
#             print(f"\nPrefix: '{prefix}'")
#             completions = [
#                 prefix + " " + word for word in ["you", "fox", "cat", "xyz", "meet"]
#             ]
#             scores = [(completion, self.score_with_lm(completion)) 
#                     for completion in completions]
#             scores.sort(key=lambda x: x[1], reverse=True)
#             print("Top completions by score:")
#             for completion, score in scores[:3]:
#                 print(f"  '{completion}': {score:.4f}")
    
    
#     def score_with_lm(self, text: str) -> float:
#         if self.lm is None:
#             return 0.0
#         return self.lm.score(text.lower().strip(), bos=True, eos=True)

#     @staticmethod
#     def normalize_text(text: str) -> str:
#         """
#         Basic normalization => lower + remove non-alpha + space
#         """
#         text = text.lower()
#         text = re.sub(r"[^a-z ]", "", text)
#         return text

#     def _basic_ctc_decode(self, logits: np.ndarray, sequence_length: int) -> List[str]:
#         if self.debug:
#             print("[DEBUG] _basic_ctc_decode => start")
#         argmax_indices = np.argmax(logits, axis=-1)

#         if len(argmax_indices.shape) == 0:
#             argmax_indices = np.array([argmax_indices])
#         if len(argmax_indices.shape) == 1:
#             argmax_indices = np.expand_dims(argmax_indices, axis=0)

#         predictions = []
#         for sequence in argmax_indices:
#             decoded = []
#             last_idx = None
#             for idx in sequence[:sequence_length]:
#                 if idx != self.blank_index and idx != last_idx:
#                     decoded.append(self.ind2char[idx])
#                 last_idx = idx
#             text = "".join(decoded)
#             if self.use_bpe and hasattr(self, 'sp') and self.sp:
#                 # Could decode with sp here, but we keep naive
#                 pass
#             predictions.append(text)
#         return predictions

#     def test_decoder(self, sample_text: str = "test decoder functionality"):
#         if self.debug:
#             print("[DEBUG] test_decoder =>", sample_text)
#         encoded = self.encode(sample_text)
#         decoded = self.decode(encoded[0].tolist())
#         if self.debug:
#             print(f"[DEBUG] Original: '{sample_text}' => Decoded: '{decoded}'")

#         seq_length = 50
#         vocab_size = len(self)
#         fake_logits = torch.randn(1, seq_length, vocab_size)

#         if self.decoder is not None:
#             if self.debug:
#                 print("[DEBUG] Testing pyctcdecode on fake logits => ctc_decode()")
#             out = self.ctc_decode(fake_logits)
#             if self.debug:
#                 print(f"[DEBUG] Decoded with LM => '{out}'")
#         else:
#             if self.debug:
#                 print("[DEBUG] No LM => calling _basic_ctc_decode (char-based) on fake logits")
#             basic_dec = self._basic_ctc_decode(fake_logits.numpy(), seq_length)
#             if self.debug:
#                 print(f"[DEBUG] Basic ctc decoded => '{basic_dec[0]}'")

#     def __len__(self):
#         return len(self.vocab)

#     def ctc_beam_search(self, probs, beam_size=None, use_lm=False, debug=False) -> List[Tuple[str, float]]:
#         """
#         Beam search with optional LM. If LM => uses pyctcdecode, else => naive beam.
#         """
#         beam_size = beam_size or self.beam_size
#         if self.use_lm and self.decoder is not None:
#             try:
#                 if isinstance(probs, torch.Tensor):
#                     probs = probs.cpu().numpy()
#                 elif isinstance(probs, list):
#                     probs = np.array(probs)

#                 if self.debug:
#                     print(f"[DEBUG] ctc_beam_search => shape={probs.shape}, use_bpe={self.use_bpe}, lm_weight={self.lm_weight}")
#                 beams = self.decoder.decode_beams(
#                     probs,
#                     beam_prune_logp=-10.0,
#                     token_min_logp=-5.0,
#                     hotwords=[],
#                     hotword_weight=10.0,
#                 )
#                 formatted_beams = []
#                 for beam in beams[:beam_size]:
#                     text = beam[0]
#                     acoustic_score = beam[3]
#                     lm_score = beam[4]

#                     # Subword => remove "▁"
#                     if self.use_bpe and hasattr(self, 'sp') and self.sp is not None:
#                         text = text.replace("▁", " ")

#                     # Remove "??"
#                     if "??" in text:
#                         if self.debug:
#                             print(f"[DEBUG-lm] ctc_beam_search => removing '??'")
#                         text = text.replace("??", "")

#                     # Merge spaces
#                     text = re.sub(r'\s+', ' ', text).strip()

#                     combined_score = (1 - self.lm_weight)*acoustic_score + self.lm_weight*lm_score
#                     text_len = max(1, len(text.split())) if self.use_bpe else max(1, len(text))
#                     norm_score = combined_score / text_len
#                     formatted_beams.append((text.lower(), norm_score))

#                 if not formatted_beams:
#                     if self.debug:
#                         print("[DEBUG] No valid beams => fallback to standard beam search")
#                     return self._standard_beam_search(probs, debug)

#                 # Sort by normalized score
#                 return sorted(formatted_beams, key=lambda x: -x[1])

#             except Exception as e:
#                 if self.debug:
#                     print(f"[DEBUG] LM decode_beams failed => {e}, fallback to standard beam search")
#                 return self._standard_beam_search(probs, debug)
#         else:
#             return self._standard_beam_search(probs, debug)

#     def _standard_beam_search(self, probs, debug=False) -> List[Tuple[str, float]]:
#         """
#         Naive beam search w/o LM => prefix expansions.
#         """
#         beam_size = self.beam_size
#         if isinstance(probs, np.ndarray):
#             probs = torch.from_numpy(probs)
#         if probs.device != torch.device('cpu'):
#             probs = probs.cpu()

#         dp = {("", self.blank_token): 0.0}
#         log_probs = torch.log(probs + 1e-8)

#         for t, prob in enumerate(log_probs):
#             new_dp = defaultdict(lambda: float('-inf'))
#             top_k = torch.topk(prob, k=min(beam_size, len(prob)))

#             if self.debug and t < 3:
#                 print(f"[DEBUG] _standard_beam_search => t={t}, top_k.indices={top_k.indices.tolist()}")

#             for val, ind in zip(top_k.values, top_k.indices):
#                 curr_char = self.ind2char[ind.item()]
#                 next_lp = val.item()

#                 for (prefix, last_char), old_lp in dp.items():
#                     if last_char == curr_char and curr_char != " ":
#                         new_prefix = prefix
#                     else:
#                         if curr_char != self.blank_token:
#                             if curr_char == " " and prefix.endswith(" "):
#                                 continue
#                             new_prefix = prefix + curr_char
#                         else:
#                             new_prefix = prefix

#                     new_score = old_lp + next_lp
#                     key = (new_prefix, curr_char)
#                     new_dp[key] = max(new_dp[key], new_score)

#             if len(new_dp) > 0:
#                 max_score = max(new_dp.values())
#                 new_dp = {k: v - max_score for k, v in new_dp.items()}

#             dp = dict(sorted(new_dp.items(), key=lambda x: -x[1])[:beam_size])

#         final_beams = []
#         for (text, _), score in dp.items():
#             # Subword => remove "▁" and "??"
#             if self.use_bpe and hasattr(self, 'sp') and self.sp is not None:
#                 text = text.replace("▁", " ")
#             if "??" in text:
#                 if self.debug:
#                     print(f"[DEBUG] _standard_beam_search => removing '??'")
#                 text = text.replace("??", "")
#             text = text.lower().strip()
#             text_len = max(1, len(text.split())) if self.use_bpe else max(1, len(text))
#             norm_score = score / text_len
#             final_beams.append((text, norm_score))

#         final_beams.sort(key=lambda x: -x[1])
#         if not final_beams:
#             final_beams = [("", float('-inf'))]
#         return final_beams[:beam_size]


# import re
# from collections import defaultdict
# import torch
# import kenlm
# from pyctcdecode import build_ctcdecoder
# import numpy as np
# import os
# from typing import List, Tuple, Optional, Union

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# class CTCTextEncoder:
#     def __init__(
#         self,
#         arpa_path: Optional[str] = None,
#         binary_path: Optional[str] = "4-gram_lc_correct.bin",
#         unigram_path: Optional[str] = "librispeech-vocab.txt",
#         pretrained_tokenizer: str = "sentencepiece_model/librispeech_unigram_model.model",
#         lm_weight: float = 0.5,
#         beam_size: int = 100,
#         use_lm: bool = False,
#         use_bpe: bool = False,
#         blank_token: str = "<pad>",
#         unk_token: str = "<unk>",
#         **kwargs
#     ):
#         """
#         Initialize encoder with conditional subword (SentencePiece) or char-based vocab.
#         """
#         # Debug switch
#         self.debug = False  # <-- Set to True for verbose printing

#         if self.debug:
#             print("[DEBUG] Initializing CTCTextEncoder...")
#             print(f"  -> arpa_path={arpa_path}, binary_path={binary_path}, unigram_path={unigram_path}")
#             print(f"  -> pretrained_tokenizer={pretrained_tokenizer}")
#             print(f"  -> lm_weight={lm_weight}, beam_size={beam_size}")
#             print(f"  -> use_lm={use_lm}, use_bpe={use_bpe}")
#             print(f"  -> blank_token={blank_token}, unk_token={unk_token}")

#         self.beam_size = beam_size
#         self.lm_weight = lm_weight
#         self.arpa_path = arpa_path
#         self.binary_path = binary_path
#         self.blank_token = blank_token
#         self.unk_token = unk_token
#         self.use_lm = use_lm
#         self.use_bpe = use_bpe

#         # Load unigrams if provided
#         self.unigrams = None
#         if unigram_path and os.path.exists(unigram_path):
#             if self.debug:
#                 print(f"[DEBUG] Loading unigrams from {unigram_path}")
#             with open(unigram_path, 'r', encoding='utf-8') as f:
#                 self.unigrams = [line.strip().lower() for line in f if line.strip()]
#             if self.debug:
#                 print(f"[DEBUG] Loaded {len(self.unigrams)} unigrams")

#         # Initialize vocabulary/tokenizer
#         self._initialize_vocabulary(pretrained_tokenizer)

#         # Create index mappings
#         self.ind2char = dict(enumerate(self.vocab))
#         self.char2ind = {v: k for k, v in self.ind2char.items()}
#         self.blank_index = self.char2ind.get(self.blank_token, None)

#         if self.debug:
#             print("\n[DEBUG] Vocabulary Info:")
#             print(f"  -> vocab size: {len(self.vocab)}")
#             print(f"  -> blank_token='{self.blank_token}' => index={self.blank_index}")
#             print("  -> sample vocab:", self.vocab[:30])

#             sample_inds = list(self.ind2char.keys())[:10]
#             print("  -> Sample ind2char:", {k: self.ind2char[k] for k in sample_inds})

#             sample_chars = list(self.char2ind.keys())[:10]
#             print("  -> Sample char2ind:", {k: self.char2ind[k] for k in sample_chars})

#         # Optionally load LM
#         if self.use_lm:
#             self._initialize_language_model()
#         else:
#             if self.debug:
#                 print("[DEBUG] No LM usage. self.decoder=None")
#             self.lm = None
#             self.decoder = None

#         if self.debug:
#             print("[DEBUG] CTCTextEncoder initialized.\n")

#     def _initialize_vocabulary(self, pretrained_tokenizer: str):
#         """
#         If use_bpe=True, load a SentencePiece model from `pretrained_tokenizer` path.
#         Otherwise, use a basic character-based vocab.
#         """
#         if self.use_bpe:
#             if self.debug:
#                 print("[DEBUG] use_bpe=True => Loading SentencePiece model.")
#             import sentencepiece as spm

#             if not os.path.exists(pretrained_tokenizer):
#                 raise FileNotFoundError(
#                     f"SentencePiece model not found at: {pretrained_tokenizer}"
#                 )

#             # Load the SentencePiece model
#             self.sp = spm.SentencePieceProcessor()
#             self.sp.load(pretrained_tokenizer)

#             vocab_size = self.sp.get_piece_size()
#             if self.debug:
#                 print(f"[DEBUG] Loaded SP model => vocab_size={vocab_size}")
#                 print(f"[DEBUG] sp.IdToPiece(0)={self.sp.id_to_piece(0)} (often <unk>)")

#             self.labels = [self.sp.id_to_piece(i) for i in range(vocab_size)]
#             self.vocab = self.labels  # len(self.vocab) = vocab_size
#         else:
#             # Character-based => do not touch
#             if self.debug:
#                 print("Initializing character-based vocabulary without using tokenizer.")
#             self.vocab = [
#                 'a','b','c','d','e','f','g','h','i','j',
#                 'k','l','m','n','o','p','q','r','s','t',
#                 'u','v','w','x','y','z',' '
#             ]
#             self.vocab += [self.blank_token, self.unk_token]
#             self.labels = self.vocab
#             self.sp = None

#     def encode(self, text: str) -> torch.Tensor:
#         """
#         Encode text. If bpe => subword, else => char-based.
#         """
#         if self.use_bpe and hasattr(self, 'sp') and self.sp is not None:
#             # text = text.upper() if your model is uppercase
#             token_ids = self.sp.encode(text, out_type=int)
#             if self.debug:
#                 print(f"[DEBUG-bpe] encode => text='{text}' => token_ids={token_ids}")
#             unknown_count = sum(1 for tid in token_ids if self.sp.id_to_piece(tid) == "<unk>")
#             if unknown_count > 0 and self.debug:
#                 print(f"[DEBUG-bpe] <unk> count={unknown_count}, might indicate coverage issues.")
#             return torch.tensor([token_ids], dtype=torch.long)
#         else:
#             # char-based => unchanged
#             normalized_text = self.normalize_text(text)
#             token_indices = [
#                 self.char2ind.get(char, self.char2ind.get(self.unk_token))
#                 for char in normalized_text
#             ]
#             return torch.tensor(token_indices).unsqueeze(0)

#     def decode_simple(self, indices: List[int]) -> str:
#         """
#         Simple CTC decode => collapses repeats, removes blank.
#         If bpe => subword decode after collapsing, else => char-based.
#         """
#         if self.use_bpe and hasattr(self, 'sp') and self.sp is not None:
#             collapsed = []
#             prev_idx = None
#             for idx in indices:
#                 if idx == self.blank_index:
#                     prev_idx = idx
#                     continue
#                 if idx != prev_idx:
#                     collapsed.append(idx)
#                 prev_idx = idx

#             if self.debug:
#                 print(f"[DEBUG-bpe] decode_simple => collapsed={collapsed}")

#             text = self.sp.decode(collapsed)
#             if self.debug:
#                 print(f"[DEBUG-bpe] decode_simple => text='{text}' before filtering unknown placeholders")

#             # Convert "▁" => space
#             text = text.replace("▁", " ")

#             # Remove "⁇"
#             if "⁇" in text:
#                 if self.debug:
#                     print(f"[DEBUG-bpe] Found '⁇' in text => '{text}' => removing them.")
#                 text = text.replace("⁇", "")

#             # Merge repeated spaces
#             text = re.sub(r'\s+', ' ', text).strip()

#             return text
#         else:
#             # char-based => do not touch
#             decoded_chars = []
#             prev_idx = None
#             for idx in indices:
#                 if idx == self.blank_index:
#                     prev_idx = idx
#                     continue
#                 if idx == prev_idx:
#                     continue
#                 if 0 <= idx < len(self.ind2char):
#                     decoded_chars.append(self.ind2char[idx])
#                 prev_idx = idx
#             text = "".join(decoded_chars).strip().lower()
#             return text

#     def decode(self, indices: List[int]) -> str:
#         """
#         If LM => self.decoder.decode(...), else => decode_simple
#         """
#         if self.decoder:
#             decoded_text = self.decoder.decode(indices)
#             decoded_text = decoded_text.lower().replace("▁", " ").strip()
#             return decoded_text
#         else:
#             return self.decode_simple(indices).lower()

#     def decode_logits(self, logits: Union[torch.Tensor, List[List[float]], np.ndarray]) -> str:
#         """
#         If LM => self.decoder.decode(logits), else => greedy => decode_simple
#         """
#         if isinstance(logits, torch.Tensor):
#             logits = logits.cpu().numpy()
#         elif isinstance(logits, list):
#             logits = np.array(logits)

#         if logits.ndim == 3:
#             logits = logits[0]
#         if logits.ndim != 2:
#             raise ValueError(f"Logits should be 2D, got shape {logits.shape}")

#         if self.debug:
#             print(f"[DEBUG] decode_logits => shape={logits.shape}, use_lm={self.use_lm}, use_bpe={self.use_bpe}")

#         if self.decoder:
#             text = self.decoder.decode(logits)
#             text = text.replace("▁", " ").lower().strip()
#             if self.debug:
#                 print(f"[DEBUG-lm] decode_logits => partial='{text[:50]}...'")
#             return text
#         else:
#             predicted_indices = np.argmax(logits, axis=-1).tolist()
#             return self.decode_simple(predicted_indices)

#     def decode_indices(self, indices: Union[torch.Tensor, List[int], np.ndarray]) -> str:
#         """
#         Direct decode => no LM
#         """
#         if isinstance(indices, torch.Tensor):
#             indices = indices.squeeze().tolist()
#         elif isinstance(indices, np.ndarray):
#             indices = indices.tolist()
#         return self.decode_simple(indices)

#     def ctc_decode(self, logits: Union[torch.Tensor, List[int], np.ndarray]) -> str:
#         """
#         CTC decode from logits or token indices.
#         If bpe => subword approach, else => char-based approach
#         """
#         if isinstance(logits, np.ndarray):
#             logits = torch.from_numpy(logits)
#         elif isinstance(logits, list):
#             logits = torch.tensor(logits)

#         if self.debug:
#             print(f"[DEBUG] ctc_decode => shape={logits.shape}, use_bpe={self.use_bpe}")

#         if self.use_bpe:
#             # subword path
#             if logits.dim() == 3:
#                 return self.decode_logits(logits)
#             elif logits.dim() == 2:
#                 return self.decode_logits(logits)
#             elif logits.dim() == 1:
#                 return self.decode_indices(logits)
#             else:
#                 raise ValueError(f"Unsupported logits shape: {logits.shape}.")
#         else:
#             # char-based => do NOT TOUCH
#             if logits.dim() == 3:
#                 return self.decode_logits(logits)
#             elif logits.dim() == 2:
#                 return self.decode_logits(logits)
#             elif logits.dim() == 1:
#                 return self.decode_indices(logits)
#             else:
#                 raise ValueError(f"Unsupported logits shape: {logits.shape}.")

#     def _initialize_language_model(self):
#         if self.debug:
#             print("[DEBUG] _initialize_language_model => start")
#         self.lm = None
#         self.decoder = None

#         model_path = self.binary_path if self.binary_path else self.arpa_path
#         if self.debug:
#             print(f"[DEBUG] model_path => {model_path}")
#         if not model_path or not os.path.exists(model_path):
#             if self.debug:
#                 print("[DEBUG] No valid LM path => skipping LM init.")
#             return

#         try:
#             self.lm = kenlm.Model(model_path)
#             if self.debug:
#                 print("[DEBUG] KenLM model loaded from:", model_path)

#             decoder_config = {
#                 "labels": self.labels if self.use_bpe else self.vocab,
#                 "kenlm_model_path": model_path,
#                 "alpha": self.lm_weight,
#                 "beta": 0.1,
#                 "unk_score_offset": -10.0,
#             }

#             if self.unigrams:
#                 if self.debug:
#                     print("[DEBUG] Found unigrams => adding to decoder_config.")
#                 with open("unigrams_list.txt", "w", encoding="utf-8") as f:
#                     for unigram in self.unigrams:
#                         f.write(f"{unigram}\n")
#                 decoder_config["unigrams"] = self.unigrams

#             self.decoder = build_ctcdecoder(**decoder_config)
#             if self.debug:
#                 print("[DEBUG] LM-based decoder successfully initialized.")
#         except Exception as e:
#             if self.debug:
#                 print(f"[DEBUG] LM init failed => {str(e)}")
#             self.decoder = None

#     def test_language_model(self):
#         if self.debug:
#             print("[DEBUG] test_language_model =>")
#         if self.lm is None:
#             if self.debug:
#                 print("[DEBUG] No LM loaded.")
#             return

#         sample_sents = ["this is a test", "hello world", "aaaa bbbb cccc dddd"]
#         for s in sample_sents:
#             score = self.lm.score(s, bos=True, eos=True)
#             if self.debug:
#                 print(f"LM Score for '{s}' => {score:.4f}")

#     def score_with_lm(self, text: str) -> float:
#         if self.lm is None:
#             return 0.0
#         return self.lm.score(text.lower().strip(), bos=True, eos=True)

#     @staticmethod
#     def normalize_text(text: str) -> str:
#         """
#         Basic normalization => lower + remove non-alpha + space
#         """
#         text = text.lower()
#         text = re.sub(r"[^a-z ]", "", text)
#         return text

#     def _basic_ctc_decode(self, logits: np.ndarray, sequence_length: int) -> List[str]:
#         if self.debug:
#             print("[DEBUG] _basic_ctc_decode => start")
#         argmax_indices = np.argmax(logits, axis=-1)

#         if len(argmax_indices.shape) == 0:
#             argmax_indices = np.array([argmax_indices])
#         if len(argmax_indices.shape) == 1:
#             argmax_indices = np.expand_dims(argmax_indices, axis=0)

#         predictions = []
#         for sequence in argmax_indices:
#             decoded = []
#             last_idx = None
#             for idx in sequence[:sequence_length]:
#                 if idx != self.blank_index and idx != last_idx:
#                     decoded.append(self.ind2char[idx])
#                 last_idx = idx
#             text = "".join(decoded)
#             if self.use_bpe and hasattr(self, 'sp') and self.sp:
#                 # Could decode with self.sp here, but we keep naive
#                 pass
#             predictions.append(text)
#         return predictions

#     def test_decoder(self, sample_text: str = "test decoder functionality"):
#         if self.debug:
#             print("[DEBUG] test_decoder =>", sample_text)
#         encoded = self.encode(sample_text)
#         decoded = self.decode(encoded[0].tolist())
#         if self.debug:
#             print(f"[DEBUG] Original: '{sample_text}' => Decoded: '{decoded}'")

#         seq_length = 50
#         vocab_size = len(self)
#         fake_logits = torch.randn(1, seq_length, vocab_size)

#         if self.decoder is not None:
#             if self.debug:
#                 print("[DEBUG] Testing pyctcdecode on fake logits => ctc_decode()")
#             out = self.ctc_decode(fake_logits)
#             if self.debug:
#                 print(f"[DEBUG] Decoded with LM => '{out}'")
#         else:
#             if self.debug:
#                 print("[DEBUG] No LM => calling _basic_ctc_decode (char-based) on fake logits")
#             basic_dec = self._basic_ctc_decode(fake_logits.numpy(), seq_length)
#             if self.debug:
#                 print(f"[DEBUG] Basic ctc decoded => '{basic_dec[0]}'")

#     def __len__(self):
#         return len(self.vocab)

#     def ctc_beam_search(self, probs, beam_size=None, use_lm=False, debug=False) -> List[Tuple[str, float]]:
#         """
#         Beam search with optional LM. If LM => uses pyctcdecode, else => naive beam.
#         """
#         beam_size = beam_size or self.beam_size
#         if self.use_lm and self.decoder is not None:
#             try:
#                 if isinstance(probs, torch.Tensor):
#                     probs = probs.cpu().numpy()
#                 elif isinstance(probs, list):
#                     probs = np.array(probs)

#                 if self.debug:
#                     print(f"[DEBUG] ctc_beam_search => shape={probs.shape}, use_bpe={self.use_bpe}")
#                 beams = self.decoder.decode_beams(
#                     probs,
#                     beam_prune_logp=-10.0,
#                     token_min_logp=-5.0,
#                     hotwords=[],
#                     hotword_weight=10.0,
#                 )
#                 formatted_beams = []
#                 for beam in beams[:beam_size]:
#                     text = beam[0]
#                     acoustic_score = beam[3]
#                     lm_score = beam[4]

#                     if self.use_bpe and hasattr(self, 'sp') and self.sp is not None:
#                         text = text.replace("▁", " ")
#                     text = text.lower().strip()

#                     combined_score = (1 - self.lm_weight)*acoustic_score + self.lm_weight*lm_score
#                     text_len = max(1, len(text.split())) if self.use_bpe else max(1, len(text))
#                     norm_score = combined_score / text_len
#                     formatted_beams.append((text, norm_score))

#                 if formatted_beams:
#                     return sorted(formatted_beams, key=lambda x: -x[1])
#                 else:
#                     if self.debug:
#                         print("[DEBUG] No valid beams => fallback to standard beam search")
#                     return self._standard_beam_search(probs, debug)
#             except Exception as e:
#                 if self.debug:
#                     print(f"[DEBUG] LM decode_beams failed => {e} => fallback standard beam search")
#                 return self._standard_beam_search(probs, debug)
#         else:
#             return self._standard_beam_search(probs, debug)

#     def _standard_beam_search(self, probs, debug=False) -> List[Tuple[str, float]]:
#         """
#         Naive beam search w/o LM => prefix expansions.
#         """
#         beam_size = self.beam_size
#         if isinstance(probs, np.ndarray):
#             probs = torch.from_numpy(probs)
#         if probs.device != torch.device('cpu'):
#             probs = probs.cpu()

#         dp = {("", self.blank_token): 0.0}
#         log_probs = torch.log(probs + 1e-8)

#         for t, prob in enumerate(log_probs):
#             new_dp = defaultdict(lambda: float('-inf'))
#             top_k = torch.topk(prob, k=min(beam_size, len(prob)))

#             # only if self.debug
#             if self.debug and t < 3:
#                 print(f"[DEBUG] _standard_beam_search: t={t}, top_k.indices={top_k.indices}")

#             for val, ind in zip(top_k.values, top_k.indices):
#                 curr_char = self.ind2char[ind.item()]
#                 next_lp = val.item()

#                 for (prefix, last_char), old_lp in dp.items():
#                     if last_char == curr_char and curr_char != " ":
#                         new_prefix = prefix
#                     else:
#                         if curr_char != self.blank_token:
#                             if curr_char == " " and prefix.endswith(" "):
#                                 continue
#                             new_prefix = prefix + curr_char
#                         else:
#                             new_prefix = prefix

#                     new_score = old_lp + next_lp
#                     key = (new_prefix, curr_char)
#                     # keep only max
#                     new_dp[key] = max(new_dp[key], new_score)

#             if len(new_dp) > 0:
#                 max_score = max(new_dp.values())
#                 new_dp = {k: v - max_score for k, v in new_dp.items()}

#             # Keep top beams
#             dp = dict(sorted(new_dp.items(), key=lambda x: -x[1])[:beam_size])

#         final_beams = []
#         for (text, _), score in dp.items():
#             if self.use_bpe and hasattr(self, 'sp') and self.sp is not None:
#                 text = text.replace("▁", " ")
#             text = text.lower().strip()
#             text_len = max(1, len(text.split())) if self.use_bpe else max(1, len(text))
#             norm_score = score / text_len
#             final_beams.append((text, norm_score))

#         final_beams.sort(key=lambda x: -x[1])
#         if not final_beams:
#             final_beams = [("", float('-inf'))]
#         return final_beams[:beam_size]
