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
            print(f"[DEBUG] Loading unigrams from {unigram_path}")
            with open(unigram_path, 'r', encoding='utf-8') as f:
                self.unigrams = [line.strip().lower() for line in f if line.strip()]
            print(f"[DEBUG] Loaded {len(self.unigrams)} unigrams")

        # Initialize vocabulary/tokenizer
        self._initialize_vocabulary(pretrained_tokenizer)

        # Create index mappings
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.blank_index = self.char2ind.get(self.blank_token, None)

        print("\n[DEBUG] Vocabulary Info:")
        print(f"  -> vocab size: {len(self.vocab)}")
        print(f"  -> blank_token='{self.blank_token}' => index={self.blank_index}")
        print("  -> sample vocab:", self.vocab[:30])

        # Show some mapping samples
        sample_inds = list(self.ind2char.keys())[:10]
        print("  -> Sample ind2char:", {k: self.ind2char[k] for k in sample_inds})

        sample_chars = list(self.char2ind.keys())[:10]
        print("  -> Sample char2ind:", {k: self.char2ind[k] for k in sample_chars})

        # Optionally load LM
        if self.use_lm:
            self._initialize_language_model()
        else:
            print("[DEBUG] No LM usage. self.decoder=None")
            self.lm = None
            self.decoder = None

        print("[DEBUG] CTCTextEncoder initialized.\n")

    def _initialize_vocabulary(self, pretrained_tokenizer: str):
        """
        If use_bpe=True, load a SentencePiece model from `pretrained_tokenizer` path.
        Otherwise, use a basic character-based vocab.
        """
        if self.use_bpe:
            print("[DEBUG] use_bpe=True => Loading SentencePiece model.")
            import sentencepiece as spm

            if not os.path.exists(pretrained_tokenizer):
                raise FileNotFoundError(
                    f"SentencePiece model not found at: {pretrained_tokenizer}"
                )

            # Load the SentencePiece model
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(pretrained_tokenizer)

            vocab_size = self.sp.get_piece_size()
            print(f"[DEBUG] Loaded SP model => vocab_size={vocab_size}")

            self.labels = [self.sp.id_to_piece(i) for i in range(vocab_size)]
            self.vocab = self.labels  # len(self.vocab) = vocab_size
        # else:
        #     print("[DEBUG] use_bpe=False => Using char-based vocab.")
        #     self.vocab = [
        #         "a","b","c","d","e","f","g","h","i","j",
        #         "k","l","m","n","o","p","q","r","s","t",
        #         "u","v","w","x","y","z"," "
        #     ]
        #     # Put blank at the front
        #     self.vocab = [self.blank_token] + self.vocab
        #     self.labels = self.vocab
        #     self.sp = None


        else:
            print("Initializing character-based vocabulary without using tokenizer.")
            self.vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                        'y', 'z', ' ']
            self.vocab += [self.blank_token, self.unk_token]
            self.labels = self.vocab
            self.ind2char = dict(enumerate(self.vocab))
            self.char2ind = {v: k for k, v in self.ind2char.items()}
            self.blank_index = self.char2ind.get(self.blank_token, None)
            self.sp = None

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text: if bpe => sp.encode, else char-based => char2ind.
        Returns a [1, seq_len] tensor of token IDs.
        """
        if self.use_bpe and self.sp is not None:
            token_ids = self.sp.encode(text, out_type=int)
            return torch.tensor([token_ids], dtype=torch.long)
        else:
            normalized_text = self.normalize_text(text)
            token_indices = [
                self.char2ind.get(char, self.char2ind.get(self.unk_token))
                for char in normalized_text
            ]
            return torch.tensor(token_indices).unsqueeze(0)

    def decode_simple(self, indices: List[int]) -> str:
        """
        Simple CTC decoding: collapse repeats, remove blank. Lowercase for char-based.
        For BPE, do sp.decode on collapsed subwords.
        """
        if self.use_bpe and self.sp is not None:
            # BPE path => subword decode
            collapsed = []
            prev_idx = None
            for idx in indices:
                if idx == self.blank_index:
                    prev_idx = idx
                    continue
                if idx != prev_idx:
                    collapsed.append(idx)
                prev_idx = idx

            text = self.sp.decode(collapsed)
            text = text.replace("▁", " ").strip()
            return text
        else:
            # Char-based => replicate old version's approach
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
        If LM is present => use decoder.decode, else decode_simple.
        """
        if self.decoder:
            decoded_text = self.decoder.decode(indices)
            decoded_text = decoded_text.lower().replace("▁", " ").strip()
            return decoded_text
        else:
            return self.decode_simple(indices).lower()

    def decode_logits(self, logits: Union[torch.Tensor, List[List[float]], np.ndarray]) -> str:
        """
        If LM => decoder.decode(logits).
        Else => greedy decode -> decode_simple.
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        elif isinstance(logits, list):
            logits = np.array(logits)

        if logits.ndim == 3:
            logits = logits[0]
        if logits.ndim != 2:
            raise ValueError(f"Logits should be 2D, got shape {logits.shape}")

        if self.decoder:
            text = self.decoder.decode(logits)
            text = text.replace("▁", " ").lower().strip()
            return text
        else:
            predicted_indices = np.argmax(logits, axis=-1).tolist()
            return self.decode_simple(predicted_indices)

    def decode_indices(self, indices: Union[torch.Tensor, List[int], np.ndarray]) -> str:
        """
        Directly decode given token indices with no LM.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.squeeze().tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()
        return self.decode_simple(indices)

    def ctc_decode(self, logits: Union[torch.Tensor, List[int], np.ndarray]) -> str:
        """
        CTC decode from logits -> decode or decode_simple.

        *** Key Fix: If use_bpe=False => char-based decode_simple directly,
        otherwise => new approach with LM or subword decode. ***
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        elif isinstance(logits, list):
            logits = torch.tensor(logits)

        # If we have subword => same as new version, else replicate old approach
        if self.use_bpe:
            # Subword path
            if logits.dim() == 3:
                # pass to decode_logits => will check self.decoder if present
                return self.decode_logits(logits)
            elif logits.dim() == 2:
                return self.decode_logits(logits)
            elif logits.dim() == 1:
                return self.decode_indices(logits)
            else:
                raise ValueError(f"Unsupported logits shape: {logits.shape}. Expected 1D, 2D, or 3D.")
        else:
            # Char-based path => replicate old approach
            if logits.dim() == 3:
                # We do greedy decode or LM? The old version always calls decode_logits
                return self.decode_logits(logits)
            elif logits.dim() == 2:
                return self.decode_logits(logits)
            elif logits.dim() == 1:
                return self.decode_indices(logits)
            else:
                raise ValueError(f"Unsupported logits shape: {logits.shape}. Expected 1D, 2D, or 3D.")

    def _initialize_language_model(self):
        print("[DEBUG] _initialize_language_model => start")
        self.lm = None
        self.decoder = None

        model_path = self.binary_path if self.binary_path else self.arpa_path
        print(f"[DEBUG] model_path => {model_path}")
        if not model_path or not os.path.exists(model_path):
            print("[DEBUG] No valid LM path => skipping LM init.")
            return

        try:
            self.lm = kenlm.Model(model_path)
            print("[DEBUG] KenLM model loaded from:", model_path)

            decoder_config = {
                "labels": self.labels if self.use_bpe else self.vocab,
                "kenlm_model_path": model_path,
                "alpha": self.lm_weight,
                "beta": 0.1,
                "unk_score_offset": -10.0,
            }

            if self.unigrams:
                print("[DEBUG] Found unigrams => adding to decoder_config.")
                with open("unigrams_list.txt", "w", encoding="utf-8") as f:
                    for unigram in self.unigrams:
                        f.write(f"{unigram}\n")
                decoder_config["unigrams"] = self.unigrams

            self.decoder = build_ctcdecoder(**decoder_config)
            print("[DEBUG] LM-based decoder successfully initialized.")
        except Exception as e:
            print(f"[DEBUG] LM init failed => {str(e)}")
            self.decoder = None

    def test_language_model(self):
        """
        Quick test for LM if loaded.
        """
        print("[DEBUG] test_language_model =>")
        if self.lm is None:
            print("[DEBUG] No LM loaded.")
            return

        sample_sents = ["this is a test", "hello world", "aaaa bbbb cccc dddd"]
        for s in sample_sents:
            score = self.lm.score(s, bos=True, eos=True)
            print(f"LM Score for '{s}' => {score:.4f}")

    def score_with_lm(self, text: str) -> float:
        if self.lm is None:
            return 0.0
        return self.lm.score(text.lower().strip(), bos=True, eos=True)

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Basic normalization => lower + remove non-alpha + space.
        """
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def _basic_ctc_decode(self, logits: np.ndarray, sequence_length: int) -> List[str]:
        """
        Basic CTC greedy decode without LM.
        Typically for use_bpe=False path or fallback test.
        """
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
            if self.use_bpe and self.sp:
                # Optionally sp.decode, but we keep naive
                pass
            predictions.append(text)
        return predictions

    def test_decoder(self, sample_text: str = "test decoder functionality"):
        """
        Debug method: encode & decode a sample_text, then random logits decode.
        """
        print("[DEBUG] test_decoder =>", sample_text)
        encoded = self.encode(sample_text)
        decoded = self.decode(encoded[0].tolist())
        print(f"[DEBUG] Original: '{sample_text}' => Decoded: '{decoded}'")

        seq_length = 50
        vocab_size = len(self)
        fake_logits = torch.randn(1, seq_length, vocab_size)

        if self.decoder is not None:
            print("[DEBUG] Testing pyctcdecode on fake logits.")
            out = self.ctc_decode(fake_logits)
            print(f"[DEBUG] Decoded with LM => '{out}'")
        else:
            print("[DEBUG] No LM => basic ctc decode on fake logits.")
            basic_dec = self._basic_ctc_decode(fake_logits.numpy(), seq_length)
            print(f"[DEBUG] Basic ctc decoded => '{basic_dec[0]}'")

    def __len__(self):
        return len(self.vocab)

    def ctc_beam_search(self, probs, beam_size=None, use_lm=False, debug=False) -> List[Tuple[str, float]]:
        """
        Beam search with optional LM. Uses pyctcdecode if LM, else fallback.
        """
        beam_size = beam_size or self.beam_size
        if self.use_lm and self.decoder is not None:
            try:
                if isinstance(probs, torch.Tensor):
                    probs = probs.cpu().numpy()
                elif isinstance(probs, list):
                    probs = np.array(probs)

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

                    if self.use_bpe and self.sp is not None:
                        text = text.replace("▁", " ")
                    text = text.lower().strip()

                    combined_score = (1 - self.lm_weight)*acoustic_score + self.lm_weight*lm_score
                    text_len = max(1, len(text.split())) if self.use_bpe else max(1, len(text))
                    norm_score = combined_score / text_len
                    formatted_beams.append((text, norm_score))

                if formatted_beams:
                    return sorted(formatted_beams, key=lambda x: -x[1])
                else:
                    print("[DEBUG] No valid beams => fallback to standard beam search")
                    return self._standard_beam_search(probs, debug)
            except Exception as e:
                print(f"[DEBUG] LM decode_beams failed => {e} => fallback standard beam search")
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
                    new_dp[key] = max(new_dp[key], new_score)

            if len(new_dp) > 0:
                max_score = max(new_dp.values())
                new_dp = {k: v - max_score for k, v in new_dp.items()}

            dp = dict(sorted(new_dp.items(), key=lambda x: -x[1])[:beam_size])

        final_beams = []
        for (text, _), score in dp.items():
            if self.use_bpe and self.sp is not None:
                text = text.replace("▁", " ")
            text = text.lower().strip()
            text_len = max(1, len(text.split())) if self.use_bpe else max(1, len(text))
            norm_score = score / text_len
            final_beams.append((text, norm_score))

        final_beams.sort(key=lambda x: -x[1])
        if not final_beams:
            final_beams = [("", float('-inf'))]
        return final_beams[:beam_size]
