import re
from string import ascii_lowercase
from collections import defaultdict

import torch

from typing import List, Tuple
import kenlm # for LM usage
from transformers import AutoTokenizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warnings in transformers

# TODO add CTC decode - done
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    # def __init__(self, alphabet=None, arpa_path=None, lm_weight=0.5, **kwargs):
    # def __init__(self, alphabet=None, arpa_path='4-gram.arpa', lm_weight=0.5, **kwargs):
    #     """
    #     Args:
    #         alphabet (list): alphabet for language. If None, will be set to ascii
    #         arpa_path (str): path to ARPA language model file
    #         lm_weight (float): weight for language model scoring (0 to 1)
    #     """

    #     if alphabet is None:
    #         alphabet = list(ascii_lowercase + " ")

    #     self.alphabet = alphabet
    #     self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

    #     self.ind2char = dict(enumerate(self.vocab))
    #     self.char2ind = {v: k for k, v in self.ind2char.items()}
        
    #     # Initialize language model if provided
    #     self.lm = None
    #     self.lm_weight = lm_weight
    #     if arpa_path:
    #         self.lm = kenlm.Model(arpa_path)
        
    def __init__(self, alphabet=None, arpa_path='4-gram.arpa', binary_path='4-gram.bin', lm_weight=0.5, 
                 use_bpe=True, pretrained_tokenizer="bert-base-uncased", **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, will be set to ascii
            arpa_path (str): path to ARPA language model file
            lm_weight (float): weight for language model scoring
            use_bpe (bool): whether to use BPE tokenization
            pretrained_tokenizer (str): HuggingFace tokenizer to use if use_bpe=True
        """
        self.EMPTY_TOK = ""
        self.use_bpe = use_bpe
        
        # if use_bpe:
        #     # Load pretrained tokenizer
        #     self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
        #     # Set vocabulary from tokenizer
        #     self.vocab = [self.EMPTY_TOK] + list(self.tokenizer.vocab.keys())
        # else:
        #     # Original char-level setup
        #     if alphabet is None:
        #         alphabet = list(ascii_lowercase + " ")
        #     self.alphabet = alphabet
        #     self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        
        ### NEW VERSION TO FIX SPACES ERRROR!!! ###
        # Initialize vocabulary
        if use_bpe:
            # BPE tokenization
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
            self.EMPTY_TOK = self.tokenizer.pad_token or "[PAD]"
            self.vocab = list(self.tokenizer.vocab.keys())

            # Ensure the blank token is explicitly added if not in vocab
            if self.EMPTY_TOK not in self.vocab:
                self.vocab.append(self.EMPTY_TOK)
            print(f"Loaded BPE vocabulary of size: {len(self.vocab)}")
        else:
            # Character-level setup
            if alphabet is None:
                if self.unigrams:
                    # Build alphabet from unigrams
                    alphabet_set = set()
                    for word in self.unigrams:
                        alphabet_set.update(word)
                    alphabet_set.add(" ")
                    alphabet = sorted(list(alphabet_set))
                else:
                    alphabet = list(ascii_lowercase + " ")

            self.alphabet = alphabet
            self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        ### NEW VERSION TO FIX SPACES ERRROR!!! ###
        
        # Create index mappings
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        # Initialize language model if provided
        self.lm = None
        self.lm_weight = lm_weight
        if binary_path:
            self.lm = kenlm.Model(binary_path) # prefer binary format
            self.lm_state = kenlm.State()
            self.lm.BeginSentenceWrite(self.lm_state)
        elif arpa_path:
            self.lm = kenlm.Model(arpa_path)
            self.lm_state = kenlm.State()
            self.lm.BeginSentenceWrite(self.lm_state)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    # def encode(self, text) -> torch.Tensor:
    #     text = self.normalize_text(text)
    #     try:
    #         return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
    #     except KeyError:
    #         unknown_chars = set([char for char in text if char not in self.char2ind])
    #         raise Exception(
    #             f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
    #         )

    def encode(self, text) -> torch.Tensor:
        """
        Encode text using either BPE or char-level encoding
        """
        text = self.normalize_text(text)
        
        if self.use_bpe:
            try:
                # Tokenize with BPE
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                # Convert to tensor and add batch dimension
                return torch.tensor(tokens).unsqueeze(0)
            except Exception as e:
                raise Exception(f"BPE encoding error: {str(e)}")
        else:
            # Original char-level encoding
            try:
                return torch.tensor([self.char2ind[char] for char in text]).unsqueeze(0)
            except KeyError:
                unknown_chars = set([char for char in text if char not in self.char2ind])
                raise Exception(f"Unknown chars: '{' '.join(unknown_chars)}'")
    
    
    # def decode(self, inds) -> str:
    #     """
    #     Raw decoding without CTC.
    #     Used to validate the CTC decoding implementation.

    #     Args:
    #         inds (list): list of tokens.
    #     Returns:
    #         raw_text (str): raw text with empty tokens and repetitions.
    #     """
    #     return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    # def ctc_decode(self, inds) -> str:
    #     #pass  # TODO
        
    #     ### TO CHECK A LOT !!! ###
    #     # decoded_chars = []
    #     # prev_token = None

    #     # for token in inds:
    #     #     token = int(token)
    #     #     # Skip blank tokens
    #     #     if token == 0:
    #     #         prev_token = token
    #     #         continue

    #     #     # Only append if token != prev token to remove consecutive duplicates
    #     #     if token != prev_token:
    #     #         decoded_chars.append(self.ind2char[token])
    #     #     prev_token = token

    #     # # Join characters to form the decoded string
    #     # decoded_str = "".join(decoded_chars).strip()
    #     # return decoded_str
    
    #     ## TO CHECK A LOT !!! ##
        
    #     decoded = []
    #     last_char_ind = self.EMPTY_TOK
    #     for ind in inds:
    #         if last_char_ind == ind:
    #             continue
    #         if ind != self.EMPTY_TOK:
    #             decoded.append(self.ind2char[ind])
    #         last_char_ind = ind # update last char index
        
    #     decoded_text = "".join(decoded)
        
    #     return decoded_text
    
    def decode(self, indices) -> str:
        """
        Decode indices to text using either BPE or char-level decoding
        """
        if self.use_bpe:
            # Filter out EMPTY_TOK
            valid_indices = [int(idx) for idx in indices if int(idx) != 0]
            # Decode with BPE tokenizer
            try:
                return self.tokenizer.decode(valid_indices)
            except:
                # Fallback to basic decoding if tokenizer fails
                return " ".join([self.ind2char[idx] for idx in valid_indices])
        else:
            # Original char-level decoding
            return "".join([self.ind2char[int(ind)] for ind in indices if int(ind) != 0])

    def ctc_decode(self, inds) -> str:
        """
        CTC decoding - works with both BPE and char-level
        """
        decoded = []
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
            # Join BPE tokens and post-process
            text = " ".join(decoded)
            return self.tokenizer.clean_up_tokenization(text)
        else:
            # Original char-level joining
            return "".join(decoded)
    

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
    
    
    # Implement Beam search
    # def ctc_beam_search(self, probs, beam_size=10):
    #     """
    #     Beam search implementation for CTC decoding.
    #     Args:
    #         probs (Tensor): log probabilities from the model.
    #         beam_size (int): beam size.
    #     Returns:
    #         decoded_texts (list): list of decoded texts.
    #     """
    #     dp = {("", self.EMPTY_TOK): 1.0} # (decoded_text, last_char_):, prob - initialize with empty string
        
    #     for prob in probs:
    #         dp = self.expand_and_merge_path(dp, prob)
    #         dp = self.truncate_paths(dp, beam_size)
    #         dp = [(prefix, proba) for (prefix, _), proba in sorted(dp.items(), key=lambda x: -x[1])]
    #     return dp
    
    # def ctc_beam_search_with_lm(self, probs: torch.Tensor, beam_size: int = 10) -> List[Tuple[str, float]]:
    #     """
    #     Beam search with language model scoring
        
    #     Args:
    #         probs (Tensor): probabilities from model output (B, T, V)
    #         beam_size (int): beam size
    #     Returns:
    #         list of tuples (text, score)
    #     """
    #     # Initial beam with empty string
    #     dp = {("", self.EMPTY_TOK): (1.0, 0.0)}  # (text, last_char): (ctc_prob, lm_score)
        
    #     for prob in probs:
    #         dp = self._expand_and_merge_path_with_lm(dp, prob)
    #         dp = self._truncate_paths_with_lm(dp, beam_size)
        
    #     # Final scoring combining CTC and LM scores
    #     final_beams = []
    #     for (text, _), (ctc_prob, lm_score) in dp.items():
    #         combined_score = (1 - self.lm_weight) * ctc_prob + self.lm_weight * lm_score
    #         final_beams.append((text, combined_score))
        
    #     # Sort by combined score
    #     final_beams.sort(key=lambda x: x[1], reverse=True)
    #     return final_beams[:beam_size]
    
    
    # def expand_and_merge_path(self, dp, next_token_probs):
    #     new_dp = defaultdict(float) # dict to store new paths
    #     for ind, next_token_prob in enumerate(next_token_probs):
    #         curr_char = self.ind2char[ind]
    #         for (prefix, last_char), v in dp.items():
    #             if last_char == curr_char:
    #                 new_prefix = prefix # if last char is the same, no need to add new char
    #             else:
    #                 if curr_char != self.EMPTY_TOK:
    #                     new_prefix = prefix + curr_char
    #                 else:
    #                     new_prefix = prefix # if current char is empty token, no need to add it 
                        
    #             new_dp[(new_prefix, curr_char)] += v * next_token_prob # sum probabilities of the same path
    #     return new_dp
    
    # def truncate_paths(self, dp, beam_size):
    #     dp = dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size]) # sort paths by probability
    #     return dp
    
    # def _expand_and_merge_path_with_lm(self, dp, next_token_probs):
    #     new_dp = {}
        
    #     for ind, next_token_prob in enumerate(next_token_probs):
    #         curr_char = self.ind2char[ind]
            
    #         for (prefix, last_char), (ctc_prob, lm_score) in dp.items():
    #             if last_char == curr_char:
    #                 new_prefix = prefix
    #             else:
    #                 if curr_char != self.EMPTY_TOK:
    #                     new_prefix = prefix + curr_char
    #                     # Update LM score only when adding new char
    #                     if self.lm is not None:
    #                         lm_score = self.score_with_lm(new_prefix)
    #                 else:
    #                     new_prefix = prefix
                
    #             key = (new_prefix, curr_char)
    #             new_ctc_prob = ctc_prob * next_token_prob
                
    #             if key not in new_dp or new_ctc_prob > new_dp[key][0]:
    #                 new_dp[key] = (new_ctc_prob, lm_score)
                    
    #     return new_dp

    # def _truncate_paths_with_lm(self, dp, beam_size):
    #     # Combine CTC and LM scores for sorting
    #     scored_paths = []
    #     for (text, last_char), (ctc_prob, lm_score) in dp.items():
    #         combined_score = (1 - self.lm_weight) * ctc_prob + self.lm_weight * lm_score
    #         scored_paths.append(((text, last_char), (ctc_prob, lm_score), combined_score))
        
    #     # Sort by combined score
    #     scored_paths.sort(key=lambda x: x[2], reverse=True)
        
    #     # Keep top beam_size paths
    #     return {path[0]: path[1] for path in scored_paths[:beam_size]}

    # def score_with_lm(self, text: str) -> float:
    #     """
    #     Score a text using the language model
        
    #     Args:
    #         text (str): text to score
    #     Returns:
    #         float: language model score (log probability)
    #     """
    #     if self.lm is None:
    #         return 0.0
    #     return self.lm.score(text)
    
    
    # def ctc_beam_search_with_lm(self, probs: torch.Tensor, beam_size: int = 10) -> List[Tuple[str, float]]:
    #     """
    #     Beam search with language model scoring - works with both BPE and char-level
    #     """
    #     # Initial beam with empty string
    #     dp = {("", self.EMPTY_TOK): (1.0, 0.0)}  # (text, last_token): (ctc_prob, lm_score)
        
    #     for prob in probs:
    #         dp = self._expand_and_merge_path_with_lm(dp, prob)
    #         dp = self._truncate_paths_with_lm(dp, beam_size)
        
    #     # Final scoring combining CTC and LM scores
    #     final_beams = []
    #     for (text, _), (ctc_prob, lm_score) in dp.items():
    #         # Clean up BPE tokens if needed
    #         if self.use_bpe:
    #             text = self.tokenizer.clean_up_tokenization(text)
            
    #         combined_score = (1 - self.lm_weight) * ctc_prob + self.lm_weight * lm_score
    #         final_beams.append((text, combined_score))
        
    #     # Sort by combined score
    #     final_beams.sort(key=lambda x: x[1], reverse=True)
    #     return final_beams[:beam_size]

    # def _expand_and_merge_path_with_lm(self, dp, next_token_probs):
    #     new_dp = {}
        
    #     for ind, next_token_prob in enumerate(next_token_probs):
    #         curr_token = self.ind2char[ind]
            
    #         for (prefix, last_token), (ctc_prob, lm_score) in dp.items():
    #             if last_token == curr_token:
    #                 new_prefix = prefix
    #             else:
    #                 if curr_token != self.EMPTY_TOK:
    #                     if self.use_bpe:
    #                         # For BPE, add space between tokens
    #                         new_prefix = prefix + " " + curr_token if prefix else curr_token
    #                     else:
    #                         # For char-level, concatenate directly
    #                         new_prefix = prefix + curr_token
                        
    #                     # Update LM score only when adding new token
    #                     if self.lm is not None:
    #                         # Clean up BPE tokens before LM scoring
    #                         if self.use_bpe:
    #                             score_text = self.tokenizer.clean_up_tokenization(new_prefix)
    #                         else:
    #                             score_text = new_prefix
    #                         lm_score = self.score_with_lm(score_text)
    #                 else:
    #                     new_prefix = prefix
                
    #             key = (new_prefix, curr_token)
    #             new_ctc_prob = ctc_prob * next_token_prob
                
    #             if key not in new_dp or new_ctc_prob > new_dp[key][0]:
    #                 new_dp[key] = (new_ctc_prob, lm_score)
                    
    #     return new_dp

    # def _truncate_paths_with_lm(self, dp, beam_size):
    #     # Combine CTC and LM scores for sorting
    #     scored_paths = []
    #     for (text, last_token), (ctc_prob, lm_score) in dp.items():
    #         # Clean up BPE tokens for scoring if needed
    #         if self.use_bpe:
    #             score_text = self.tokenizer.clean_up_tokenization(text)
    #         else:
    #             score_text = text
                
    #         combined_score = (1 - self.lm_weight) * ctc_prob + self.lm_weight * lm_score
    #         scored_paths.append(((text, last_token), (ctc_prob, lm_score), combined_score))
        
    #     # Sort by combined score
    #     scored_paths.sort(key=lambda x: x[2], reverse=True)
        
    #     # Keep top beam_size paths
    #     return {path[0]: path[1] for path in scored_paths[:beam_size]}

    # def score_with_lm(self, text: str) -> float:
    #     """
    #     Score a text using the language model
    #     Works with both BPE and char-level
    #     """
    #     if self.lm is None:
    #         return 0.0
            
    #     if self.use_bpe:
    #         # Ensure proper formatting for BPE tokens
    #         text = self.tokenizer.clean_up_tokenization(text)
    #         # You might need additional normalization here
            
    #     return self.lm.score(text)
    
    # NEW
    
    # def ctc_beam_search(self, probs, beam_size=10, use_lm=False):
    #     """
    #     Beam search implementation for CTC decoding, optionally with LM scoring.
    #     Args:
    #         probs (Tensor): probabilities from the model
    #         beam_size (int): beam size
    #         use_lm (bool): whether to use language model scoring
    #     Returns:
    #         list of tuples (text, score)
    #     """
    #     if use_lm:
    #         # Initialize with CTC prob and LM score
    #         dp = {("", self.EMPTY_TOK): (1.0, 0.0)}  # (text, last_char): (ctc_prob, lm_score)
    #     else:
    #         # Initialize with just CTC prob
    #         dp = {("", self.EMPTY_TOK): 1.0}  # (text, last_char): ctc_prob
        
    #     for prob in probs:
    #         if use_lm:
    #             dp = self._expand_and_merge_path_with_lm(dp, prob)
    #             dp = self._truncate_paths_with_lm(dp, beam_size)
    #         else:
    #             dp = self.expand_and_merge_path(dp, prob)
    #             dp = self.truncate_paths(dp, beam_size)
        
    #     if use_lm:
    #         # Combine CTC and LM scores for final result
    #         final_beams = []
    #         for (text, _), (ctc_prob, lm_score) in dp.items():
    #             combined_score = (1 - self.lm_weight) * ctc_prob + self.lm_weight * lm_score
    #             final_beams.append((text, combined_score))
    #         final_beams.sort(key=lambda x: x[1], reverse=True)
    #         return final_beams[:beam_size]
    #     else:
    #         # Return regular beam search results
    #         return [(prefix, prob) for (prefix, _), prob in sorted(dp.items(), key=lambda x: -x[1])][:beam_size]
    
    
    # NEW with debugging
    # def ctc_beam_search(self, probs, beam_size=10, use_lm=False, debug=True):
    #     if debug:
    #         print(f"\nInput probs shape: {probs.shape}")
    #         print(f"Prob range: min={probs.min():.4f}, max={probs.max():.4f}")
        
    #     if use_lm:
    #         dp = {("", self.EMPTY_TOK): (1.0, 0.0)}
    #     else:
    #         dp = {("", self.EMPTY_TOK): 1.0}
        
    #     for t, prob in enumerate(probs):
    #         if debug and t < 3:  # Print first 3 timesteps
    #             print(f"\nTimestep {t}:")
    #             top_k = torch.topk(prob, k=5)
    #             print("Top 5 tokens and probs:")
    #             for i, (p, idx) in enumerate(zip(top_k.values, top_k.indices)):
    #                 token = self.ind2char[idx.item()]
    #                 print(f"{token}: {p:.4f}")
            
    #         # Existing expansion code...
    #         if use_lm:
    #             dp = self._expand_and_merge_path_with_lm(dp, prob)
    #             dp = self._truncate_paths_with_lm(dp, beam_size)
    #         else:
    #             dp = self.expand_and_merge_path(dp, prob)
    #             dp = self.truncate_paths(dp, beam_size)
            
    #         if debug and t < 3:  # Print beam state for first 3 timesteps
    #             print("\nCurrent beam:")
    #             for (text, last_char), score in list(dp.items())[:3]:
    #                 if use_lm:
    #                     ctc_prob, lm_score = score
    #                     print(f"Text: '{text}', Last: '{last_char}', CTC: {ctc_prob:.4f}, LM: {lm_score:.4f}")
    #                 else:
    #                     print(f"Text: '{text}', Last: '{last_char}', Score: {score:.4f}")

    #     # Format final results
    #     if use_lm:
    #         final_beams = []
    #         for (text, _), (ctc_prob, lm_score) in dp.items():
    #             combined_score = (1 - self.lm_weight) * ctc_prob + self.lm_weight * lm_score
    #             final_beams.append((text, combined_score))
    #             if debug:
    #                 print(f"\nFinal beam: Text='{text}'")
    #                 print(f"CTC={ctc_prob:.4f}, LM={lm_score:.4f}, Combined={combined_score:.4f}")
    #         final_beams.sort(key=lambda x: x[1], reverse=True)
    #         return final_beams[:beam_size]
    #     else:
    #         final_beams = [(prefix, prob) for (prefix, _), prob in sorted(dp.items(), key=lambda x: -x[1])][:beam_size]
    #         if debug:
    #             print("\nFinal beams:")
    #             for text, score in final_beams[:3]:
    #                 print(f"Text: '{text}', Score: {score:.4f}")
    #         return final_beams
    
    # NEW with debugging - V2
    # def ctc_beam_search(self, probs, beam_size=10, use_lm=False, debug = True):
    #     """Use log probabilities to prevent underflow"""
    #     if use_lm:
    #         dp = {("", self.EMPTY_TOK): (0.0, 0.0)}  # Using log probs (0.0 = log(1.0))
    #     else:
    #         dp = {("", self.EMPTY_TOK): 0.0}  # Using log probs
        
    #     for t, prob in enumerate(probs):
    #         if debug and t < 3:  # Print first 3 timesteps
    #             print(f"\nTimestep {t}:")
    #             top_k = torch.topk(prob, k=5)
    #             print("Top 5 tokens and probs:")
    #             for i, (p, idx) in enumerate(zip(top_k.values, top_k.indices)):
    #                 token = self.ind2char[idx.item()]
    #                 print(f"{token}: {p:.4f}")
        
        
    #     # Convert to log probabilities if not already
    #     log_probs = torch.log(probs + 1e-8)  # Add small epsilon to avoid log(0)
        
    #     for prob in log_probs:
    #         if use_lm:
    #             dp = self._expand_and_merge_path_with_lm(dp, prob)
    #             dp = self._truncate_paths_with_lm(dp, beam_size)
    #         else:
    #             dp = self.expand_and_merge_path(dp, prob)
    #             dp = self.truncate_paths(dp, beam_size)
            
    #         if debug and t < 3:  # Print beam state for first 3 timesteps
    #             print("\nCurrent beam:")
    #             for (text, last_char), score in list(dp.items())[:3]:
    #                 if use_lm:
    #                     ctc_prob, lm_score = score
    #                     print(f"Text: '{text}', Last: '{last_char}', CTC: {ctc_prob:.4f}, LM: {lm_score:.4f}")
    #                 else:
    #                     print(f"Text: '{text}', Last: '{last_char}', Score: {score:.4f}")

    #     # Format final results
    #     if use_lm:
    #         final_beams = []
    #         for (text, _), (ctc_prob, lm_score) in dp.items():
    #             combined_score = (1 - self.lm_weight) * ctc_prob + self.lm_weight * lm_score
    #             final_beams.append((text, combined_score))
    #             if debug:
    #                 print(f"\nFinal beam: Text='{text}'")
    #                 print(f"CTC={ctc_prob:.4f}, LM={lm_score:.4f}, Combined={combined_score:.4f}")
    #         final_beams.sort(key=lambda x: x[1], reverse=True)
    #         return final_beams[:beam_size]
    #     else:
    #         final_beams = [(prefix, prob) for (prefix, _), prob in sorted(dp.items(), key=lambda x: -x[1])][:beam_size]
    #         if debug:
    #             print("\nFinal beams:")
    #             for text, score in final_beams[:3]:
    #                 print(f"Text: '{text}', Score: {score:.4f}")
    #         return final_beams
    
    # NEW with debugging - V3
    def ctc_beam_search(self, probs, beam_size=50, use_lm=False, debug=True): # beam_size = 10 default
        """Use log probabilities to prevent underflow with improved LM integration"""
        if use_lm:
            dp = {("", self.EMPTY_TOK): (0.0, 0.0)}  # Using log probs (0.0 = log(1.0))
        else:
            dp = {("", self.EMPTY_TOK): 0.0}  # Using log probs
        
        # Convert to log probabilities if not already
        log_probs = torch.log(probs + 1e-8)  # Add small epsilon to avoid log(0)
        
        for t, prob in enumerate(log_probs):
            if debug and t < 3:  # Print first 3 timesteps
                print(f"\nTimestep {t}:")
                top_k = torch.topk(torch.exp(prob), k=5)  # Convert back to prob for printing
                print("Top 5 tokens and probs:")
                for i, (p, idx) in enumerate(zip(top_k.values, top_k.indices)):
                    token = self.ind2char[idx.item()]
                    print(f"{token}: {p:.4f}")
            
            if use_lm:
                dp = self._expand_and_merge_path_with_lm(dp, prob)
                dp = self._truncate_paths_with_lm(dp, beam_size)
            else:
                dp = self.expand_and_merge_path(dp, prob)
                
                # Normalize scores periodically to prevent underflow
                if len(dp) > 0:
                    max_score = max(score for _, score in dp.items())
                    dp = {key: score - max_score for key, score in dp.items()}
                
                dp = self.truncate_paths(dp, beam_size)
            
            if debug and t < 3:  # Print beam state for first 3 timesteps
                print("\nCurrent beam:")
                for (text, last_char), score in list(dp.items())[:3]:
                    if use_lm:
                        ctc_prob, lm_score = score
                        print(f"Text: '{text}', Last: '{last_char}', CTC: {ctc_prob:.4f}, LM: {lm_score:.4f}")
                    else:
                        print(f"Text: '{text}', Last: '{last_char}', Score: {score:.4f}")

        # Format final results with length normalization
        if use_lm:
            final_beams = []
            for (text, _), (ctc_prob, lm_score) in dp.items():
                # Clean up text
                text = ' '.join(text.split())
                
                # Normalize scores by length
                text_len = max(1, len(text.split()))  # word count, avoid division by zero
                normalized_ctc = ctc_prob / text_len
                normalized_lm = lm_score / text_len
                
                # Combine normalized scores with length penalty
                length_penalty = ((5 + text_len) / 6) ** 0.65
                combined_score = ((1 - self.lm_weight) * normalized_ctc + 
                                self.lm_weight * normalized_lm) / length_penalty
                
                final_beams.append((text, combined_score))
                if debug:
                    print(f"\nFinal beam: Text='{text}'")
                    print(f"CTC={ctc_prob:.4f}, LM={lm_score:.4f}, "
                            f"Normalized Combined={combined_score:.4f}")
            
            final_beams.sort(key=lambda x: x[1], reverse=True)
            return final_beams[:beam_size]
        else:
            final_beams = []
            for (text, _), score in sorted(dp.items(), key=lambda x: -x[1])[:beam_size]:
                # Clean up text
                text = ' '.join(text.split())
                # Add simple length normalization for consistency
                text_len = max(1, len(text.split()))
                normalized_score = score / text_len
                final_beams.append((text, normalized_score))
                
            if debug:
                print("\nFinal beams:")
                for text, score in final_beams[:3]:
                    print(f"Text: '{text}', Score: {score:.4f}")
            return final_beams
    
    
    # A bit better version
    # def ctc_beam_search(self, probs, beam_size=10, use_lm=False, debug=True):
    #     """Use log probabilities to prevent underflow with score normalization"""
    #     if use_lm:
    #         dp = {("", self.EMPTY_TOK): (0.0, 0.0)}  # Using log probs (0.0 = log(1.0))
    #     else:
    #         dp = {("", self.EMPTY_TOK): 0.0}  # Using log probs
        
    #     # Convert to log probabilities if not already
    #     log_probs = torch.log(probs + 1e-8)  # Add small epsilon to avoid log(0)
        
    #     for t, prob in enumerate(log_probs):
    #         if debug and t < 3:  # Print first 3 timesteps
    #             print(f"\nTimestep {t}:")
    #             top_k = torch.topk(torch.exp(prob), k=5)  # Convert back to prob for printing
    #             print("Top 5 tokens and probs:")
    #             for i, (p, idx) in enumerate(zip(top_k.values, top_k.indices)):
    #                 token = self.ind2char[idx.item()]
    #                 print(f"{token}: {p:.4f}")
            
    #         if use_lm:
    #             dp = self._expand_and_merge_path_with_lm(dp, prob)
    #             dp = self._truncate_paths_with_lm(dp, beam_size)
    #         else:
    #             dp = self.expand_and_merge_path(dp, prob)
                
    #             # Normalize scores periodically to prevent underflow
    #             if len(dp) > 0:
    #                 max_score = max(score for _, score in dp.items())
    #                 dp = {key: score - max_score for key, score in dp.items()}
                
    #             dp = self.truncate_paths(dp, beam_size)
            
    #         if debug and t < 3:  # Print beam state for first 3 timesteps
    #             print("\nCurrent beam:")
    #             for (text, last_char), score in list(dp.items())[:3]:
    #                 if use_lm:
    #                     ctc_prob, lm_score = score
    #                     print(f"Text: '{text}', Last: '{last_char}', CTC: {ctc_prob:.4f}, LM: {lm_score:.4f}")
    #                 else:
    #                     print(f"Text: '{text}', Last: '{last_char}', Score: {score:.4f}")

    #     # Format final results
    #     if use_lm:
    #         final_beams = []
    #         for (text, _), (ctc_prob, lm_score) in dp.items():
    #             combined_score = (1 - self.lm_weight) * ctc_prob + self.lm_weight * lm_score
    #             text = ' '.join(text.split())  # Basic space cleanup
    #             final_beams.append((text, combined_score))
    #             if debug:
    #                 print(f"\nFinal beam: Text='{text}'")
    #                 print(f"CTC={ctc_prob:.4f}, LM={lm_score:.4f}, Combined={combined_score:.4f}")
    #         final_beams.sort(key=lambda x: x[1], reverse=True)
    #         return final_beams[:beam_size]
    #     else:
    #         final_beams = []
    #         for (text, _), score in sorted(dp.items(), key=lambda x: -x[1])[:beam_size]:
    #             text = ' '.join(text.split())  # Basic space cleanup
    #             final_beams.append((text, score))
    #         if debug:
    #             print("\nFinal beams:")
    #             for text, score in final_beams[:3]:
    #                 print(f"Text: '{text}', Score: {score:.4f}")
    #         return final_beams
        
    
    # def expand_and_merge_path(self, dp, next_token_probs):
    #     """Regular beam search path expansion"""
    #     new_dp = defaultdict(float)
    #     for ind, next_token_prob in enumerate(next_token_probs):
    #         curr_char = self.ind2char[ind]
    #         for (prefix, last_char), v in dp.items():
    #             if last_char == curr_char:
    #                 new_prefix = prefix
    #             else:
    #                 if curr_char != self.EMPTY_TOK:
    #                     new_prefix = prefix + curr_char
    #                 else:
    #                     new_prefix = prefix
    #             new_dp[(new_prefix, curr_char)] += v * next_token_prob
    #     return new_dp

    # def _expand_and_merge_path_with_lm(self, dp, next_token_probs):
    #     """Beam search path expansion with LM scoring"""
    #     new_dp = {}
    #     for ind, next_token_prob in enumerate(next_token_probs):
    #         curr_char = self.ind2char[ind]
    #         for (prefix, last_char), (ctc_prob, lm_score) in dp.items():
    #             if last_char == curr_char:
    #                 new_prefix = prefix
    #             else:
    #                 if curr_char != self.EMPTY_TOK:
    #                     new_prefix = prefix + curr_char
    #                     # Update LM score only when adding new char
    #                     if self.lm is not None:
    #                         lm_score = self.score_with_lm(new_prefix)
    #                 else:
    #                     new_prefix = prefix

    #             key = (new_prefix, curr_char)
    #             new_ctc_prob = ctc_prob * next_token_prob
                
    #             if key not in new_dp or new_ctc_prob > new_dp[key][0]:
    #                 new_dp[key] = (new_ctc_prob, lm_score)
    #     return new_dp
    
    
    # UPDATED FOR LOG PROBABILITIES
    def expand_and_merge_path(self, dp, next_token_log_probs):
        """Work with log probabilities and handle spaces better"""
        new_dp = defaultdict(lambda: float('-inf'))  # Log prob initialization
        
        for ind, next_token_log_prob in enumerate(next_token_log_probs):
            curr_char = self.ind2char[ind]
            
            for (prefix, last_char), log_prob in dp.items():
                if last_char == curr_char and curr_char != " ":  # Avoid repeated chars
                    new_prefix = prefix
                else:
                    if curr_char != self.EMPTY_TOK:
                        if curr_char == " " and prefix.endswith(" "):
                            continue  # Skip consecutive spaces
                        new_prefix = prefix + curr_char
                    else:
                        new_prefix = prefix
                        
                new_log_prob = log_prob + next_token_log_prob  # Add log probs
                new_dp[(new_prefix, curr_char)] = max(
                    new_dp[(new_prefix, curr_char)],
                    new_log_prob
                )
        
        return new_dp
    
    # a bit better version
    # def expand_and_merge_path(self, dp, next_token_log_probs):
    #     """Enhanced path expansion with better character handling"""
    #     new_dp = defaultdict(lambda: float('-inf'))
        
    #     for ind, next_token_log_prob in enumerate(next_token_log_probs):
    #         curr_char = self.ind2char[ind]
            
    #         for (prefix, last_char), log_prob in dp.items():
    #             # Skip if trying to add third consecutive character
    #             if len(prefix) >= 2 and prefix[-2:] == curr_char * 2:
    #                 continue
                    
    #             if last_char == curr_char and curr_char != " ":
    #                 new_prefix = prefix
    #             else:
    #                 if curr_char != self.EMPTY_TOK:
    #                     if curr_char == " " and prefix.endswith(" "):
    #                         continue
    #                     new_prefix = prefix + curr_char
    #                 else:
    #                     new_prefix = prefix
                
    #             new_log_prob = log_prob + next_token_log_prob
                
    #             # Suppress improbable transitions
    #             if new_log_prob < -100:  # Threshold for very low probability paths
    #                 continue
                    
    #             new_dp[(new_prefix, curr_char)] = max(
    #                 new_dp[(new_prefix, curr_char)],
    #                 new_log_prob
    #             )
        
    #     return new_dp

    # def _expand_and_merge_path_with_lm(self, dp, next_token_log_probs):
    #     """Similar changes for LM version"""
    #     new_dp = {}
        
    #     for ind, next_token_log_prob in enumerate(next_token_log_probs):
    #         curr_char = self.ind2char[ind]
            
    #         for (prefix, last_char), (ctc_log_prob, lm_score) in dp.items():
    #             if last_char == curr_char and curr_char != " ":
    #                 new_prefix = prefix
    #             else:
    #                 if curr_char != self.EMPTY_TOK:
    #                     if curr_char == " " and prefix.endswith(" "):
    #                         continue
    #                     new_prefix = prefix + curr_char
    #                     if self.lm is not None:
    #                         lm_score = self.score_with_lm(new_prefix)
    #                 else:
    #                     new_prefix = prefix
                
    #             new_ctc_log_prob = ctc_log_prob + next_token_log_prob
                
    #             key = (new_prefix, curr_char)
    #             if key not in new_dp or new_ctc_log_prob > new_dp[key][0]:
    #                 new_dp[key] = (new_ctc_log_prob, lm_score)
        
    #     return new_dp
    
    # UPDATED LM VERSION
    # def _expand_and_merge_path_with_lm(self, dp, next_token_log_probs):
    #     """Improved LM integration"""
    #     new_dp = {}
        
    #     for ind, next_token_log_prob in enumerate(next_token_log_probs):
    #         curr_char = self.ind2char[ind]
            
    #         for (prefix, last_char), (ctc_log_prob, lm_score) in dp.items():
    #             if last_char == curr_char and curr_char != " ":
    #                 new_prefix = prefix
    #                 new_lm_score = lm_score  # Keep previous LM score
    #             else:
    #                 if curr_char != self.EMPTY_TOK:
    #                     if curr_char == " " and prefix.endswith(" "):
    #                         continue
    #                     new_prefix = prefix + curr_char
                        
    #                     # Update LM score only on word boundaries
    #                     if curr_char == " " and self.lm is not None:
    #                         last_word = prefix.split()[-1] if prefix.split() else ""
    #                         if last_word:
    #                             new_lm_score = self.score_with_lm(last_word)
    #                     else:
    #                         new_lm_score = lm_score  # Keep previous score
    #                 else:
    #                     new_prefix = prefix
    #                     new_lm_score = lm_score
                
    #             new_ctc_log_prob = ctc_log_prob + next_token_log_prob
                
    #             key = (new_prefix, curr_char)
    #             if key not in new_dp or new_ctc_log_prob > new_dp[key][0]:
    #                 new_dp[key] = (new_ctc_log_prob, new_lm_score)
        
    #     return new_dp

    # def _expand_and_merge_path_with_lm(self, dp, next_token_log_probs):
    #     """Fix LM scoring"""
    #     new_dp = {}
        
    #     for ind, next_token_log_prob in enumerate(next_token_log_probs):
    #         curr_char = self.ind2char[ind]
            
    #         for (prefix, last_char), (ctc_log_prob, lm_score) in dp.items():
    #             if last_char == curr_char and curr_char != " ":
    #                 new_prefix = prefix
    #                 new_lm_score = lm_score
    #             else:
    #                 if curr_char != self.EMPTY_TOK:
    #                     if curr_char == " " and prefix.endswith(" "):
    #                         continue
    #                     new_prefix = prefix + curr_char
                        
    #                     # Update LM score on complete words
    #                     if curr_char == " " and self.lm is not None:
    #                         words = new_prefix.strip().split()
    #                         if words:
    #                             # Score the whole sentence
    #                             new_lm_score = self.score_with_lm(" ".join(words))
    #                     else:
    #                         new_lm_score = lm_score
    #                 else:
    #                     new_prefix = prefix
    #                     new_lm_score = lm_score
                
    #             new_ctc_log_prob = ctc_log_prob + next_token_log_prob
                
    #             key = (new_prefix, curr_char)
    #             if key not in new_dp or new_ctc_log_prob > new_dp[key][0]:
    #                 new_dp[key] = (new_ctc_log_prob, new_lm_score)
        
    #     return new_dp
    
    # Updated after LM testing
    # def score_with_lm(self, text: str) -> float:
    #     """
    #     Enhanced LM scoring that better differentiates between phrases
    #     """
    #     if self.lm is None:
    #         return 0.0
        
    #     # Handle edge cases
    #     if not text or len(text.strip()) == 0:
    #         return float('-inf')
            
    #     # Normalize text
    #     text = text.lower().strip()
    #     words = text.split()
        
    #     if len(words) <= 1:
    #         return self.lm.score(text) - 5.0  # Single word penalty
            
    #     # Score full sentence
    #     full_score = self.lm.score(text)
        
    #     # Score individual bigrams and trigrams
    #     bigram_scores = []
    #     trigram_scores = []
        
    #     for i in range(len(words) - 1):
    #         bigram = " ".join(words[i:i+2])
    #         bigram_scores.append(self.lm.score(bigram))
            
    #         if i < len(words) - 2:
    #             trigram = " ".join(words[i:i+3])
    #             trigram_scores.append(self.lm.score(trigram))
        
    #     # Combine scores
    #     avg_bigram_score = sum(bigram_scores) / len(bigram_scores) if bigram_scores else 0
    #     avg_trigram_score = sum(trigram_scores) / len(trigram_scores) if trigram_scores else 0
        
    #     # Weight the different scores
    #     final_score = (0.4 * full_score + 
    #                 0.3 * avg_bigram_score + 
    #                 0.3 * avg_trigram_score)
        
    #     # Apply penalties
    #     invalid_chars = len([c for c in text if not c.isalnum() and c != ' '])
    #     if invalid_chars > 0:
    #         final_score -= (invalid_chars * 0.5)
        
    #     # Normalize by length
    #     return final_score / len(words)

    
    # Updated for ARPA 3-gram model
    # def score_with_lm(self, text: str) -> float:
    #     """
    #     Score text using language model with state tracking
    #     """
    #     if self.lm is None:
    #         return 0.0
        
    #     # Handle empty text
    #     if not text or len(text.strip()) == 0:
    #         return float('-inf')
        
    #     # Normalize text
    #     text = text.lower().strip()
        
    #     if self.use_bpe:
    #         text = self.tokenizer.clean_up_tokenization(text)
        
    #     # Initialize scoring state
    #     state = kenlm.State()
    #     self.lm.BeginSentenceWrite(state)
    #     total_score = 0.0
        
    #     # Score each word in context
    #     words = text.split()
    #     for i, word in enumerate(words):
    #         out_state = kenlm.State()
    #         score = self.lm.BaseScore(state, word, out_state)
    #         total_score += score
    #         state = out_state
        
    #     # Normalize by length for fair comparison
    #     if len(words) > 0:
    #         total_score /= len(words)
        
    #     return total_score
    
    # def _expand_and_merge_path_with_lm(self, dp, next_token_log_probs):
    #     """More frequent LM score updates"""
    #     new_dp = {}
        
    #     for ind, next_token_log_prob in enumerate(next_token_log_probs):
    #         curr_char = self.ind2char[ind]
            
    #         for (prefix, last_char), (ctc_log_prob, lm_score) in dp.items():
    #             if last_char == curr_char and curr_char != " ":
    #                 new_prefix = prefix
    #                 new_lm_score = lm_score
    #             else:
    #                 if curr_char != self.EMPTY_TOK:
    #                     if curr_char == " " and prefix.endswith(" "):
    #                         continue
    #                     new_prefix = prefix + curr_char
                        
    #                     # Update LM score more aggressively
    #                     if curr_char == " " or (
    #                         len(new_prefix) >= 3 and 
    #                         any(new_prefix.endswith(word) for word in 
    #                             ['the', 'and', 'are', 'you', 'for', 'was'])
    #                     ):
    #                         new_lm_score = self.score_with_lm(new_prefix.strip())
    #                     else:
    #                         new_lm_score = lm_score
    #                 else:
    #                     new_prefix = prefix
    #                     new_lm_score = lm_score
                
    #             new_ctc_log_prob = ctc_log_prob + next_token_log_prob
                
    #             key = (new_prefix, curr_char)
    #             if key not in new_dp or new_ctc_log_prob > new_dp[key][0]:
    #                 new_dp[key] = (new_ctc_log_prob, new_lm_score)
        
    #     return new_dp
    
    # V4
    def score_with_lm(self, text: str) -> float:
        """
        Improved LM scoring using proper n-gram context
        """
        if self.lm is None:
            return 0.0
        
        if not text or len(text.strip()) == 0:
            return float('-inf')
        
        text = text.lower().strip()
        
        # Get the full score which properly uses n-gram context
        full_score = self.lm.score(text, bos=True, eos=True)  # Add beginning/end sentence tokens
        
        # Add penalties for obviously wrong sequences
        words = text.split()
        if len(words) <= 1:
            return full_score - 5.0  # Penalize very short sequences
            
        # Don't normalize by length - use raw score to maintain proper n-gram scoring
        return full_score

    def _expand_and_merge_path_with_lm(self, dp, next_token_log_probs):
        """Updated path expansion to use proper n-gram context"""
        new_dp = {}
        
        for ind, next_token_log_prob in enumerate(next_token_log_probs):
            curr_char = self.ind2char[ind]
            
            for (prefix, last_char), (ctc_log_prob, lm_score) in dp.items():
                if last_char == curr_char and curr_char != " ":
                    new_prefix = prefix
                    new_lm_score = lm_score
                else:
                    if curr_char != self.EMPTY_TOK:
                        if curr_char == " " and prefix.endswith(" "):
                            continue
                        new_prefix = prefix + curr_char
                        
                        # Update LM score using full context
                        if curr_char == " " or new_prefix.endswith((" a ", " the ", " in ", " of ")):
                            new_lm_score = self.lm.score(new_prefix.strip(), bos=True, eos=False)
                        else:
                            new_lm_score = lm_score
                    else:
                        new_prefix = prefix
                        new_lm_score = lm_score
                
                new_ctc_log_prob = ctc_log_prob + next_token_log_prob
                key = (new_prefix, curr_char)
                
                if key not in new_dp or new_ctc_log_prob > new_dp[key][0]:
                    new_dp[key] = (new_ctc_log_prob, new_lm_score)
        
        return new_dp
    
    def truncate_paths(self, dp, beam_size):
        """Regular beam search truncation"""
        return dict(sorted(dp.items(), key=lambda x: -x[1])[:beam_size])

    # def _truncate_paths_with_lm(self, dp, beam_size):
    #     """Beam search truncation with LM scoring"""
    #     scored_paths = []
    #     for (text, last_char), (ctc_prob, lm_score) in dp.items():
    #         combined_score = (1 - self.lm_weight) * ctc_prob + self.lm_weight * lm_score
    #         scored_paths.append(((text, last_char), (ctc_prob, lm_score), combined_score))
    #     scored_paths.sort(key=lambda x: x[2], reverse=True)
    #     return {path[0]: path[1] for path in scored_paths[:beam_size]}
    
    
    # Updated LM version
    def _truncate_paths_with_lm(self, dp, beam_size):
        """Improved score combination"""
        scored_paths = []
        for (text, last_char), (ctc_prob, lm_score) in dp.items():
            # Scale LM score to be more comparable with CTC score
            scaled_lm_score = lm_score / (len(text.split()) + 1)  # Normalize by word count
            combined_score = (1 - self.lm_weight) * ctc_prob + self.lm_weight * scaled_lm_score
            scored_paths.append(((text, last_char), (ctc_prob, lm_score), combined_score))
        scored_paths.sort(key=lambda x: x[2], reverse=True)
        return {path[0]: path[1] for path in scored_paths[:beam_size]}
    
    
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