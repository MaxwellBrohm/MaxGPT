"""MaxGPT-Ultra tokenizer.

A byte-level BPE tokenizer with:
  - digit-splitting (every digit is its own token, which helps arithmetic),
  - byte fallback (all 256 bytes are in the base vocab, so nothing is ever OOV),
  - special tokens reserved up front (chat roles, tools, reasoning, plus spares),
    so chat / tool-use / thinking work without ever resizing the vocabulary.

Train one with `train_tokenizer(...)`; load and use it with `UltraTokenizer(path)`.
The real run trains vocab=49152 on a sample of the corpus; the smoke test trains a
tiny one to validate the pipeline.
"""
from __future__ import annotations

import os
from typing import Iterable

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# --- special tokens, reserved up front (added first => low, stable ids) ---
END = "<|endoftext|>"        # document separator, doubles as BOS and EOS
PAD = "<|pad|>"
IM_START = "<|im_start|>"    # ChatML turn open  (e.g. <|im_start|>user\n...)
IM_END = "<|im_end|>"        # ChatML turn close
TOOL_CALL = "<|tool_call|>"  # model is requesting a tool (search, etc.)
TOOL_RESPONSE = "<|tool_response|>"  # tool output fed back in
THINK_START = "<think>"      # reasoning scratchpad
THINK_END = "</think>"

SPECIAL_TOKENS = [
    END, PAD, IM_START, IM_END, TOOL_CALL, TOOL_RESPONSE, THINK_START, THINK_END,
] + [f"<|reserved_{i}|>" for i in range(16)]   # 16 spare slots for the future


def _byte_level_bpe() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))          # byte-level => no UNK ever
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),   # "1234" -> "1","2","3","4"
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(text_iterator: Iterable[str], vocab_size: int = 49152,
                    out_path: str = "tokenizer/maxgpt-ultra.tokenizer.json") -> Tokenizer:
    tok = _byte_level_bpe()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,                        # reserve ids 0..N
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes -> true byte fallback
        show_progress=False,
    )
    tok.train_from_iterator(text_iterator, trainer=trainer)
    tok.add_special_tokens(SPECIAL_TOKENS)   # ensure they encode atomically
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tok.save(out_path)
    return tok


class UltraTokenizer:
    """Loads a trained tokenizer and adds chat rendering + special-token ids."""

    def __init__(self, path: str):
        self.tok = Tokenizer.from_file(path)
        self.specials = {t: self.tok.token_to_id(t) for t in SPECIAL_TOKENS}
        self.eos_id = self.specials[END]
        self.bos_id = self.specials[END]   # we reuse <|endoftext|> for both
        self.pad_id = self.specials[PAD]

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special)

    def render_chat(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """messages: [{'role': 'system'|'user'|'assistant', 'content': str}, ...] -> ChatML string."""
        parts = [f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages]
        if add_generation_prompt:
            parts.append(f"{IM_START}assistant\n")
        return "".join(parts)

    def encode_chat(self, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
        return self.encode(self.render_chat(messages, add_generation_prompt))
