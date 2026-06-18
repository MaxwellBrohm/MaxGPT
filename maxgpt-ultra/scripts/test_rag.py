"""CPU smoke test for the RAG / inference layer.

Checks attachment text extraction + chunking, retrieval relevance, grounded-prompt
assembly (with the "data, not instructions" hygiene wrapper), and an end-to-end
response() call. (Generation quality from an untrained tiny model is gibberish; this
verifies the plumbing.)

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_rag.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

import torch

from model import ModelConfig, MaxGPTUltra
from tokenizer.tokenizer import train_tokenizer, UltraTokenizer
from rag.retriever import TfidfRetriever
from rag.attachments import extract_text, chunk
from rag.chat import build_context, build_prompt, respond

DOCS = [
    "Cats are small domestic felines that purr and like to chase mice around the house.",
    "Rockets burn fuel and oxidizer to produce thrust and reach orbit around the Earth.",
    "Python is a popular high-level programming language used for data science and AI.",
]


def main() -> None:
    print("=" * 72)
    print("MaxGPT-Ultra RAG / inference smoke test")
    print("=" * 72)

    print("\n[1] attachment extraction + chunking")
    d = tempfile.mkdtemp(prefix="mgu_rag_")
    paths = {}
    for name, body in {"a.txt": "plain text file about cats and dogs",
                       "b.md": "# Notes\nmarkdown about rockets",
                       "c.csv": "name,role\nmax,builder"}.items():
        p = os.path.join(d, name)
        open(p, "w").write(body)
        paths[name] = p
    assert "cats and dogs" in extract_text(paths["a.txt"])
    assert "rockets" in extract_text(paths["b.md"])
    assert "builder" in extract_text(paths["c.csv"])
    long = ("paragraph one. " * 80) + "\n" + ("paragraph two. " * 80)
    cks = chunk(long, size=300, overlap=40)
    assert len(cks) > 1 and all(len(c) <= 320 for c in cks)
    print(f"  extracted txt/md/csv; chunked a long doc into {len(cks)} pieces ✓")

    print("\n[2] retriever returns the relevant chunk")
    r = TfidfRetriever()
    r.add(DOCS)
    top_cat = r.query("domestic feline that purrs", k=1)[0][0]
    top_rocket = r.query("reach orbit burning fuel", k=1)[0][0]
    assert "Cats" in top_cat and "Rockets" in top_rocket
    print(f"  'feline purrs' -> cats doc; 'orbit fuel' -> rockets doc ✓")

    print("\n[3] grounded prompt assembly (with hygiene wrapper)")
    tok_path = "/tmp/maxgpt_ultra_rag_tok.json"
    train_tokenizer(iter(DOCS * 60 + ["the assistant answers questions about cats rockets python"] * 60),
                    vocab_size=700, out_path=tok_path)
    tok = UltraTokenizer(tok_path)
    ctx = build_context("tell me about cats", retriever=r, attachment_text="notes about dogs", k=2)
    sources = [s for s, _ in ctx]
    assert "attachment" in sources and "doc" in sources
    prompt = build_prompt(tok, "tell me about cats", ctx)
    assert "data, not instructions" in prompt and "Cats are small" in prompt and "tell me about cats" in prompt
    print(f"  context sources={sources}; prompt embeds retrieved text + the query ✓")

    print("\n[4] end-to-end response() runs")
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=96, n_layers=2, n_heads=4,
                      n_kv_heads=2, mlp_hidden=192, seq_len=64)
    model = MaxGPTUltra(cfg)
    out = respond(model, tok, "what are cats?", retriever=r, max_new_tokens=12, device="cpu")
    assert isinstance(out, str)
    print(f"  response() returned a string ({len(out)} chars) ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
