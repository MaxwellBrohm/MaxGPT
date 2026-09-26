"""CPU smoke test for SFT (supervised fine-tuning).

Checks the three things that must be right: assistant-only loss masking, the all-masked
batch can't NaN, and a tiny model actually learns to produce the assistant reply after
fine-tuning on chat data.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_sft.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

import torch

from model import ModelConfig, MaxGPTUltra, generate
from tokenizer.tokenizer import train_tokenizer, UltraTokenizer
from posttrain.sft_data import encode_chat_example, SFTDataset
from train.trainer import Trainer

TOK = "/tmp/maxgpt_ultra_sft_tok.json"
OUT = "/tmp/maxgpt_ultra_sft_run"

# enough text to train a usable tiny tokenizer that covers our chat tokens
CORPUS = ["ping pong hello world the assistant replies politely and helpfully to the user",
          "You are a terse assistant. ping pong hello world test reply done."] * 200

EXAMPLES = [
    {"messages": [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}]},
    {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]},
] * 80


def main() -> None:
    print("=" * 72)
    print("MaxGPT-Ultra SFT smoke test")
    print("=" * 72)

    train_tokenizer(iter(CORPUS), vocab_size=800, out_path=TOK)
    tok = UltraTokenizer(TOK)

    print("\n[1] assistant-only loss masking")
    toks, sup = encode_chat_example(
        [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}], tok)
    supervised = tok.decode([t for t, s in zip(toks, sup) if s])
    unsupervised = tok.decode([t for t, s in zip(toks, sup) if not s])
    print(f"  supervised tokens decode to: {supervised!r}")
    assert "pong" in supervised and "ping" not in supervised, "masking supervised the wrong tokens"
    assert "ping" in unsupervised, "user text should be unsupervised"
    print("  loss is on the assistant reply only ✓")

    print("\n[2] all-masked batch does not NaN")
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=96, n_layers=2, n_heads=4,
                      n_kv_heads=2, mlp_hidden=192, seq_len=64)     # 64: room for two packed chats in [4]/[5]
    m = MaxGPTUltra(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.full((2, 16), -100)
    _, loss = m(x, y)
    assert torch.isfinite(loss), "all-masked batch produced non-finite loss"
    print(f"  all-masked loss = {float(loss):.3f} (finite) ✓")

    print("\n[4] document mask: a packed conversation's logits ignore its neighbour entirely")
    ta, _ = encode_chat_example([{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}], tok)
    tb, _ = encode_chat_example([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}], tok)
    ta = ta + [tok.eos_id]                          # conversation A ends with the separator
    packed = torch.tensor([ta + tb])
    assert packed.size(1) <= cfg.seq_len, packed.size(1)
    m.eval()
    m.doc_mask, m.doc_sep_id = True, tok.eos_id
    with torch.no_grad():
        ref_b = m(packed)[0][0, len(ta):]
        g = torch.Generator().manual_seed(1)
        other = packed.clone()
        other[0, :len(ta) - 1] = torch.randint(0, cfg.vocab_size, (len(ta) - 1,), generator=g)   # rewrite A, keep its separator
        got_b = m(other)[0][0, len(ta):]
        assert torch.equal(ref_b, got_b), "B's logits changed when A's tokens changed: the mask leaks"
        m.doc_mask = False
        leak_b = m(other)[0][0, len(ta):]
        assert not torch.allclose(ref_b, leak_b, atol=1e-6), "without the mask B should see A (sanity)"
        m.doc_mask = True
    print(f"  {len(tb)} B positions bit-identical under a rewritten A; they differ once the mask is off ✓")

    print("\n[5] document mask: a packed conversation matches the same conversation alone (RoPE is relative)")
    with torch.no_grad():
        alone_b = m(torch.tensor([tb]))[0][0]
        assert torch.allclose(ref_b, alone_b, atol=1e-4, rtol=1e-4), float((ref_b - alone_b).abs().max())
        m.doc_mask = False
        packed_nomask_b = m(packed)[0][0, len(ta):]
        assert not torch.allclose(packed_nomask_b, alone_b, atol=1e-4, rtol=1e-4), "sanity: without the mask packing changes B"
        m.doc_mask = True
    m.train()
    print(f"  max |packed - alone| = {float((ref_b - alone_b).abs().max()):.2e} ✓")

    print("\n[6] decontamination: a conversation quoting 13 words of an eval example is dropped")
    import json as _json, tempfile
    from posttrain.sft_data import Decontaminator, ReplayBlend, clean_messages
    bench = tempfile.mkdtemp(prefix="mgu_bench_")
    ctx = "the quick brown fox jumps over the lazy dog while the cat watches from the window sill"
    with open(os.path.join(bench, "piqa.jsonl"), "w") as f:
        f.write(_json.dumps({"context": ctx, "choices": [" a", " b"], "answer": 0}) + "\n")
    dc = Decontaminator(bench)
    assert dc.sources == 1 and dc.grams
    quoted = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "Sure: " + ctx.upper() + "!"}]
    clean = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "the quick brown fox is a common typing sentence about a dog"}]
    assert dc.conversation_contaminated(quoted) and not dc.conversation_contaminated(clean)
    assert not Decontaminator("/tmp/no_such_bench_dir").is_contaminated(ctx)      # no suite -> nothing dropped
    assert clean_messages([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]) is not None
    assert clean_messages([{"role": "user", "content": "hi"}]) is None                       # must end with the assistant
    assert clean_messages([{"role": "user", "content": "hi"}, {"role": "tool", "content": "x"}, {"role": "assistant", "content": "y"}]) is None
    assert clean_messages([{"role": "user", "content": "  "}, {"role": "assistant", "content": "y"}]) is None
    print("  13-gram overlap dropped (case-insensitive), paraphrase kept, role/empty filters hold ✓")

    print("\n[7] replay blend: 25% of every batch is plain pretraining windows, state round-trips")
    from data.prepare import tokenize_to_shards
    from data.loader import PackedShardDataset
    shards7 = "/tmp/maxgpt_ultra_sft_replay_shards"
    import shutil; shutil.rmtree(shards7, ignore_errors=True)
    tokenize_to_shards(CORPUS, tok, shards7, shard_size=2048)
    blend = ReplayBlend(SFTDataset(EXAMPLES, tok, seq_len=24), PackedShardDataset(shards7, 24), frac=0.25)
    x7, y7 = blend.next_batch(8)
    assert x7.shape == (8, 24) and y7.shape == (8, 24)
    masked_rows = int((y7 == -100).any(dim=1).sum())          # SFT windows carry -100 somewhere; replay windows never do
    assert masked_rows == 6 and int(((y7 != -100).all(dim=1)).sum()) == 2, (masked_rows,)
    st7 = blend.state_dict()
    nxt = blend.next_batch(4)
    blend2 = ReplayBlend(SFTDataset(EXAMPLES, tok, seq_len=24), PackedShardDataset(shards7, 24), frac=0.25)
    blend2.load_state_dict(st7)
    again = blend2.next_batch(4)
    assert torch.equal(nxt[0], again[0]) and torch.equal(nxt[1], again[1])
    assert ReplayBlend(SFTDataset(EXAMPLES, tok, seq_len=24), PackedShardDataset(shards7, 24), frac=0.0).n_replay(8) == 0
    print("  6 SFT + 2 replay windows per batch of 8; resumed blend reproduces the next batch ✓")

    print("\n[3] model learns the assistant replies")
    ds = SFTDataset(EXAMPLES, tok, seq_len=24)
    tcfg = {"micro_batch": 8, "grad_accum": 1, "total_tokens": 24 * 8 * 200,
            "warmup_tokens": 24 * 8 * 5, "lr": 3e-3, "decay_frac": 0.1, "z_loss": 0.0,
            "autosave_minutes": 9999, "log_every": 1000}
    trainer = Trainer(m, ds, tcfg, device="cpu", out_dir=OUT, seed=0)
    for _ in range(120):
        trainer.train_step()

    def reply(user):
        prompt = tok.render_chat([{"role": "user", "content": user}], add_generation_prompt=True)
        ids = torch.tensor([tok.encode(prompt)])
        gen = generate(m, ids, max_new_tokens=8, temperature=0.0, eos_id=tok.eos_id)
        return tok.decode(gen[0, ids.shape[1]:].tolist(), skip_special=True)

    r_ping, r_hello = reply("ping"), reply("hello")
    print(f"  user 'ping'  -> assistant '{r_ping.strip()}'")
    print(f"  user 'hello' -> assistant '{r_hello.strip()}'")
    assert "pong" in r_ping and "world" in r_hello, "model did not learn the chat mapping"
    print("  learned ping->pong and hello->world ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
