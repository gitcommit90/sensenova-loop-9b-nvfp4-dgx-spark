"""Build FR-Spec style hot-vocab maps for the MTP draft lm_head from TRAIN-split corpora only (calib/raw):
oasst2 (multilingual chat, all messages), glaive-function-calling-v2 (tool calls), mbpp (code), gsm8k train (math).
The target model always verifies against the FULL vocabulary; this only restricts which tokens the draft can propose.
Output: hot_<K>.pt = sorted int64 tensor of token ids (special/added tokens always included) + stats json."""
import json, collections, sys, re, argparse
from pathlib import Path
import pyarrow.parquet as pq, torch
from tokenizers import Tokenizer
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--raw-dir", required=True, help="Pinned corpus tree used by quantization/build_calib.py")
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = str(Path(a.raw_dir).resolve()) + "/"; OUT = a.out
Path(OUT).mkdir(parents=True, exist_ok=True)
tok = Tokenizer.from_file(a.tokenizer)
texts = {}
texts["oasst2"] = [t for t in pq.read_table(R + "OpenAssistant__oasst2/data/train-00000-of-00001-88ba0162028a73fc.parquet", columns=["text"]).column("text").to_pylist() if t]
mb = pq.read_table(R + "google-research-datasets__mbpp/full/train-00000-of-00001.parquet").to_pylist()
texts["mbpp"] = [r["text"] + "\n" + r["code"] for r in mb]
gs = pq.read_table(R + "openai__gsm8k/main/train-00000-of-00001.parquet").to_pylist()
texts["gsm8k"] = [r["question"] + "\n" + r["answer"] for r in gs]
gl = json.load(open(R + "glaiveai__glaive-function-calling-v2/glaive-function-calling-v2.json"))
texts["glaive"] = [r["system"] + "\n" + r["chat"] for r in gl[:40000]]
cnt = collections.Counter(); per = {}
for k, ts in texts.items():
    c = collections.Counter()
    for i in range(0, len(ts), 2000):
        for e in tok.encode_batch(ts[i:i + 2000], add_special_tokens=False): c.update(e.ids)
    per[k] = sum(c.values())
    # weight sources so the big tool corpus does not swamp chat/code/math: normalize each to equal mass
    tot = sum(c.values())
    for t, n in c.items(): cnt[t] += n / tot
special = sorted(tok.get_added_tokens_decoder().keys())
ranked = [t for t, _ in cnt.most_common()]
stats = {"tokens_per_source": per, "distinct_tokens_seen": len(ranked), "n_special": len(special), "vocab": tok.get_vocab_size()}
for K in (32768, 65536):
    ids = set(special)
    for t in ranked:
        if len(ids) >= K: break
        ids.add(t)
    ids = torch.tensor(sorted(ids), dtype=torch.int64)
    torch.save(ids, f"{OUT}/hot_{K // 1024}k.pt")
    cov = sum(cnt[t] for t in ids.tolist()) / sum(cnt.values())
    stats[f"hot_{K // 1024}k"] = {"n": len(ids), "weighted_coverage_of_corpus": round(cov, 5)}
json.dump(stats, open(f"{OUT}/token_map_stats.json", "w"), indent=1); print(stats)
