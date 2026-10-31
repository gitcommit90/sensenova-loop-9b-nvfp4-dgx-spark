"""Attribute GPU kernels to each cudaGraphLaunch (via correlation id) and summarize per replay site."""
import gzip, json, sys, collections
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from trace_summary_lib import cat
d = json.load(gzip.open(sys.argv[1])); ev = [e for e in d["traceEvents"] if e.get("ph") == "X"]
launch = {e["args"].get("correlation"): e for e in ev if e.get("cat") == "cuda_runtime" and e["name"] == "cudaGraphLaunch"}
k = [e for e in ev if e.get("cat") == "kernel"]
per = collections.defaultdict(list)
for e in k:
    c = e["args"].get("correlation")
    per[c if c in launch else None].append(e)
# order launches by time; label by position in a cycle
rows = []
for c, L in launch.items():
    ks = per.get(c, [])
    if not ks: continue
    t0 = min(x["ts"] for x in ks); t1 = max(x["ts"] + x["dur"] for x in ks)
    cc = collections.Counter(); g = collections.Counter()
    for x in ks:
        cc[cat(x["name"])] += x["dur"]
        if "gemvx" in x["name"] or "wmma" in x["name"]: g[(x["name"][60:100], x["dur"] // 100 * 100)] += 1
    rows.append((L["ts"], len(ks), (t1 - t0) / 1e3, sum(x["dur"] for x in ks) / 1e3, dict((a, round(b / 1e3, 1)) for a, b in cc.most_common(5)), g.most_common(4)))
rows.sort()
for r in rows[:9]: print(f"n={r[1]:6d} span={r[2]:7.1f}ms busy={r[3]:7.1f}ms {r[4]}\n      {r[5]}")
nk = per.get(None, [])
print("non-graph kernels:", len(nk), round(sum(x['dur'] for x in nk) / 1e3, 1), "ms")
cc = collections.Counter()
for x in nk: cc[x["name"][:90]] += x["dur"]
for n, v in cc.most_common(8): print(f"   {v/1e3:7.1f} {n}")
