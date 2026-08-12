import collections, sys
tot = 0
hits = collections.Counter()
leaf = collections.Counter()
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    stack, _, cnt = line.rpartition(" ")
    try:
        c = int(cnt)
    except ValueError:
        continue
    tot += c
    frames = stack.split(";")
    leaf[frames[-1]] += c
    for pat in ("choose_one", "maybe_load_cache", "sparse_mla_sm120", "autotune",
                "get_cache_key_extras", "decode_dsv4", "hot_cache"):
        if any(pat in f for f in frames):
            hits[pat] += c
print("total samples:", tot)
print("--- stacks containing ---")
for k, v in hits.most_common():
    print(f"  {k:24s} {v:7d}  {100*v/tot:6.2f}%")
print("--- top leaf frames ---")
for k, v in leaf.most_common(20):
    print(f"  {v:7d} {100*v/tot:6.2f}%  {k[:110]}")
