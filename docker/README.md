# Docker assets

## Build order

```bash
# 1. SM120 DeepGEMM on top of the stock upstream image
docker build -f Dockerfile.vllm027-gb10 -t vllm027-gb10:sm120 .

# 2. Bake the DSpark patches (context must contain patches/patch-dspark-sm120.py)
cp ../patches/patch-dspark-sm120.py .
docker build -f Dockerfile.vllm027-patched -t vllm027-gb10:patched .
```

**Build on each node separately.** There is no registry between our two machines, so both images
are built locally on both. If you have a registry, push once and pull — you avoid the
"same tag, different image ID" divergence that is easy to create otherwise.

Step 1 takes a while (it compiles DeepGEMM). Step 2 is seconds.

## What to change for your machines

The compose files are **exactly what we run**, not a sanitized template. Adjust:

| Setting | Ours | What it is |
|---|---|---|
| `MASTER_ADDR` / `--master-addr` | `192.168.100.10` | head node on the CX-7 point-to-point fabric |
| `VLLM_HOST_IP` | `.10` (head) / `.11` (worker) | this node's fabric IP |
| `NODE_RANK` / `--node-rank` | `0` / `1` | head is 0 |
| `NCCL_IB_HCA` | `rocep1s0f0` | your RDMA device (`ibv_devices`) |
| `NCCL_SOCKET_IFNAME` | `enp1s0f0np0` | the CX-7 interface |
| `MN_IF_NAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME` | `enP7s7` | the *management* interface for multi-node bootstrap |
| `${HOME}/.cache/huggingface` | | model + compile/autotune cache |

Everything else — the serve flags, NCCL tuning, `VLLM_CACHE_ROOT` isolation — should transfer.

## Start order

**Worker first, then head.** The head node's `vllm serve` expects the worker to be reachable when
it initializes the process group.

```bash
ssh worker 'docker compose -f docker-compose.pollux.yml up -d'
sleep 20
ssh head   'docker compose -f docker-compose.castor.yml  up -d'
```

Boot to `/health` 200 is **~6 minutes** (~2 min of that is loading 48 safetensors shards, ~80 s is
engine init: memory profile, KV allocation, warmup, CUDA graph capture).

## Two operational warnings

- **Never run `docker compose --remove-orphans`** in a directory that also holds your previous
  stack's compose file. Compose reports the retired container as an orphan of the project and will
  delete it — that is your rollback.
- **Set the retired container's restart policy to `no`** explicitly
  (`docker update --restart no <name>`). Two containers with `unless-stopped` bound to the same
  port will race at boot, and the loser's logs will not tell you why it failed.

## Verify before declaring success

`/health` returns 200 on a build that dies at the first decode step. Always:

```bash
python3 ../bench/needle.py     # real generation, verbatim long-context recall
python3 ../bench/conc12.py     # saturation, incl. the >64-token verify pass
docker logs <container> 2>&1 | grep -icE "Check failed|Assertion"   # must be 0
docker logs <container> 2>&1 | grep -i "WARNING.*tuning bucket"     # perf cliffs
```
