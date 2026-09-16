# HiCache on Intel XPU (BMG)

Hierarchical KV cache offloading for SGLang on Intel XPU. Extends effective context by spilling KV cache from device → host memory → disk.

## Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  L1: Device (GPU/XPU)     - Fastest, limited by VRAM       │
│  ↓ evict (D→H)  ↑ load_back (H→D)                          │
├─────────────────────────────────────────────────────────────┤
│  L2: Host (Pinned RAM)    - 10-50x larger than L1          │
│  ↓ spill (H→Disk)  ↑ prefetch (Disk→H)                     │
├─────────────────────────────────────────────────────────────┤
│  L3: Disk (SSD/tmpfs)     - Unlimited*, slowest            │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Minimal HiCache (L2 only, 2x host pool)
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_back

# With L3 disk tier
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_back \
    --hicache-storage-backend file
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/path/to/l3/storage
```

## Tier Controls

| Tier | Flag | Formula | Example |
|------|------|---------|---------|
| **L1 (Device)** | `--max-total-tokens N` | Direct size in tokens | `--max-total-tokens 65536` |
| **L2 (Host)** | `--hicache-ratio R` | `L2 = L1 × R` tokens | `--hicache-ratio 2.0` → 2× device |
| **L3 (Disk)** | No built-in limit | Unbounded (use tmpfs) | See below |

### L3 Disk Size Control

L3 has no built-in limit. Use a size-limited tmpfs:

```bash
# Create 20GB tmpfs for L3
sudo mount -t tmpfs -o size=20G tmpfs /mnt/hicache_l3
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/mnt/hicache_l3
```

## Write Policies

| Policy | When D→H happens | Overhead | Best for |
|--------|------------------|----------|----------|
| `write_back` | On eviction only | Low (no overhead until full) | **Production** |
| `write_through` | Every decode step | High (constant) | Debugging |

**Recommendation:** Use `write_back` for production.

## Performance Characteristics

### Benefits
- **Extended context:** Serve requests larger than device memory
- **Higher throughput:** Reuse cached prefixes across requests
- **Graceful degradation:** Falls back to recompute if cache miss

### Overhead (measured on BMG TP=2)

| Config | Overhead vs Baseline | Source |
|--------|---------------------|--------|
| HiCache OFF | 0% (baseline ~67 tok/s) | - |
| HiCache ON (optimized) | ~15% (~57 tok/s) | Per-step all_reduce sync |

**Note:** Overhead is per-step synchronization cost, not transfer cost. Transfer only happens during eviction/load_back.

### When HiCache Helps

✅ **Good fit:**
- Long conversations with shared system prompts
- Multi-turn chat with prefix reuse
- Batch processing similar documents
- Context exceeds device memory

❌ **Poor fit:**
- All requests are unique (no prefix reuse)
- Requests fit entirely in device memory
- Latency-critical single-request workloads

## Key Metrics

```bash
# Check HiCache metrics
curl -s http://localhost:30000/metrics | grep -E "hicache|evict|load_back"
```

| Metric | Meaning |
|--------|---------|
| `sglang:hicache_host_used_tokens` | Tokens currently in L2 |
| `sglang:hicache_host_total_tokens` | L2 capacity |
| `sglang:evicted_tokens_total` | Total D→H transfers |
| `sglang:load_back_tokens_total` | Total H→D restores |

## Quick Validation Tests

### Test L2 (Host Memory)

```bash
# Validates D→H backup works
bash quick_l2_validate.sh
```

Expected output:
```
HiCache host pool: 4160 tokens
Host used before: 0 tokens
...
OK: Data backed up to host (+1024 tokens)
PASS: L2 backup confirmed
```

### Test L2 Load-Back (H→D Restore)

```bash
# Validates H→D restore works (restarts server)
MODEL_PATH=/path/to/model bash test_l2_loadback.sh
```

Expected output:
```
load_back_tokens BEFORE: 0
load_back_tokens AFTER: 128
PASS: L2 load_back worked! (+128 tokens restored H→D)
```

### Test L3 (Disk)

```bash
# Validates L3 disk storage works (restarts server)
MODEL_PATH=/path/to/model bash test_l3_storage.sh
```

## Troubleshooting

### No eviction happening
- Check `max_total_tokens` is small enough to create pressure
- Verify `hicache_host_used_tokens` increases under load

### No load_back happening
- Ensure you're re-sending requests with **identical prefixes**
- Check `load_back_tokens_total` metric
- Minimum threshold: 10 tokens

### Empty outputs with Qwen3 thinking models
- Qwen3 uses `<think>` blocks that consume tokens
- Ensure `max_total_tokens` is large enough for thinking + output
- For testing, use `/v1/completions` endpoint instead of `/v1/chat/completions`

### Debug logging
```bash
SGLANG_HICACHE_DEBUG=1 python -m sglang.launch_server ...
```

## Files in this Directory

| File | Purpose |
|------|---------|
| `start_qwen3_6_service_hicache.sh` | Launch script for Qwen3.6-35B with HiCache |
| `quick_l2_validate.sh` | Quick L2 backup test (uses running server) |
| `test_l2_loadback.sh` | Full L2 load_back test (restarts server) |
| `test_l3_storage.sh` | L3 disk storage test (restarts server) |
| `validate_hicache_hits.sh` | Comprehensive L2/L3 validation |

## Known Limitations

1. **L3 has no size limit** - use tmpfs to bound disk usage
2. **~15% overhead** when HiCache enabled (per-step sync cost)
3. **No L2 read metric exposed** by default in some code paths
4. **Qwen3 thinking mode** can produce empty visible content if thinking consumes all tokens

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_HICACHE_DEBUG` | `0` | Enable debug logging |
| `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` | `/tmp/hicache` | L3 storage directory |
| `SGLANG_MAMBA_CONV_DTYPE` | - | Override Mamba conv dtype (set to `float16` for XPU) |
| `SGLANG_MAMBA_SSM_DTYPE` | - | Override Mamba SSM dtype (set to `float16` for XPU) |
