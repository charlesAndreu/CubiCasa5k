# CubiCasa5k `train_full` — recent performance and memory work

## DataLoader memory (`num_workers`, `prefetch_factor`)

**What changed:** Defaults and recommended YAML values for `train_full` were reduced (for example `num_workers: 8`, `prefetch_factor: 2` instead of much higher worker counts).

**Why it matters:**

- **`train_full`** loads **wider tensors** than `train_simple` (many heatmap channels plus a stacked label tensor), so **each worker process uses more RAM** for batches and buffers.
- PyTorch can use **`persistent_workers=True`**, which keeps worker processes alive across epochs. You then have **two pools** (training and validation), so total worker-related memory is roughly **scaled by `(num_workers × 2)`** (plus prefetch queues).
- A **large `prefetch_factor`** multiplies how many batches each worker holds **ahead of time**, which increases peak memory with little benefit once the GPU is the bottleneck.

**Practical takeaway:** If the machine **swaps** or training becomes **I/O-bound and sluggish**, lowering **`num_workers`** and **`prefetch_factor`** is usually the first lever—before shrinking **`batch_size`**. Setting **`debug: true`** forces **`num_workers: 0`** for quick, low-RAM debugging.

---

## GPU optimization (mixed precision / AMP)

**What changed:** **`train_full.py`** uses **automatic mixed precision (AMP)** whenever **`torch.cuda.is_available()`** — there is **no `amp` config flag**; CPU runs stay **FP32**.

**How it works:**

- On **CUDA**, train and val forward passes (and loss inside those regions) run under **`torch.amp.autocast`** so many ops use **lower precision** where safe.
- If the GPU supports it, **`torch.bfloat16`** is used (**no `GradScaler`**). Otherwise **`float16`** is used with **`GradScaler`** for stable gradients.
- **Validation** uses the same autocast policy as training so **val loss stays comparable** to the training objective.

**Practical takeaway:** AMP often **speeds up** steps and can **lower VRAM** on GPU-bound workloads. To force **FP32** on GPU you would need a small code change (bypass autocast), not YAML.
