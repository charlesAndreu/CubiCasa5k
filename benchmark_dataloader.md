# DataLoader benchmark (`benchmark_dataloader.py`)

Measures **RoomLoader / IconLoader** throughput with the same **DataLoader** settings as `train_simple` (LMDB, augmentations, **`--image-size`**, `num_workers`, `prefetch_factor`, `pin_memory`).

Each run prints **`image_size`**, **`batch tensor shape (N,C,H,W)`** (should match `--image-size` for H and W), and **samples/s**.

Run from the **repository root** with your conda env (e.g. `charles-cubicasa`):

```bash
cd /path/to/CubiCasa5k
conda activate charles-cubicasa
```

---

## Quick smoke test (few batches)

```bash
python benchmark_dataloader.py \
  --segmentation-map room \
  --data-path data/cubicasa5k/ \
  --batches 15 \
  --warmup-batches 3 \
  --batch-size 4 \
  --num-workers 0
```

---

## Image size (single run)

Same as `train_simple --image-size` — controls crop/resize target before batching:

```bash
python benchmark_dataloader.py \
  --segmentation-map room \
  --data-path data/cubicasa5k/ \
  --batches 50 \
  --batch-size 26 \
  --num-workers 2 \
  --image-size 256
```

Faster augments / smaller tensors (compare quality in training separately):

```bash
python benchmark_dataloader.py \
  --segmentation-map room \
  --data-path data/cubicasa5k/ \
  --batches 50 \
  --batch-size 26 \
  --num-workers 2 \
  --image-size 128
```

Heavier (more pixels per batch):

```bash
python benchmark_dataloader.py \
  --segmentation-map room \
  --data-path data/cubicasa5k/ \
  --batches 50 \
  --batch-size 26 \
  --num-workers 2 \
  --image-size 384
```

---

## Image size sweep (one command, several sizes)

Runs a separate timed benchmark for each size (good for a quick table):

```bash
python benchmark_dataloader.py \
  --segmentation-map room \
  --data-path data/cubicasa5k/ \
  --batches 50 \
  --batch-size 26 \
  --num-workers 2 \
  --sweep-image-sizes 128 256 384
```

You can combine with `--to-gpu` or other flags as usual.

---

## Baseline close to `train_simple` defaults

```bash
python benchmark_dataloader.py \
  --segmentation-map room \
  --data-path data/cubicasa5k/ \
  --batches 50 \
  --warmup-batches 5 \
  --batch-size 26 \
  --num-workers 2
```

---

## More workers (check if throughput improves)

```bash
python benchmark_dataloader.py \
  --segmentation-map room \
  --data-path data/cubicasa5k/ \
  --batches 50 \
  --batch-size 26 \
  --num-workers 8 \
  --prefetch-factor 4
```

---

## Single-process loader (same idea as `train_simple --debug`)

```bash
python benchmark_dataloader.py \
  --segmentation-map room \
  --data-path data/cubicasa5k/ \
  --batches 50 \
  --batch-size 26 \
  --debug
```

---

## Include host → GPU copy + CUDA sync

Closer to “decode/augment + feed GPU” per batch:

```bash
python benchmark_dataloader.py \
  --segmentation-map room \
  --data-path data/cubicasa5k/ \
  --batches 50 \
  --batch-size 26 \
  --num-workers 4 \
  --to-gpu
```

---

## Same augmentation as `train_simple --scale`

```bash
python benchmark_dataloader.py \
  --segmentation-map room \
  --data-path data/cubicasa5k/ \
  --batches 50 \
  --scale
```

---

## Reading the output

| Field | Meaning |
|--------|--------|
| **config: image_size=…** | Same meaning as `train_simple --image-size` (output H×W after augmentations). |
| **batch tensor shape (N,C,H,W)** | Confirms H=W=`image_size` for typical runs. |
| **samples/s** | Higher is better: more training images per second through decode + augment + batch. |
| **batches/s** | Batches per second (depends on `batch-size`). |
| **ms/batch** | Average time per batch. |

**Comparing `image_size`:** Larger sizes usually mean **lower samples/s** (more CPU in augments and larger batches to collate). If changing **`--image-size`** barely moves **samples/s**, the bottleneck is likely **LMDB read / unpickle of full-resolution** data before crops, not the final square size alone.

**Comparing runs:** If raising `num_workers` barely changes **samples/s**, you may be limited by **disk I/O**, **CPU in augmentations**, or something else—not worker count.

**`--to-gpu`:** Adds H2D transfer and `cuda.synchronize()` to the timed loop, so times reflect “loader + copy to GPU,” not loader alone.

---

## Data path

`--data-path` must point to a directory that contains **`cubi_lmdb/`** and **`train.txt`** (same layout as `train_simple`). Change `--data-path` if your dataset lives elsewhere.

Use **`--segmentation-map icon`** for the icon loader instead of `room`.
