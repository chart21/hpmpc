# Step-by-Step Setup Instructions

This guide covers the full setup from launching the Docker container to running HPMPC executions on an H100 or A100 machine.

| GPU | Architecture | `arch` flag | `GPU_ARCHITECTURE` |
|-----|-------------|-------------|-------------------|
| H100 | sm_90 | `arch=sm_90` | `90` |
| A100 | sm_80 | `arch=sm_80` | `80` |

---

## 1. Launch the Docker Container

From the repo root on the host machine:

```bash
# First time: build the image and launch with all GPUs
./docker-run.sh --build --gpus all

# Subsequent runs (image already built)
./docker-run.sh --gpus all
```

This mounts the current directory (the repo) into `/hpmpc` inside the container and drops you into a bash shell.

> All subsequent steps are run **inside the container** at `/hpmpc`.

---

## 2. Initialize Submodules

If not already done on the host, initialize the submodules (Pygeon, PIGEON/FlexNN, ConvTriple):

```bash
git submodule update --init --recursive
```

---

## 3. Set Up ConvTriple (HE-based Triple Generation)

ConvTriple has its own dependencies (emp-tool, emp-ot, Microsoft SEAL 4.1, and troy-nova for GPU) that must be built before compiling hpmpc with `CONV_TRIPLES=1` or `CHEETAH_GPU=1`.

The GPU architecture is read from the `GPU_ARCHITECTURE` environment variable, which is preset in the Docker image. Override it if needed before running `deps.sh`:

| GPU | Command |
|-----|---------|
| H100 | `GPU_ARCHITECTURE=90 ./deps.sh -gpu` (or just `./deps.sh -gpu` — 90 is the default in the image) |
| A100 | `GPU_ARCHITECTURE=80 ./deps.sh -gpu` |

```bash
cd nn/ConvTriple
rm -rf deps build

# H100
./deps.sh -gpu

# A100
GPU_ARCHITECTURE=80 ./deps.sh -gpu

./build.sh
cd ../..
```

> `-gpu` builds emp-tool, emp-ot, SEAL 4.1, and troy-nova (GPU HE). This is a superset of the CPU-only build, so a single `./deps.sh -gpu` covers all four execution variants below.
> This step only needs to be done once. If `nn/ConvTriple/deps/` already exists, `deps.sh` is skipped automatically.

---

## 4. Set Up CUDA GEMM (GPU Online Phase)

Required for `USE_CUDA_GEMM=2` in the `make` command. Use the correct `arch` for your GPU:

```bash
cd core/cuda

# H100
make -j arch=sm_90 CUDA_PATH=/usr/local/cuda CUTLASS_PATH=/cutlass

# A100
make -j arch=sm_80 CUDA_PATH=/usr/local/cuda CUTLASS_PATH=/cutlass

cd ../..
```

> The compiled `.o` files land in `core/cuda/bin/` and are linked automatically when `USE_CUDA_GEMM` is set.

---

## 5. Download Pretrained Models and Datasets (for PIGEON inference)

Required when `FUNCTION_IDENTIFIER` targets a neural network (e.g. 287).

```bash
cd nn/Pygeon
python download_pretrained.py single_model datasets
cd ../..
```

Then export the environment variables so the runtime can find the files:

```bash
export MODEL_DIR=nn/Pygeon/models/pretrained
export DATA_DIR=nn/Pygeon/data/datasets
export MODEL_FILE=vgg16_cifar_standard.bin
export SAMPLES_FILE=CIFAR-10_standard_test_images.bin
export LABELS_FILE=CIFAR-10_standard_test_labels.bin
```

---

## 6. Run Executions

Use `run_cases.sh` to build and run any or all of the four cases. Logs are saved to `logs/case<N>_make.log` and `logs/case<N>_run.log`.

```
Usage: ./run_cases.sh [OPTIONS]

  -c, --case <N>         Run only case N (1–4). Default: run all cases.
  -m, --make-only        Only run make, skip scripts/run.sh.
  -o, --override <FLAGS> Extra make flags appended to the case (requires -c).
  -h, --help             Show help.
```

### Quick reference

```bash
./run_cases.sh              # run all 4 cases
./run_cases.sh -c 1         # run only case 1
./run_cases.sh -c 2 -m      # make only for case 2, skip run
./run_cases.sh -c 2 -o "NUM_INPUTS=100 DATTYPE=256"  # case 2 with extra flags
```

### Cases

| Case | Description | Key flags |
|------|-------------|-----------|
| 1 | CPU only, skip preprocessing | `TRIPLE_PRECOMPUTE_OPT=1 SKIP_PRE=1` |
| 2 | GPU online phase, skip preprocessing | `SKIP_PRE=1 USE_CUDA_GEMM=2` |
| 3 | GPU preprocessing, CPU online phase | `SKIP_PRE=0 CHEETAH_GPU=1` |
| 4 | GPU preprocessing + GPU online phase | `SKIP_PRE=0 CHEETAH_GPU=1 USE_CUDA_GEMM=2` |

**Requires per case:**
- Case 1 — Steps 2 and 3
- Case 2 — Steps 2, 3, and 4
- Case 3 — Steps 2, 3 (`./deps.sh -gpu`), and 4
- Case 4 — Steps 2, 3 (`./deps.sh -gpu`), and 4

---

## Summary Checklist

| Step | H100 | A100 | Required for |
|------|------|------|-------------|
| `git submodule update --init --recursive` | ✓ | ✓ | All executions |
| `./deps.sh -gpu` + `build.sh` | `GPU_ARCHITECTURE=90` (default) | `GPU_ARCHITECTURE=80` | All preprocessing variants |
| `core/cuda make arch=...` | `arch=sm_90` | `arch=sm_80` | `USE_CUDA_GEMM=2` runs |
| Download pretrained models/datasets | ✓ | ✓ | Runs with real data (`FUNCTION_IDENTIFIER=287`) |
