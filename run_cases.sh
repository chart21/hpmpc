#!/bin/bash

# ─────────────────────────────────────────
# run_cases.sh — Build and run HPMPC test cases
# ─────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ─── Base make flags shared across all cases ───
BASE_MAKE="make -j PARTY=all PROTOCOL=4 PRE=1 FUNCTION_IDENTIFIER=287 \
    ROT_PREPROCESSING=1 CHEETAH_THREADS=16 CHEETAH_BOOL_OT_TYPE=0 \
    MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=1 RESHARE_OPT=0 RESHARE_OPT_SIM=0 \
    A2B_ONLINE_OPT=1 A_KNOWN_TO_EVALUATORS_OPT=1 FUSE_RELU_AVG=1 FUSE_CONV_BN=1 \
    BITLENGTH=32 COMPRESS=0 INTERLEAVE_COMM=1 CHEETAH_DISCONNECT=1 \
    CHEETAH_WAN_OPT=0 ADDITIONAL_GEMM_THREADS=16"

# ─── Case definitions ───
declare -A CASE_DESC
declare -A CASE_MAKE_EXTRA

CASE_DESC[1]="CPU only (skip preprocessing)"
CASE_MAKE_EXTRA[1]="TRIPLE_PRECOMPUTE_OPT=1 SKIP_PRE=1"

CASE_DESC[2]="GPU online phase, skip preprocessing (USE_CUDA_GEMM=2)"
CASE_MAKE_EXTRA[2]="SKIP_PRE=1 USE_CUDA_GEMM=2"

CASE_DESC[3]="GPU preprocessing, CPU online phase (CHEETAH_GPU=1)"
CASE_MAKE_EXTRA[3]="SKIP_PRE=0 CHEETAH_GPU=1"

CASE_DESC[4]="GPU preprocessing + GPU online phase (CHEETAH_GPU=1 USE_CUDA_GEMM=2)"
CASE_MAKE_EXTRA[4]="SKIP_PRE=0 CHEETAH_GPU=1 USE_CUDA_GEMM=2"

VALID_CASES="1 2 3 4"

# ─────────────────────────────────────────
print_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run HPMPC build-and-execute test cases, saving make and run logs separately.

OPTIONS:
  -c, --case <N>        Run only case N (1-4). Default: run all cases.
  -m, --make-only       Only run make, skip scripts/run.sh.
  -o, --override <FLAGS> Additional make flags appended to the case's make command.
                        Only applies when --case is also specified.
                        Example: -o "NUM_INPUTS=100 DATTYPE=256"
  -G <player:device>    Assign a CUDA device to a player for GPU cases (3 and 4).
                        Can be repeated. Passed directly to scripts/run.sh -G.
                        Example: -G 0:0 -G 1:1
  -h, --help            Show this help message.

CASES:
  1  CPU only (skip preprocessing)
  2  GPU online phase, skip preprocessing  (USE_CUDA_GEMM=2)
  3  GPU preprocessing, CPU online phase   (CHEETAH_GPU=1)
  4  GPU preprocessing + GPU online phase  (CHEETAH_GPU=1 USE_CUDA_GEMM=2)

LOGS:
  Saved to logs/case<N>_make.log and logs/case<N>_run.log

EXAMPLES:
  $0                              # run all 4 cases
  $0 -c 2                         # run only case 2
  $0 -c 2 -m                      # make only for case 2
  $0 -c 2 -o "NUM_INPUTS=100"     # run case 2 with extra make flag
  $0 -c 3 -G 0:0 -G 1:1           # case 3 with P0→GPU0, P1→GPU1
  $0 -G 0:0 -G 1:1                # all cases, GPU cases use GPU0/GPU1
EOF
}

# ─────────────────────────────────────────
run_case() {
    local n="$1"
    local extra_override="$2"
    local make_only="$3"

    local desc="${CASE_DESC[$n]}"
    local extra="${CASE_MAKE_EXTRA[$n]} ${extra_override}"
    local make_log="$LOG_DIR/case${n}_make.log"
    local run_log="$LOG_DIR/case${n}_run.log"

    echo ""
    echo "══════════════════════════════════════════"
    echo "  Case $n: $desc"
    echo "══════════════════════════════════════════"

    echo "  → make log: $make_log"
    eval "$BASE_MAKE $extra" > "$make_log" 2>&1 && \
        echo "  ✓ make succeeded" || \
        { echo "  ✗ make FAILED — see $make_log"; return 1; }

    if [ "$make_only" = "true" ]; then
        echo "  (skipping run — --make-only set)"
        return 0
    fi

    echo "  → run log:  $run_log"
    # Pass -G flags for GPU cases (3 and 4) if provided
    RUN_GPU_ARGS=""
    if [[ "$n" == "3" || "$n" == "4" ]]; then
        RUN_GPU_ARGS="$GPU_ARGS"
    fi
    eval "scripts/run.sh -p all -n 2 $RUN_GPU_ARGS" > "$run_log" 2>&1 && \
        echo "  ✓ run succeeded" || \
        echo "  ✗ run FAILED — see $run_log"
}

# ─────────────────────────────────────────
# Parse arguments
CASE=""
MAKE_ONLY="false"
OVERRIDE=""
GPU_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--case)
            CASE="$2"; shift 2 ;;
        -m|--make-only)
            MAKE_ONLY="true"; shift ;;
        -o|--override)
            OVERRIDE="$2"; shift 2 ;;
        -G)
            GPU_ARGS="$GPU_ARGS -G $2"; shift 2 ;;
        -h|--help)
            print_help; exit 0 ;;
        *)
            echo "❌ Unknown option: $1"; echo ""; print_help; exit 1 ;;
    esac
done

# Validate case number
if [ -n "$CASE" ]; then
    if [[ ! " $VALID_CASES " =~ " $CASE " ]]; then
        echo "❌ Invalid case '$CASE'. Must be one of: $VALID_CASES"
        exit 1
    fi
fi

# Warn if --override is used without --case
if [ -n "$OVERRIDE" ] && [ -z "$CASE" ]; then
    echo "⚠️  --override only applies when --case is specified. Ignoring override for multi-case run."
    OVERRIDE=""
fi

# ─────────────────────────────────────────
# Run
if [ -n "$CASE" ]; then
    run_case "$CASE" "$OVERRIDE" "$MAKE_ONLY"
else
    for n in 1 2 3 4; do
        run_case "$n" "" "$MAKE_ONLY"
    done
fi

echo ""
echo "Done. Logs saved in $LOG_DIR/"
