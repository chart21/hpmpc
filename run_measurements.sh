#!/bin/bash

# ─────────────────────────────────────────────────────────────────────────────
# run_measurements.sh — Run HPMPC measurement configs across GPU/CPU variants
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ─── Test definitions ─────────────────────────────────────────────────────────
declare -A TEST_DESC
declare -A TEST_CMD_EXTRA

TEST_DESC[1]="GPU (default config)"
TEST_CMD_EXTRA[1]=""

TEST_DESC[2]="CPU (no GPU, 24 GEMM threads)"
TEST_CMD_EXTRA[2]="--override USE_CUDA_GEMM=0 CHEETAH_GPU=0 ADDITIONAL_GEMM_THREADS=24"

VALID_TESTS="1 2"
BASE_CONFIG="measurements/configs/artifacts/triad/2pc/GPU"

# ─────────────────────────────────────────────────────────────────────────────
print_help() {
    cat <<EOF
Usage: $0 -p <PID> -a <IPA> -b <IPB> [OPTIONS]

Run HPMPC 2PC measurement configs, saving output logs per test.

REQUIRED:
  -p <PID>          Party ID (0 or 1)
  -a <IPA>          IP address of party 0
  -b <IPB>          IP address of party 1

OPTIONS:
  -t, --test <N>    Run only test N (1-2). Default: run all tests.
  -n, --num <N>     Number of iterations per run (passed as -i <N>).
  -h, --help        Show this help message.

TESTS:
  1  GPU — measurements/configs/artifacts/triad/2pc/GPU
  2  CPU — same config with USE_CUDA_GEMM=0 CHEETAH_GPU=0 ADDITIONAL_GEMM_THREADS=24

LOGS:
  Saved to logs/measurement_test<N>_<timestamp>.log

EXAMPLES:
  $0 -p 0 -a 192.168.1.1 -b 192.168.1.2
  $0 -p 1 -a 192.168.1.1 -b 192.168.1.2 -t 1
  $0 -p 0 -a 192.168.1.1 -b 192.168.1.2 -n 5
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
run_test() {
    local n="$1"
    local desc="${TEST_DESC[$n]}"
    local extra="${TEST_CMD_EXTRA[$n]}"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="$LOG_DIR/measurement_test${n}_${timestamp}.log"

    echo ""
    echo "══════════════════════════════════════════"
    echo "  Test $n: $desc"
    echo "══════════════════════════════════════════"

    local cmd="python3 measurements/run_config.py $BASE_CONFIG -p $PID -a $IPA -b $IPB"
    [ -n "$NUM_ITER" ] && cmd="$cmd -i $NUM_ITER"
    [ -n "$extra"    ] && cmd="$cmd $extra"

    echo "  → cmd: $cmd"
    echo "  → log: $log_file"

    eval "$cmd" > "$log_file" 2>&1 && \
        echo "  ✓ test succeeded" || \
        echo "  ✗ test FAILED — see $log_file"
}

# ─────────────────────────────────────────────────────────────────────────────
# Parse arguments
PID=""
IPA=""
IPB=""
TEST=""
NUM_ITER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p) PID="$2";      shift 2 ;;
        -a) IPA="$2";      shift 2 ;;
        -b) IPB="$2";      shift 2 ;;
        -t|--test) TEST="$2"; shift 2 ;;
        -n|--num)  NUM_ITER="$2"; shift 2 ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "❌ Unknown option: $1"; echo ""; print_help; exit 1 ;;
    esac
done

# Validate required args
missing=""
[ -z "$PID" ] && missing="$missing -p"
[ -z "$IPA" ] && missing="$missing -a"
[ -z "$IPB" ] && missing="$missing -b"
if [ -n "$missing" ]; then
    echo "❌ Missing required arguments:$missing"
    echo ""
    print_help
    exit 1
fi

# Validate test number
if [ -n "$TEST" ]; then
    if [[ ! " $VALID_TESTS " =~ " $TEST " ]]; then
        echo "❌ Invalid test '$TEST'. Must be one of: $VALID_TESTS"
        exit 1
    fi
fi

echo "Party: $PID | IPA: $IPA | IPB: $IPB${NUM_ITER:+ | Iterations: $NUM_ITER}"

# ─────────────────────────────────────────────────────────────────────────────
if [ -n "$TEST" ]; then
    run_test "$TEST"
else
    for n in 1 2; do
        run_test "$n"
    done
fi

echo ""
echo "Done. Logs saved in $LOG_DIR/"
