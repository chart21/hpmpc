#!/bin/bash

set -e

FUNC=182
DATTYPE=32
NUM_INPUTS=1
PROCESS_NUM=1

STD="standard"

export MODEL_DIR=nn/Pygeon/models/pretrained
export DATA_DIR=nn/Pygeon/data/datasets

if (( $FUNC == 182 )); then
    export MODEL_FILE=MNIST_LeNet5/LeNet5_MNIST_${STD}_best.bin
    export SAMPLES_FILE=MNIST_${STD}_test_images.bin
    export LABELS_FILE=MNIST_${STD}_test_labels.bin
else
    export MODEL_FILE=Cifar_adam_001/ResNet18_avg_CIFAR-10_standard_best.bin
    export SAMPLES_FILE=CIFAR-10_standard_test_images.bin
    export LABELS_FILE=CIFAR-10_standard_test_labels.bin
fi

terminate() {
    echo "TERMINATING..."
    pkill -9 -f run-P
}

log() {
    echo -e "\033[35m$@\033[0m"
}

error() {
    echo -e "\033[31m$@\033[0m"
    exit -1
}

trap terminate SIGINT

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dattype)
            if (( "$2" == 32  || "$2" == 128 || "$2" == 256 )); then
                DATTYPE=$2
            fi

            log "Set DATTYPE to '$DATTYPE'"
            shift
            ;;
        -p|--process)
            PROCESS_NUM=$2
            log "Set PROCESS_NUM to '$PROCESS_NUM'"
            shift
            ;;
        -k|--kill)
            terminate
            ;;
        -r|--run|"")
            log "Input: $(( $NUM_INPUTS * DATTYPE / 32 ))"
            log "Process_num: ${PROCESS_NUM}"

            make -j PARTY=all FUNCTION_IDENTIFIER=${FUNC} MODELOWNER=P_0 \
                DATAOWNER=P_1 DATTYPE=${DATTYPE} PROTOCOL=4 \
                NUM_INPUTS=${NUM_INPUTS} BITLENGTH=32 \
                PROCESS_NUM=1 PRE=1 FAKE_TRIPLES=1 AB2_TRIPLES=1 BN2D_TRIPLES=0 \
                FC_TRIPLES=0 CONV_TRIPLES=0

            ./scripts/run.sh -p all -n 2
            ;;
        -t|--test)
            if (( "$2" < 54 || "$2" > 59 )); then
                error "Unknown test '$2'"
            fi
            FUNC="$2"

            shift

            log "RUNNING TEST $FUNC..."

            make -j PARTY=all PROTOCOL=4 FUNCTION_IDENTIFIER=${FUNC} \
                BITLENGTH=32 PRE=1 DATTYPE=${DATTYPE} NUM_INPUTS=1 \
                SPLITROLES=0 PROCESS_NUM=${PROCESS_NUM} USE_CUDA_GEMM=0 \
                SKIP_PRE=0 FAKE_TRIPLES=0 AB2_TRIPLES=1 BN2D_TRIPLES=0 FC_TRIPLES=0 CONV_TRIPLES=0

            ./scripts/run.sh -p all -n 2
            ;;
        *)
            error "Unknown option '$1'"
    esac
    shift
done
