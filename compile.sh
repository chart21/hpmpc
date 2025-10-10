#!/bin/bash

set -e

THREADS=32

FUNC=182 # 182 or 170 or (176 for bench)
DATTYPE=32
NUM_INPUTS=1
PROCESS_NUM=1

FC_TRIPLES=1
CONV_TRIPLES=1
BN_TRIPLES=1
FAKE=0

PARTY=all
IP_HOST="127.0.0.1"

RUN=1
BUILD=1

STD="standard" # standard or custom

export MODEL_DIR=nn/Pygeon/models/pretrained
export DATA_DIR=nn/Pygeon/data/datasets

if (( $FUNC == 182 )); then
    export MODEL_FILE=MNIST_LeNet5/LeNet5_MNIST_${STD}_best.bin
    export SAMPLES_FILE=MNIST_${STD}_test_images.bin
    export LABELS_FILE=MNIST_${STD}_test_labels.bin
elif (( $FUNC == 170 )); then
    export MODEL_FILE=Cifar_adam_001/ResNet18_avg_CIFAR-10_${STD}_best.bin
    export SAMPLES_FILE=CIFAR-10_${STD}_test_images.bin
    export LABELS_FILE=CIFAR-10_${STD}_test_labels.bin
fi

terminate() {
    echo "TERMINATING..."
    pkill -9 -f run-P
}

log() {
    echo -e "\033[35m$@\033[0m"
}

print_help() {
    echo "OPTIONS"
    echo -e "\t-a, --host <IP>"
    echo -e "\t-b, --bench <FUNC>"
    echo -e "\t-c, --compile"
    echo -e "\t-nc, --no-compile"
    echo -e "\t-d, --dattype <DAT>"
    echo -e "\t-k, --kill"
    echo -e "\t-n, --num-process <PROCESS_NUM>"
    echo -e "\t-p, --party <ID>"
    echo -e "\t-r, --run"
    echo -e "\t-t, --test <FUNC>"
    echo -e "\t-h, --help"
}

error() {
    echo -e "\033[31m$@\033[0m"
    exit -1
}


trap terminate SIGINT

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--compile)
            RUN=0
            ;;
        -nc|--no-compile)
            BUILD=0
            ;;
        -a|--host)
            IP_HOST="$2"
            log "IP set to '${IP_HOST}'"
            shift
            ;;
        -d|--dattype)
            if (( "$2" == 32  || "$2" == 128 || "$2" == 256 )); then
                DATTYPE=$2
            fi

            log "Set DATTYPE to '$DATTYPE'"
            shift
            ;;
        -p|--party)
            PARTY=$2
            log "Set PARTY_ID to '$PARTY'"
            shift
            ;;
        -n|--num-process)
            PROCESS_NUM=$2
            log "Set PROCESS_NUM to '$PROCESS_NUM'"
            shift
            ;;
        -k|--kill)
            terminate
            ;;
        -b|--bench)
            if [[ $BUILD = "1" ]]; then
                make -j PARTY=${PARTY} FUNCTION_IDENTIFIER=$2 MODELOWNER=-1 \
                    DATAOWNER=-1 DATTYPE=${DATTYPE} PROTOCOL=4 \
                    NUM_INPUTS=${NUM_INPUTS} BITLENGTH=32 \
                    PROCESS_NUM=${PROCESS_NUM} PRE=1 FAKE_TRIPLES=${FAKE} \
                    AB2_TRIPLES=1 BN2D_TRIPLES=${BN_TRIPLES} \
                    FC_TRIPLES=${FC_TRIPLES} CONV_TRIPLES=${CONV_TRIPLES} INTERLEAVE_COMM=1 \
                    CHEETAH_THREADS=${THREADS}
            fi

            if [[ $RUN = "1" ]]; then
                ./scripts/run.sh -a ${IP_HOST} -p ${PARTY} -n 2
            fi
            shift
            ;;
        -r|--run)
            log "Input: $(( $NUM_INPUTS * DATTYPE / 32 ))"
            log "Process_num: ${PROCESS_NUM}"

            make -j PARTY=${PARTY} FUNCTION_IDENTIFIER=${FUNC} MODELOWNER=P_0 \
                DATAOWNER=P_1 DATTYPE=${DATTYPE} PROTOCOL=4 \
                NUM_INPUTS=${NUM_INPUTS} BITLENGTH=32 \
                PROCESS_NUM=${PROCESS_NUM} PRE=1 FAKE_TRIPLES=${FAKE} AB2_TRIPLES=1 BN2D_TRIPLES=${BN_TRIPLES} \
                FC_TRIPLES=${FC_TRIPLES} CONV_TRIPLES=${CONV_TRIPLES} INTERLEAVE_COMM=1

            ./scripts/run.sh -a ${IP_HOST} -p ${PARTY} -n 2
            ;;
        -t|--test)
            if (( "$2" < 54 || "$2" > 59 )); then
                error "Unknown test '$2'"
            fi
            FUNC="$2"

            shift

            log "RUNNING TEST $FUNC..."

            make -j PARTY=${PARTY} PROTOCOL=4 FUNCTION_IDENTIFIER=${FUNC} \
                BITLENGTH=32 PRE=1 DATTYPE=${DATTYPE} NUM_INPUTS=1 \
                SPLITROLES=0 PROCESS_NUM=${PROCESS_NUM} USE_CUDA_GEMM=0 \
                SKIP_PRE=0 FAKE_TRIPLES=0 AB2_TRIPLES=1 \
                BN2D_TRIPLES=${BN_TRIPLES} FC_TRIPLES=${FC_TRIPLES} CONV_TRIPLES=${CONV_TRIPLES}

            ./scripts/run.sh -a ${IP_HOST} -p ${PARTY} -n 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            error "Unknown option '$1'"
    esac
    shift
done
