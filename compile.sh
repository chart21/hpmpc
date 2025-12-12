#!/bin/bash

set -e

: "${COLORTERM:=false}"

PURPLE=""
CLR=""

if [[ $COLORTERM = "truecolor" ]]; then
    PURPLE="\e[35m"
    CLR="\e[0m"
fi
log() {
    echo -e "$PURPLE$@$CLR"
}

build() {
    cd "$CONVTRIPLE_PATH"
    if [[ ! -d "deps" ]]; then
        if [[ $GPU == 1 ]]; then
            ./deps.sh -gpu
        else
            ./deps.sh
        fi
    fi

    ./build.sh
    cd -

    string=$(echo "$@" | tr -s ' ')
    log "$string"

    if [[ $BUILD = "1" ]]; then
        make clean
        eval $string
    fi
}

run() {
    if [[ $RUN = "1" ]]; then
        ./scripts/run.sh -a "${IP_HOST}" -b "${IP_HOST}" -p ${PARTY} -n 2
    fi
}

CONVTRIPLE_PATH="nn/ConvTriple"
THREADS=16

OPTIMIZED_BIT_INJECTION_RELU=1
BIT_INJECTION_PREPROCESSING_OPT=0 # COT + multiplex
ONLINE_OPT=0
FUSE=0

FUNC=182 # 182 or 170 or (176 for bench)
DATTYPE=32
NUM_INPUTS=1
PROCESS_NUM=1

FC_TRIPLES=1
CONV_TRIPLES=1
BN_TRIPLES=1
FAKE=0
A_KNOWN=1

GPU=0

PARTY=all
IP_HOST="127.0.0.1"

RUN=1
BUILD=1

STD="standard" # standard or custom

get_var() {
    STATIC_VARIABLES="\
    PARTY=${PARTY} \
    PROTOCOL=4 \
    FUNCTION_IDENTIFIER=${FUNC} \
    NUM_INPUTS=${NUM_INPUTS} \
    BITLENGTH=32 \
    DATTYPE=${DATTYPE} \
    PRE=1 \
    SKIP_PRE=0 \
    PROCESS_NUM=${PROCESS_NUM} \
    INTERLEAVE_COMM=1 \
    \
    CHEETAH_THREADS=${THREADS} \
    CHEETAH_GPU=${GPU} \
    \
    A_KNOWN=${A_KNOWN} \
    FAKE_TRIPLES=${FAKE} \
    BN2D_TRIPLES=${BN_TRIPLES} \
    FC_TRIPLES=${FC_TRIPLES} \
    CONV_TRIPLES=${CONV_TRIPLES} \
    \
    OPTIMIZED_BIT_INJECTION_RELU=${OPTIMIZED_BIT_INJECTION_RELU} \
    BIT_INJECTION_PREPROCESSING_OPT=${BIT_INJECTION_PREPROCESSING_OPT} \
    A2B_ONLINE_OPT=${ONLINE_OPT} \
    A_KNOWN_TO_EVALUATORS_OPT=${ONLINE_OPT} \
    \
    FUSE_CONV_BN=${FUSE} \
    FUSE_RELU_AVG=${FUSE} \
    "
    echo $STATIC_VARIABLES
}

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

print_help() {
    echo "OPTIONS"
    TAB="  "
    SPACING="\t\t"
    echo -e "$TAB--fuse:\tenables 'FUSE_CONV_BN' and 'FUSE_RELU_AVG'"
    echo -e "$TAB--injection\tenables 'BIT_INJECTION_PREPROCESSING_OPT'"
    echo -e "$TAB--online\tenables 'A2B_ONLINE_OPT' and 'A_KNOWN_TO_EVALUATORS_OPT'"
    echo -e "$TAB-a, --host <IP>"
    echo -e "$TAB-b, --bench <FUNC>"
    echo -e "$TAB-c, --compile"
    echo -e "$TAB-ct, --cheetah-threads"
    echo -e "$TAB-nc, --no-compile"
    echo -e "$TAB-d, --dattype <DAT>"
    echo -e "$TAB-k, --kill"
    echo -e "$TAB-n, --num-process <PROCESS_NUM>"
    echo -e "$TAB-p, --party <ID>"
    echo -e "$TAB-r, --run"
    echo -e "$TAB-t, --test <FUNC>"
    echo -e "$TAB-h, --help"
}

error() {
    echo -e "\033[31m$@\033[0m"
    exit -1
}


trap terminate SIGINT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --online)
            ONLINE_OPT=1
            ;;
        --fuse)
            FUSE=1
            ;;
        --injection)
            BIT_INJECTION_PREPROCESSING_OPT=1
            ;;
        -g|--gpu)
            GPU=1
            ;;
        -c|--compile)
            RUN=0
            ;;
        -ct|--cheetah-threads)
            THREADS=$2
            shift
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
            FUNC=$2

            build "make -j $(get_var) MODELOWNER=-1 DATAOWNER=-1"

            run

            shift
            ;;
        -r|--run)
            log "Input: $(( $NUM_INPUTS * DATTYPE / 32 ))"
            log "Process_num: ${PROCESS_NUM}"

            build "make -j $(get_var) MODELOWNER=P_0 DATAOWNER=P_1"

            run
            ;;
        -t|--test)
            if (( "$2" < 54 || "$2" > 59 )); then
                error "Unknown test '$2'"
            fi
            FUNC="$2"

            shift

            log "RUNNING TEST $FUNC..."

            build "make -j $(get_var) SPLITROLES=0 USE_CUDA_GEMM=0"

            run
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
