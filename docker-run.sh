#!/bin/bash

# ─────────────────────────────────────────
# Configuration — edit these
IMAGE_NAME="hpmpc-gpu"
DOCKERFILE="docker/Dockerfile"
MOUNT_TARGET="/hpmpc"   # path inside the container where the repo is mounted
# ─────────────────────────────────────────
#
# Usage:
#   ./docker-run.sh                               → run the container (with mount)
#   ./docker-run.sh --build                       → build then run
#   ./docker-run.sh --image myimage               → run with a custom image name
#   ./docker-run.sh --build --image myimage       → build and run with a custom image name
#   ./docker-run.sh --no-mount                    → run without mounting local directory
#   ./docker-run.sh --gpus                        → run with all GPUs
#   ./docker-run.sh --gpus all                    → run with all GPUs (explicit)
#   ./docker-run.sh --gpus 0                      → run with GPU device 0
#   ./docker-run.sh --gpus 0,1                    → run with GPU devices 0 and 1
#
# ─────────────────────────────────────────

set -e

# Must be run from project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse flags
BUILD=false
NO_MOUNT=false
CUSTOM_IMAGE=""
GPU_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --build) BUILD=true; shift ;;
    --no-mount) NO_MOUNT=true; shift ;;
    --image)
      if [ -z "$2" ]; then
        echo "❌ --image requires a name"; exit 1
      fi
      CUSTOM_IMAGE="$2"; shift 2 ;;
    --gpus)
      if [[ -n "$2" && "$2" != --* ]]; then
        VAL="$2"; shift 2
      else
        VAL="all"; shift
      fi
      if [[ "$VAL" == "all" ]]; then
        GPU_ARGS=(--gpus all)
      else
        GPU_ARGS=(--gpus "\"device=$VAL\"")
      fi
      ;;
    *) echo "❌ Unknown option: $1"; exit 1 ;;
  esac
done

# Resolve image name
RESOLVED_IMAGE="${CUSTOM_IMAGE:-$IMAGE_NAME}"

# Build if requested
if [ "$BUILD" = true ]; then
  echo "🔨 Building Docker image '$RESOLVED_IMAGE' from $DOCKERFILE..."
  docker build -t "$RESOLVED_IMAGE" -f "$DOCKERFILE" .
  echo "✅ Build complete."
fi

# Check the image exists before trying to run
if ! docker image inspect "$RESOLVED_IMAGE" &>/dev/null; then
  echo "❌ Image '$RESOLVED_IMAGE' not found. Run with --build first:"
  echo "   ./docker-run.sh --build"
  exit 1
fi

# Build mount flag
if [ "$NO_MOUNT" = false ]; then
  MOUNT_FLAG="-v ${SCRIPT_DIR}:${MOUNT_TARGET}"
  echo "📂 Mounting local directory: $SCRIPT_DIR → $MOUNT_TARGET"
else
  MOUNT_FLAG=""
  echo "📂 Running without mount."
fi

# Run the container
if [ ${#GPU_ARGS[@]} -gt 0 ]; then
  echo "🚀 Starting container '$RESOLVED_IMAGE' with GPUs: ${GPU_ARGS[*]}..."
else
  echo "🚀 Starting container '$RESOLVED_IMAGE' (no GPUs)..."
fi
CONTAINER_NAME="${RESOLVED_IMAGE}-container"
echo "🐳 Container name: $CONTAINER_NAME"

if docker container inspect "$CONTAINER_NAME" &>/dev/null; then
  read -r -p "⚠️  Container '$CONTAINER_NAME' already exists. Remove it and start fresh? [y/N] " confirm
  if [[ "$confirm" =~ ^[Yy]$ ]]; then
    docker rm -f "$CONTAINER_NAME"
    echo "🗑️  Old container removed."
  else
    echo "❌ Aborted. Remove it manually with: docker rm -f $CONTAINER_NAME"
    exit 1
  fi
fi

docker run "${GPU_ARGS[@]}" -it --name "$CONTAINER_NAME" $MOUNT_FLAG "$RESOLVED_IMAGE"
