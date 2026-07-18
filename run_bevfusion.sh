#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="bevfusion:latest"
DOCKERFILE_PATH="$ROOT_DIR/docker/Dockerfile"
BUILD_LOG="$ROOT_DIR/docker_build.log"
DEFAULT_DATASET_ROOT="/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/"
DATASET_ROOT="${BEVLOG_DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"
RESULTS_DIR="${ROOT_DIR}/results"
CODE_DIR="${ROOT_DIR}/official_mmdet3d"

# Cria as pastas necessárias
mkdir -p "$RESULTS_DIR"
mkdir -p "$CODE_DIR"

# ----------------------------------------------------------------------
# Garante que o diretório do código esteja populado (extrai da imagem se necessário)
# ----------------------------------------------------------------------
if [ ! "$(ls -A "$CODE_DIR" 2>/dev/null)" ]; then
    echo "[bevfusion] CODE_DIR está vazio. Extraindo código da imagem..."
    docker run --rm --entrypoint tar "$IMAGE_NAME" \
        -c -C /workspace/official_mmdet3d . | tar -x -C "$CODE_DIR"
    echo "[bevfusion] Código extraído para $CODE_DIR"
fi

# ----------------------------------------------------------------------
# Constrói a imagem se necessário
# ----------------------------------------------------------------------
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "[bevfusion] Docker image '$IMAGE_NAME' not found. Building..."
    docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_NAME" . | tee "$BUILD_LOG"
else
    echo "[bevfusion] Docker image '$IMAGE_NAME' already exists."
fi

# ----------------------------------------------------------------------
# Parâmetros do container
# ----------------------------------------------------------------------
DOCKER_RUN_ARGS=(--gpus all -it --rm --shm-size 8g)

# Monta o código editável do host (agora populado)
echo "[bevfusion] Mounting code from: $CODE_DIR"
DOCKER_RUN_ARGS+=(-v "$CODE_DIR:/workspace/official_mmdet3d")

# Monta o dataset (se existir)
if [[ -d "$DATASET_ROOT" ]]; then
    echo "[bevfusion] Mounting dataset from: $DATASET_ROOT"
    DOCKER_RUN_ARGS+=(-v "$DATASET_ROOT:/workspace/official_mmdet3d/data:ro")
else
    echo "[bevfusion] Dataset path not found: $DATASET_ROOT"
fi

# Monta a pasta de resultados
echo "[bevfusion] Results will be saved to: $RESULTS_DIR"
DOCKER_RUN_ARGS+=(-v "$RESULTS_DIR:/workspace/results")

# ----------------------------------------------------------------------
# Executa o container
# ----------------------------------------------------------------------
echo "[bevfusion] Starting container..."
docker run "${DOCKER_RUN_ARGS[@]}" "$IMAGE_NAME"