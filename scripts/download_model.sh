#!/usr/bin/env bash
set -euo pipefail

# Download source: auto (try HF then ModelScope), hf, modelscope
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-auto}"
export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
export HF_ENDPOINT_PUBLIC="${HF_ENDPOINT_PUBLIC:-https://hf-mirror.com}"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRETRAIN_DIR="${PRETRAIN_DIR:-${REPO_ROOT}/pretrain_model}"
MODEL_PATHS_FILE="${PRETRAIN_DIR}/.model_paths.env"
mkdir -p "${PRETRAIN_DIR}"

# HuggingFace repo_id -> ModelScope model ids (tried in order)
modelscope_ids() {
    case "$1" in
        google/siglip-so400m-patch14-384)
            echo "AI-ModelScope/siglip-so400m-patch14-384 ai-modelscope/siglip-so400m-patch14-384"
            ;;
        google/gemma-3-270m)
            echo "google/gemma-3-270m"
            ;;
        lerobot/pi05_base)
            echo "lerobot/pi05_base AI-ModelScope/pi05_base"
            ;;
        *)
            echo ""
            ;;
    esac
}

local_model_dir() {
    case "$1" in
        google/siglip-so400m-patch14-384) echo "${PRETRAIN_DIR}/siglip-so400m-patch14-384" ;;
        google/gemma-3-270m) echo "${PRETRAIN_DIR}/gemma-3-270m" ;;
        lerobot/pi05_base) echo "${PRETRAIN_DIR}/pi05_base" ;;
        *) echo "${PRETRAIN_DIR}/$(echo "$1" | tr '/' '_')" ;;
    esac
}

is_complete_model_dir() {
    local dir="$1"
    [[ -d "${dir}" ]] || return 1
    [[ -f "${dir}/config.json" ]] || return 1
    # At least one weight file present
    compgen -G "${dir}/*.safetensors" >/dev/null \
        || compgen -G "${dir}/*.bin" >/dev/null \
        || compgen -G "${dir}/model*.safetensors" >/dev/null
}

ensure_modelscope_cli() {
    if command -v modelscope >/dev/null 2>&1; then
        return 0
    fi
    echo ">>> Installing modelscope CLI ..."
    pip install -q modelscope
}

download_via_hf() {
    local repo_id="$1"
    local gated="$2"
    local local_dir="$3"

    if ! command -v hf >/dev/null 2>&1; then
        echo "hf CLI not found, skipping HuggingFace download for ${repo_id}"
        return 1
    fi

    echo ">>> [HF] Downloading ${repo_id}"
    if [[ "${gated}" == "true" ]]; then
        if [[ -z "${HF_TOKEN:-}" ]] && ! hf auth whoami >/dev/null 2>&1; then
            echo ">>> [HF] ${repo_id} is gated and no HF token is configured"
            return 1
        fi
        if [[ -n "${local_dir}" ]]; then
            env -u HF_ENDPOINT hf download "${repo_id}" --local-dir "${local_dir}"
        else
            env -u HF_ENDPOINT hf download "${repo_id}"
        fi
    elif [[ -n "${local_dir}" ]]; then
        HF_ENDPOINT="${HF_ENDPOINT_PUBLIC}" hf download "${repo_id}" --local-dir "${local_dir}"
    else
        HF_ENDPOINT="${HF_ENDPOINT_PUBLIC}" hf download "${repo_id}"
    fi
}

download_via_modelscope() {
    local repo_id="$1"
    local local_dir="$2"
    local ms_ids
    ms_ids="$(modelscope_ids "${repo_id}")"

    if [[ -z "${ms_ids}" ]]; then
        echo ">>> [ModelScope] No mapping for ${repo_id}"
        return 1
    fi

    ensure_modelscope_cli
    mkdir -p "${local_dir}"

    local ms_id
    for ms_id in ${ms_ids}; do
        echo ">>> [ModelScope] Downloading ${ms_id} -> ${local_dir}"
        if modelscope download --model "${ms_id}" --local_dir "${local_dir}"; then
            return 0
        fi
        echo ">>> [ModelScope] Failed ${ms_id}, trying next candidate..."
        rm -rf "${local_dir}"
        mkdir -p "${local_dir}"
    done

    return 1
}

download_repo() {
    local repo_id="$1"
    local gated="${2:-false}"
    local local_dir
    local_dir="$(local_model_dir "${repo_id}")"
    local hf_ok=false
    local ms_ok=false

    if is_complete_model_dir "${local_dir}"; then
        echo ">>> Skip ${repo_id}: already present at ${local_dir}"
        return 0
    fi

    rm -rf "${local_dir}"
    mkdir -p "${local_dir}"

    if [[ "${DOWNLOAD_SOURCE}" == "hf" || "${DOWNLOAD_SOURCE}" == "auto" ]]; then
        if download_via_hf "${repo_id}" "${gated}" "${local_dir}"; then
            hf_ok=true
        fi
    fi

    if [[ "${hf_ok}" == "false" && ( "${DOWNLOAD_SOURCE}" == "modelscope" || "${DOWNLOAD_SOURCE}" == "auto" ) ]]; then
        rm -rf "${local_dir}"
        mkdir -p "${local_dir}"
        if download_via_modelscope "${repo_id}" "${local_dir}"; then
            ms_ok=true
        fi
    fi

    if [[ "${hf_ok}" == "true" || "${ms_ok}" == "true" ]]; then
        if is_complete_model_dir "${local_dir}"; then
            echo ">>> OK ${repo_id} -> ${local_dir}"
            return 0
        fi
        echo ">>> Download finished but ${local_dir} looks incomplete"
        return 1
    fi

    echo ">>> FAILED ${repo_id}"
    if [[ "${gated}" == "true" ]]; then
        echo "    HuggingFace: accept https://huggingface.co/${repo_id} and run 'hf login'"
        echo "    ModelScope:  try DOWNLOAD_SOURCE=modelscope (google/gemma-3-270m on modelscope.cn)"
    fi
    return 1
}

write_model_paths_file() {
    cat > "${MODEL_PATHS_FILE}" <<EOF
# Generated by scripts/download_model.sh
PRETRAIN_DIR=${PRETRAIN_DIR}
VALUE_VISION_REPO_ID=${PRETRAIN_DIR}/siglip-so400m-patch14-384
VALUE_LANGUAGE_REPO_ID=${PRETRAIN_DIR}/gemma-3-270m
PI05_PRETRAINED_PATH=${PRETRAIN_DIR}/pi05_base
EOF
    echo ">>> Wrote ${MODEL_PATHS_FILE}"
}

# --- RECAP value training backbones (Step 1) ---
download_repo "google/siglip-so400m-patch14-384" false
download_repo "google/gemma-3-270m" true

# --- VLA finetuning base (Step 4) ---
# download_repo "lerobot/pi05_base" false

write_model_paths_file

echo ">>> Done."
echo "    Local models: ${PRETRAIN_DIR}"
echo "    Source ${MODEL_PATHS_FILE} before training, or run run_finetune_recap.sh (auto-detects local models)."

echo ">>> Linking local models to HuggingFace repo_id"
ln -s /work/lerobot/pretrain_model/siglip-so400m-patch14-384 google/siglip-so400m-patch14-384
ln -s /work/lerobot/pretrain_model/gemma-3-270m google/gemma-3-270m