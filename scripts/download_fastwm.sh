#!/usr/bin/env bash
# Download Fast-WAM weights used by finetune_screw_sorting_fastwam.sh.
# Same layout as download_model.sh: mirror + existing proxy, files under pretrain_model/.
#
#   lerobot/fastwam_base                         policy weights (--policy.path)
#   Wan-AI/Wan2.2-TI2V-5B-Diffusers              frozen VAE + UMT5 text encoder
#   google/umt5-xxl                              tokenizer only (not the XXL weights)
#
# The VAE loader calls from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers") and
# the tokenizer calls from_pretrained("google/umt5-xxl"). Those two must land in
# the HF hub cache (HF_HOME), not only in a plain folder. fastwam_base is also
# copied to pretrain_model/fastwam_base for --policy.path.
#
#   bash scripts_files/download_fastwam.sh
# Re-run the same command to resume. Do not delete pretrain_model/fastwam_base/.cache.
# If the transfer stalls on xet: HF_HUB_DISABLE_XET=1 bash scripts_files/download_fastwam.sh
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_ENDPOINT_PUBLIC="${HF_ENDPOINT_PUBLIC:-https://hf-mirror.com}"
# Non-interactive shells do not source /root/.bashrc. Keep an explicit proxy.
export http_proxy="${http_proxy:-${HTTP_PROXY:-http://10.0.0.112:3128}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-http://10.0.0.112:3128}}"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
# docker exec does not load the interactive PATH where `hf` / `modelscope` live.
export PATH="/worksapce/miniconda3/envs/lerobot/bin:${PATH}:/root/.local/bin"
# ModelScope's CDN is much faster than the HF xet bridge through this proxy
# (~2 MB/s vs ~140 KB/s). HF remains the fallback.
export DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-modelscope}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRETRAIN_DIR="${PRETRAIN_DIR:-${REPO_ROOT}/pretrain_model}"
MODEL_PATHS_FILE="${PRETRAIN_DIR}/.fastwam_paths.env"
mkdir -p "${PRETRAIN_DIR}" "${HF_HOME}"

hf_env() {
    HF_ENDPOINT="${HF_ENDPOINT_PUBLIC}" "$@"
}

require_hf() {
    if ! command -v hf >/dev/null 2>&1; then
        echo "hf CLI not found. Install huggingface_hub and retry." >&2
        exit 1
    fi
}

is_complete_model_dir() {
    local dir="$1"
    [[ -d "${dir}" ]] || return 1
    [[ -f "${dir}/config.json" ]] || return 1
    compgen -G "${dir}/*.safetensors" >/dev/null \
        || compgen -G "${dir}/*.bin" >/dev/null \
        || compgen -G "${dir}/model*.safetensors" >/dev/null \
        || compgen -G "${dir}/model.safetensors.index.json" >/dev/null
}

has_weights() {
    local dir="$1"
    [[ -d "${dir}" ]] || return 1
    compgen -G "${dir}/*.safetensors" >/dev/null \
        || compgen -G "${dir}/*.bin" >/dev/null \
        || compgen -G "${dir}/model*.safetensors" >/dev/null \
        || [[ -f "${dir}/model.safetensors.index.json" ]]
}

# Frozen VAE + text encoder. Do not require a root config.json: weights live in subfolders.
is_complete_wan_diffusers() {
    local dir="$1"
    [[ -f "${dir}/vae/config.json" ]] || return 1
    [[ -f "${dir}/text_encoder/config.json" ]] || return 1
    has_weights "${dir}/vae" || return 1
    has_weights "${dir}/text_encoder" || return 1
}

is_complete_umt5_tokenizer() {
    local dir="$1"
    [[ -f "${dir}/tokenizer_config.json" ]] || return 1
    [[ -f "${dir}/spiece.model" || -f "${dir}/tokenizer.model" || -f "${dir}/tokenizer.json" ]]
}

snapshot_dir_for() {
    local repo_id="$1"
    local cache_name="models--${repo_id//\//--}"
    local rev
    rev="$(cat "${HF_HOME}/hub/${cache_name}/refs/main" 2>/dev/null || true)"
    if [[ -n "${rev}" && -d "${HF_HOME}/hub/${cache_name}/snapshots/${rev}" ]]; then
        echo "${HF_HOME}/hub/${cache_name}/snapshots/${rev}"
        return 0
    fi
    return 1
}

link_snapshot() {
    local repo_id="$1"
    local link_path="$2"
    local snap
    snap="$(snapshot_dir_for "${repo_id}")" || return 1
    ln -sfn "${snap}" "${link_path}"
    echo ">>> Linked ${link_path} -> ${snap}"
}

download_via_modelscope() {
    local repo_id="$1"
    local local_dir="$2"
    shift 2

    if ! command -v modelscope >/dev/null 2>&1; then
        echo ">>> [ModelScope] CLI not found, skipping ${repo_id}"
        return 1
    fi
    mkdir -p "${local_dir}"
    echo ">>> [ModelScope] Downloading ${repo_id} -> ${local_dir}"
    if [[ $# -gt 0 ]]; then
        modelscope download --model "${repo_id}" --local_dir "${local_dir}" --max-workers 8 --include "$@"
    else
        modelscope download --model "${repo_id}" --local_dir "${local_dir}" --max-workers 8
    fi
}

# Point the HF hub cache at a local directory so from_pretrained(repo_id) does
# not fetch the same files again from the slow xet bridge.
publish_hub_snapshot() {
    local repo_id="$1"
    local src="$2"
    local cache_name="models--${repo_id//\//--}"
    local rev="modelscope"
    local snap="${HF_HOME}/hub/${cache_name}/snapshots/${rev}"
    local rel f dest

    mkdir -p "${HF_HOME}/hub/${cache_name}/refs" "${snap}"
    while IFS= read -r -d '' f; do
        rel="${f#${src}/}"
        dest="${snap}/${rel}"
        mkdir -p "$(dirname "${dest}")"
        ln -sfn "${f}" "${dest}"
    done < <(find "${src}" -type f \
        ! -path '*/.cache/*' \
        ! -path '*/.git/*' \
        ! -path '*/._____temp/*' \
        ! -name '*.lock' \
        ! -name '*.metadata' \
        -print0)
    printf '%s' "${rev}" > "${HF_HOME}/hub/${cache_name}/refs/main"
    echo ">>> Published ${repo_id} into ${snap}"
}

download_fastwam_base() {
    local repo_id="lerobot/fastwam_base"
    local local_dir="${PRETRAIN_DIR}/fastwam_base"

    if is_complete_model_dir "${local_dir}"; then
        echo ">>> Skip ${repo_id}: already present at ${local_dir}"
        return 0
    fi

    # Keep a partial directory so the downloader can resume model.safetensors.
    if [[ "${DOWNLOAD_SOURCE}" == "modelscope" || "${DOWNLOAD_SOURCE}" == "auto" ]]; then
        download_via_modelscope "${repo_id}" "${local_dir}"
    fi
    if is_complete_model_dir "${local_dir}"; then
        echo ">>> OK ${repo_id} -> ${local_dir}"
        return 0
    fi
    if [[ "${DOWNLOAD_SOURCE}" == "modelscope" ]]; then
        echo ">>> ${repo_id} is still incomplete after ModelScope" >&2
        return 1
    fi

    if [[ "${DOWNLOAD_SOURCE}" == "hf" || "${DOWNLOAD_SOURCE}" == "auto" ]]; then
        mkdir -p "${local_dir}"
        echo ">>> [HF] Downloading ${repo_id} via ${HF_ENDPOINT_PUBLIC}"
        hf_env hf download "${repo_id}" --local-dir "${local_dir}"
    fi

    if is_complete_model_dir "${local_dir}"; then
        echo ">>> OK ${repo_id} -> ${local_dir}"
        return 0
    fi
    echo ">>> Download finished but ${local_dir} looks incomplete" >&2
    return 1
}

download_into_cache() {
    local repo_id="$1"
    shift
    echo ">>> [HF] Downloading ${repo_id} into ${HF_HOME} via ${HF_ENDPOINT_PUBLIC}"
    hf_env hf download "${repo_id}" "$@" --quiet
}

prepare_local_dir() {
    local local_dir="$1"
    local check_fn="$2"
    if [[ -L "${local_dir}" ]]; then
        local target
        target="$(readlink -f "${local_dir}")"
        if "${check_fn}" "${target}"; then
            return 0
        fi
        rm "${local_dir}"
    fi
}

download_wan_diffusers() {
    local repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    local local_dir="${PRETRAIN_DIR}/Wan2.2-TI2V-5B-Diffusers"
    local snap=""

    prepare_local_dir "${local_dir}" is_complete_wan_diffusers
    if is_complete_wan_diffusers "${local_dir}"; then
        echo ">>> Skip ${repo_id}: already present at ${local_dir}"
        publish_hub_snapshot "${repo_id}" "${local_dir}"
        return 0
    fi
    if snap="$(snapshot_dir_for "${repo_id}")" && is_complete_wan_diffusers "${snap}"; then
        echo ">>> Skip ${repo_id}: already in HF cache at ${snap}"
        ln -sfn "${snap}" "${local_dir}"
        return 0
    fi

    if [[ "${DOWNLOAD_SOURCE}" == "modelscope" || "${DOWNLOAD_SOURCE}" == "auto" ]]; then
        download_via_modelscope "${repo_id}" "${local_dir}" \
            "vae/*" "text_encoder/*" "model_index.json"
    fi
    if is_complete_wan_diffusers "${local_dir}"; then
        publish_hub_snapshot "${repo_id}" "${local_dir}"
        echo ">>> OK ${repo_id} -> ${local_dir}"
        return 0
    fi
    if [[ "${DOWNLOAD_SOURCE}" == "modelscope" ]]; then
        echo ">>> ${repo_id} is still incomplete after ModelScope" >&2
        return 1
    fi

    download_into_cache "${repo_id}" \
        --include "vae/*" \
        --include "text_encoder/*" \
        --include "model_index.json"
    link_snapshot "${repo_id}" "${local_dir}"
    snap="$(snapshot_dir_for "${repo_id}")"
    if is_complete_wan_diffusers "${snap}"; then
        echo ">>> OK ${repo_id} -> ${snap}"
        return 0
    fi
    echo ">>> ${repo_id} cache is missing vae/ or text_encoder/ weights" >&2
    return 1
}

download_umt5_tokenizer() {
    local repo_id="google/umt5-xxl"
    local local_dir="${PRETRAIN_DIR}/umt5-xxl"
    local snap=""

    prepare_local_dir "${local_dir}" is_complete_umt5_tokenizer
    if is_complete_umt5_tokenizer "${local_dir}"; then
        echo ">>> Skip ${repo_id}: tokenizer already present at ${local_dir}"
        publish_hub_snapshot "${repo_id}" "${local_dir}"
        return 0
    fi
    if snap="$(snapshot_dir_for "${repo_id}")" && is_complete_umt5_tokenizer "${snap}"; then
        echo ">>> Skip ${repo_id}: tokenizer already in HF cache at ${snap}"
        ln -sfn "${snap}" "${local_dir}"
        return 0
    fi

    # Tokenizer files only. The XXL weight shards are not used; the text encoder
    # comes from Wan-AI/Wan2.2-TI2V-5B-Diffusers/text_encoder.
    if [[ "${DOWNLOAD_SOURCE}" == "modelscope" || "${DOWNLOAD_SOURCE}" == "auto" ]]; then
        download_via_modelscope "${repo_id}" "${local_dir}" \
            "tokenizer*" "spiece.model" "special_tokens_map.json"
    fi
    if is_complete_umt5_tokenizer "${local_dir}"; then
        publish_hub_snapshot "${repo_id}" "${local_dir}"
        echo ">>> OK ${repo_id} tokenizer -> ${local_dir}"
        return 0
    fi
    if [[ "${DOWNLOAD_SOURCE}" == "modelscope" ]]; then
        echo ">>> ${repo_id} tokenizer files are incomplete after ModelScope" >&2
        return 1
    fi

    download_into_cache "${repo_id}" \
        --include "tokenizer*" \
        --include "spiece.model" \
        --include "special_tokens_map.json"
    link_snapshot "${repo_id}" "${local_dir}"
    snap="$(snapshot_dir_for "${repo_id}")"
    if is_complete_umt5_tokenizer "${snap}"; then
        echo ">>> OK ${repo_id} tokenizer -> ${snap}"
        return 0
    fi
    echo ">>> ${repo_id} tokenizer files are incomplete" >&2
    return 1
}

write_model_paths_file() {
    cat > "${MODEL_PATHS_FILE}" <<EOF
# Generated by scripts_files/download_fastwam.sh
PRETRAIN_DIR=${PRETRAIN_DIR}
HF_HOME=${HF_HOME}
FASTWAM_PRETRAINED_PATH=${PRETRAIN_DIR}/fastwam_base
WAN_DIFFUSERS_PATH=${PRETRAIN_DIR}/Wan2.2-TI2V-5B-Diffusers
UMT5_TOKENIZER_PATH=${PRETRAIN_DIR}/umt5-xxl
EOF
    echo ">>> Wrote ${MODEL_PATHS_FILE}"
}

require_hf
echo ">>> DOWNLOAD_SOURCE=${DOWNLOAD_SOURCE}"
echo ">>> HF_HOME=${HF_HOME}"
echo ">>> HF_ENDPOINT=${HF_ENDPOINT_PUBLIC}"
echo ">>> proxy=${https_proxy}"

download_fastwam_base
download_wan_diffusers
download_umt5_tokenizer
write_model_paths_file

echo ">>> Done."
echo "    Policy:     ${PRETRAIN_DIR}/fastwam_base"
echo "    Wan cache:  ${PRETRAIN_DIR}/Wan2.2-TI2V-5B-Diffusers"
echo "    Tokenizer:  ${PRETRAIN_DIR}/umt5-xxl"
echo "    Source ${MODEL_PATHS_FILE} before training."
