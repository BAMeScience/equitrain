#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  00_download_omat24.sh --split <train|val|salex> [--all | <subset>...]
  00_download_omat24.sh --split <salex> --all

Notes:
  - train/val subsets: rattled-1000, rattled-1000-subsampled, rattled-500,
    rattled-500-subsampled, rattled-300, rattled-300-subsampled,
    aimd-from-PBE-1000-npt, aimd-from-PBE-1000-nvt,
    aimd-from-PBE-3000-npt, aimd-from-PBE-3000-nvt, rattled-relax
  - salex split: train or val (use --split salex and subset train/val)

Examples:
  00_download_omat24.sh --split train rattled-1000 rattled-relax
  00_download_omat24.sh --split val --all
  00_download_omat24.sh --split salex train
USAGE
}

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required" >&2
  exit 1
fi

SPLIT=""
SUBSETS=()
ALL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      SPLIT="$2"; shift 2 ;;
    --all)
      ALL=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      SUBSETS+=("$1"); shift ;;
  esac
done

if [[ -z "$SPLIT" ]]; then
  echo "error: --split is required" >&2
  usage
  exit 1
fi

TRAIN_BASE="https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train"
VAL_BASE="https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val"
SALEX_BASE="https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex"

TRAIN_SUBSETS=(
  rattled-1000
  rattled-1000-subsampled
  rattled-500
  rattled-500-subsampled
  rattled-300
  rattled-300-subsampled
  aimd-from-PBE-1000-npt
  aimd-from-PBE-1000-nvt
  aimd-from-PBE-3000-npt
  aimd-from-PBE-3000-nvt
  rattled-relax
)

VAL_SUBSETS=(
  rattled-1000
  rattled-1000-subsampled
  rattled-500
  rattled-500-subsampled
  rattled-300
  rattled-300-subsampled
  aimd-from-PBE-1000-npt
  aimd-from-PBE-1000-nvt
  aimd-from-PBE-3000-npt
  aimd-from-PBE-3000-nvt
  rattled-relax
)

mkdir -p train val salex

fetch() {
  local base="$1"
  local subset="$2"
  local target_dir="$3"
  local url="${base}/${subset}.tar.gz"
  local out="${target_dir}/${subset}.tar.gz"
  if [[ -f "$out" ]]; then
    echo "skip: $out exists"
    return
  fi
  echo "downloading: $url"
  curl -L --fail --retry 3 --retry-delay 5 -o "$out" "$url"
}

case "$SPLIT" in
  train)
    if $ALL; then
      SUBSETS=("${TRAIN_SUBSETS[@]}")
    fi
    if [[ ${#SUBSETS[@]} -eq 0 ]]; then
      echo "error: no subsets provided" >&2
      usage
      exit 1
    fi
    for subset in "${SUBSETS[@]}"; do
      fetch "$TRAIN_BASE" "$subset" train
    done
    ;;
  val)
    if $ALL; then
      SUBSETS=("${VAL_SUBSETS[@]}")
    fi
    if [[ ${#SUBSETS[@]} -eq 0 ]]; then
      echo "error: no subsets provided" >&2
      usage
      exit 1
    fi
    for subset in "${SUBSETS[@]}"; do
      fetch "$VAL_BASE" "$subset" val
    done
    ;;
  salex)
    if $ALL; then
      SUBSETS=("train" "val")
    fi
    if [[ ${#SUBSETS[@]} -eq 0 ]]; then
      echo "error: no subsets provided (train|val)" >&2
      usage
      exit 1
    fi
    for subset in "${SUBSETS[@]}"; do
      if [[ "$subset" != "train" && "$subset" != "val" ]]; then
        echo "error: salex subset must be train or val" >&2
        exit 1
      fi
      fetch "$SALEX_BASE" "$subset" salex
    done
    ;;
  *)
    echo "error: unknown split '$SPLIT'" >&2
    usage
    exit 1
    ;;
esac
