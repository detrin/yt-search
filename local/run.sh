#!/bin/bash

# --- Configuration Section ---
# These can be moved to a separate config file if needed
NUM_YT_RESULTS=100
MAX_RECURSION=100
TOP_K=10
LOG_DIR="logs"
OUTPUT_DIR="outputs"
TMP_DIRS=("data/tmp_srt" "data/tmp_txt")
PYTHON_VERSION="3.12"
REQUIREMENTS="requirements.txt"

# --- Initialize Environment ---
set -eo pipefail
SCRIPT_NAME=$(basename "$0")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Setup Directories ---
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# --- Logging Setup ---
exec > >(tee -a "${LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log")
exec 2>&1

log() {
    local level=$1
    local message=$2
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [${level}] ${message}"
}

# --- Cleanup Handler ---
cleanup() {
    log "INFO" "Cleaning up temporary directories"
    for dir in "${TMP_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            rm -rf "$dir"
            log "DEBUG" "Removed directory: $dir"
        fi
    done
}

trap cleanup EXIT

# --- Dependency Check ---
check_python() {
    if ! command -v "python${PYTHON_VERSION}" &> /dev/null; then
        log "ERROR" "Python ${PYTHON_VERSION} not found. Please install it first."
        exit 1
    fi
}

# --- Input Validation ---
validate_input() {
    if [ -z "$question" ]; then
        log "ERROR" "No question provided. Usage: $0 <question>"
        exit 1
    fi
}


question=$1
validate_input

# Create temporary directories
for dir in "${TMP_DIRS[@]}"; do
    mkdir -p "$dir"
    log "DEBUG" "Created directory: $dir"
done

# Check dependencies
check_python

# Processing pipeline
log "INFO" "Starting processing pipeline"

log "INFO" "Downloading subtitles"
if ! bash ./download_yt_subtitles.sh -s "$question" -n "$NUM_YT_RESULTS" -o data/tmp_srt; then
    log "ERROR" "Subtitle download failed"
    exit 1
fi

log "INFO" "Converting subtitles to text"
if ! bash ./convert_srt_to_txt.sh data/tmp_srt data/tmp_txt; then
    log "ERROR" "Subtitle conversion failed"
    exit 1
fi

# Run RAG pipeline and capture output
log "INFO" "Running RAG pipeline"
OUTPUT_FILE="${OUTPUT_DIR}/output_${TIMESTAMP}.txt"

"python${PYTHON_VERSION}" rag.py \
    --input_folder data/tmp_txt \
    --question "$question" \
    --top_k "$TOP_K" \
    --max_recursion "$MAX_RECURSION" 
