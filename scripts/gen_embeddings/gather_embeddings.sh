#!/bin/bash
set -e

# ============================================================================
# Gene Embeddings Generation Script (GenePT-only)
# ============================================================================
# TusoPerturb's 11,680-D feature stack uses a single LLM embedding block:
# GenePT (OpenAI text-embedding-3-large over per-gene summaries). This driver
# runs the GenePT generator and, optionally, transfers precomputed GenePT
# embeddings from a reference AnnData to a new target AnnData that shares
# genes (much faster than regenerating).
#
# Usage:
#   bash gather_embeddings.sh INPUT_H5AD OUTPUT_H5AD [REFERENCE_H5AD]
#
# Arguments:
#   INPUT_H5AD       - Path to input .h5ad file (target dataset)
#   OUTPUT_H5AD      - Path where the AnnData with embeddings is written
#   REFERENCE_H5AD   - (Optional) Path to a reference .h5ad already containing
#                      GenePT embeddings in .uns["embeddings_genept"]. If
#                      provided, embeddings are transferred instead of
#                      regenerated.
#
# Flags:
#   --force          Regenerate GenePT even if a cached output exists
#
# Requires:
#   - OPENAI_API_KEY env var (for the text-embedding-3-large endpoint)
#   - NCBI Entrez access (for gene-summary fetching; unauthenticated is fine
#     but rate-limited — set NCBI_API_KEY to raise the limit)
# ============================================================================

# Get the directory where this script is located (allows running from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2; }

# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
FORCE_REGENERATE=false
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        FORCE_REGENERATE=true
    else
        ARGS+=("$arg")
    fi
done

if [ ${#ARGS[@]} -lt 2 ] || [ ${#ARGS[@]} -gt 3 ]; then
    log_error "Usage: bash $0 INPUT_H5AD OUTPUT_H5AD [REFERENCE_H5AD] [--force]"
    exit 1
fi

INPUT_H5AD="${ARGS[0]}"
OUTPUT_H5AD="${ARGS[1]}"
REFERENCE_H5AD="${ARGS[2]:-}"

log "============================================================================"
log "TusoPerturb GenePT embedding generation"
log "============================================================================"
log "Input:     $INPUT_H5AD"
log "Output:    $OUTPUT_H5AD"
if [ -n "$REFERENCE_H5AD" ]; then
    log "Reference: $REFERENCE_H5AD  (transfer mode)"
fi
log ""

# ----------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------
if [ ! -f "$INPUT_H5AD" ]; then
    log_error "Input file does not exist: $INPUT_H5AD"
    exit 1
fi
mkdir -p "$(dirname "$OUTPUT_H5AD")"

# ----------------------------------------------------------------------------
# Mode 1: reference transfer (fast path)
# ----------------------------------------------------------------------------
if [ -n "$REFERENCE_H5AD" ]; then
    if [ ! -f "$REFERENCE_H5AD" ]; then
        log_error "Reference file does not exist: $REFERENCE_H5AD"
        exit 1
    fi
    log "Transferring GenePT embeddings from reference..."
    python "$SCRIPT_DIR/transfer_reference_gene_embeddings.py" \
        --input "$INPUT_H5AD" \
        --reference_adata "$REFERENCE_H5AD" \
        --output "$OUTPUT_H5AD"
    log "Done. Output written to: $OUTPUT_H5AD"
    exit 0
fi

# ----------------------------------------------------------------------------
# Mode 2: full GenePT generation
# ----------------------------------------------------------------------------
if [ -z "${OPENAI_API_KEY:-}" ]; then
    log_error "OPENAI_API_KEY is not set. GenePT calls the OpenAI embeddings API."
    exit 1
fi

if [ -f "$OUTPUT_H5AD" ] && [ "$FORCE_REGENERATE" = false ]; then
    log "Output file already exists — skipping (use --force to regenerate)."
    log "Existing file: $OUTPUT_H5AD"
    exit 0
fi

OUTPUT_TMP="${OUTPUT_H5AD%.h5ad}.tmp.h5ad"
rm -f "$OUTPUT_TMP"

log "Generating GenePT embeddings..."
python "$SCRIPT_DIR/generate_genept_gene_embeddings.py" \
    --input "$INPUT_H5AD" \
    --output "$OUTPUT_TMP"

# Atomic move
mv "$OUTPUT_TMP" "$OUTPUT_H5AD"
log "Done. Output written to: $OUTPUT_H5AD"
