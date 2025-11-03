#!/usr/bin/env bash
# cleanup_generated_outputs.sh - remove generated artifacts, caches, and binaries
# Usage:
#   ./cleanup_generated_outputs.sh            # dry-run (default)
#   ./cleanup_generated_outputs.sh --apply    # perform cleanup
#   ./cleanup_generated_outputs.sh --only caches,chapters
#   ./cleanup_generated_outputs.sh --skip artifacts

set -euo pipefail

# shellcheck disable=SC2034 # referenced via associative arrays
declare -a ALL_CATEGORIES=("chapters" "artifacts" "profiles" "caches" "binaries")
declare -A CATEGORY_DESCRIPTIONS=(
    ["chapters"]="Benchmark, test, and power result folders (all_chapters_results*, benchmark_results, test_results*, power_results)."
    ["artifacts"]="Timestamped drops in artifacts/ (keeps artifacts/golden by default)."
    ["profiles"]="Profiling captures such as profiles/, profiling_results*, and Nsight trace files."
    ["caches"]="Python and TorchInductor caches (__pycache__, *.pyc, .pytest_cache, .torch_inductor)."
    ["binaries"]="Built CUDA binaries and executables emitted next to *.cu sources."
)

declare -A CATEGORY_ENABLED=()
for category in "${ALL_CATEGORIES[@]}"; do
    CATEGORY_ENABLED["$category"]=1
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

DRY_RUN=1
SKIP_PATHS=("artifacts/golden")
declare -a REMOVED_ITEMS=()
declare -A REMOVED_SET=()
declare -A REMOVED_BY_CATEGORY=()

usage() {
    cat <<'EOF'
Usage: cleanup_generated_outputs.sh [options]

Options:
  -y, --apply           Remove files instead of reporting a dry-run.
      --dry-run         Show what would be removed (default).
      --only <cats>     Limit cleanup to comma/space-separated categories.
      --skip <cats>     Skip specific categories.
      --list            Show available categories and exit.
  -h, --help            Display this help message.

Categories:
  chapters  Benchmark, test, and power result folders.
  artifacts Generated artifacts under artifacts/ (keeps artifacts/golden).
  profiles  Profiling captures (profiles/, profiling_results*, Nsight traces).
  caches    Python and TorchInductor caches.
  binaries  Built CUDA executables created next to *.cu sources.
EOF
}

list_categories() {
    echo "Available cleanup categories:"
    for category in "${ALL_CATEGORIES[@]}"; do
        printf "  %-9s %s\n" "$category" "${CATEGORY_DESCRIPTIONS[$category]}"
    done
}

error() {
    echo "cleanup_generated_outputs.sh: $*" >&2
    exit 1
}

validate_category() {
    local category="${1,,}"
    if [[ -z "${CATEGORY_DESCRIPTIONS[$category]+x}" ]]; then
        error "Unknown category '$category'. Use --list to see supported categories."
    fi
    echo "$category"
}

set_only_categories() {
    local input="$1"
    for category in "${ALL_CATEGORIES[@]}"; do
        CATEGORY_ENABLED["$category"]=0
    done
    input="${input//,/ }"
    for token in $input; do
        token="$(validate_category "$token")"
        CATEGORY_ENABLED["$token"]=1
    done
}

skip_categories() {
    local input="$1"
    input="${input//,/ }"
    for token in $input; do
        token="$(validate_category "$token")"
        CATEGORY_ENABLED["$token"]=0
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--apply|--yes)
            DRY_RUN=0
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        --only)
            shift || error "Missing value for --only"
            [[ -z "${1:-}" ]] && error "Missing value for --only"
            set_only_categories "$1"
            ;;
        --skip)
            shift || error "Missing value for --skip"
            [[ -z "${1:-}" ]] && error "Missing value for --skip"
            skip_categories "$1"
            ;;
        --list)
            list_categories
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option '$1'"
            ;;
    esac
    shift
done

remove_path() {
    local category="$1"
    local path="${2#./}"

    [[ -z "$path" || "$path" == "." ]] && return
    for skip in "${SKIP_PATHS[@]}"; do
        if [[ "$path" == "$skip" || "$path" == "$skip/"* ]]; then
            return
        fi
    done
    if [[ ! -e "$path" ]]; then
        return
    fi
    if [[ -n "${REMOVED_SET[$path]+x}" ]]; then
        return
    fi

    if (( DRY_RUN )); then
        echo "[dry-run][$category] rm -rf $path"
    else
        rm -rf "$path"
        echo "[removed][$category] $path"
    fi

    REMOVED_SET["$path"]=1
    REMOVED_ITEMS+=("$path")
    REMOVED_BY_CATEGORY["$category"]=$(( ${REMOVED_BY_CATEGORY["$category"]:-0} + 1 ))
}

remove_glob() {
    local category="$1"
    local pattern="$2"

    shopt -s nullglob dotglob globstar
    local matches=($pattern)
    shopt -u nullglob dotglob globstar

    for match in "${matches[@]}"; do
        remove_path "$category" "${match#./}"
    done
}

remove_find() {
    local category="$1"
    shift
    local -a cmd=("$@")
    cmd+=(-print0)
    while IFS= read -r -d '' entry; do
        remove_path "$category" "${entry#./}"
    done < <("${cmd[@]}")
}

clean_artifacts() {
    local category="artifacts"
    if [[ -d artifacts ]]; then
        shopt -s nullglob dotglob
        for entry in artifacts/*; do
            remove_path "$category" "${entry#./}"
        done
        shopt -u nullglob dotglob
    fi
}

clean_binaries() {
    local category="binaries"
    shopt -s nullglob
    while IFS= read -r -d '' cu_file; do
        local base="${cu_file%.cu}"
        base="${base#./}"
        if [[ -f "$base" && -x "$base" ]]; then
            remove_path "$category" "$base"
        fi
        local variant
        for variant in "${base}"_sm* "${base}"_gb*; do
            [[ -e "$variant" ]] || continue
            remove_path "$category" "$variant"
        done
    done < <(find . -type f -name '*.cu' -print0)

    while IFS= read -r -d '' exe; do
        local rel="${exe#./}"
        if git ls-files --error-unmatch "$rel" > /dev/null 2>&1; then
            continue
        fi
        if file "$rel" 2>/dev/null | grep -q 'ELF'; then
            remove_path "$category" "$rel"
        fi
    done < <(find ch* -maxdepth 1 -type f -perm -111 -print0 2>/dev/null)
    shopt -u nullglob
}

active_categories=()
for category in "${ALL_CATEGORIES[@]}"; do
    if (( CATEGORY_ENABLED["$category"] )); then
        active_categories+=("$category")
    fi
done

if [[ ${#active_categories[@]} -eq 0 ]]; then
    echo "No cleanup categories selected. Use --list to view options."
    exit 0
fi

if (( DRY_RUN )); then
    echo "Running cleanup in dry-run mode. Use --apply to remove files."
else
    echo "Running cleanup (apply mode)."
fi
printf "Active categories: %s\n\n" "${active_categories[*]}"

CHAPTER_PATTERNS=(
    "all_chapters_results"
    "all_chapters_results_bench"
    "all_chapters_results_full"
    "all_chapters_results_smoke"
    "all_chapters_results_tail"
    "benchmark_results"
    "test_results"
    "test_results_*"
    "power_results"
)

PROFILE_DIR_PATTERNS=(
    "profiles"
    "profiling_results"
    "profiling_results_new"
    "ch*/profiling_results"
)

CACHE_DIR_PATTERNS=(
    ".pytest_cache"
)

for category in "${ALL_CATEGORIES[@]}"; do
    (( CATEGORY_ENABLED["$category"] )) || continue
    case "$category" in
        chapters)
            for pattern in "${CHAPTER_PATTERNS[@]}"; do
                remove_glob "$category" "$pattern"
            done
            ;;
        artifacts)
            clean_artifacts
            ;;
        profiles)
            for pattern in "${PROFILE_DIR_PATTERNS[@]}"; do
                remove_glob "$category" "$pattern"
            done
            remove_find "$category" find . -type f -name '*.nsys-rep'
            remove_find "$category" find . -type f -name '*.ncu-rep'
            remove_find "$category" find . -type f -name '*.qdrep'
            remove_find "$category" find . -type f -name '*.sqlite'
            ;;
        caches)
            for pattern in "${CACHE_DIR_PATTERNS[@]}"; do
                remove_glob "$category" "$pattern"
            done
            remove_find "$category" find . -type d -name '__pycache__' -prune
            remove_find "$category" find . -type d -name '.torch_inductor' -prune
            remove_find "$category" find . -type f -name '*.pyc'
            remove_find "$category" find . -type f -name '*.pyo'
            ;;
        binaries)
            clean_binaries
            ;;
    esac
done

echo ""
if (( DRY_RUN )); then
    if [[ ${#REMOVED_ITEMS[@]} -eq 0 ]]; then
        echo "Dry run complete. Nothing would be removed."
    else
        echo "Dry run complete. Items that would be removed:"
        for path in "${REMOVED_ITEMS[@]}"; do
            echo "  $path"
        done
    fi
else
    if [[ ${#REMOVED_ITEMS[@]} -eq 0 ]]; then
        echo "Cleanup complete. No generated artifacts were found."
    else
        echo "Cleanup complete. Removed ${#REMOVED_ITEMS[@]} item(s)."
    fi
fi

if [[ ${#REMOVED_BY_CATEGORY[@]} -gt 0 ]]; then
    echo ""
    echo "Category counts:"
    count=0
    for category in "${active_categories[@]}"; do
        count="${REMOVED_BY_CATEGORY[$category]:-0}"
        printf "  %-9s %d\n" "$category" "$count"
    done
fi
