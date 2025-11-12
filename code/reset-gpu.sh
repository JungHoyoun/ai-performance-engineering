#!/usr/bin/env bash
# Attempts to fully reset NVIDIA GPUs by clearing processes, invoking NVML resets,
# reloading kernel modules, and issuing PCIe function-level resets as needed.

set -Eeuo pipefail

if [[ "${DEBUG:-0}" -ne 0 ]]; then
  set -x
fi

SCRIPT_BASENAME=$(basename "$0")

log() {
  printf '[%s] %s\n' "$SCRIPT_BASENAME" "$*"
}

warn() {
  printf '[%s][WARN] %s\n' "$SCRIPT_BASENAME" "$*" >&2
}

ensure_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      log "Elevating privileges with sudo"
      exec sudo --preserve-env=DEBUG bash "$0" "$@"
    else
      warn "Root privileges are required to reset the GPU"
      exit 1
    fi
  fi
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    warn "Missing required command: $cmd"
    exit 1
  fi
}

declare -a GPU_IDS=()
declare -A GPU_PERSISTENCE_STATE=()
PERSISTENCED_WAS_ACTIVE=0
declare -a DISPLAY_MANAGER_CANDIDATES=(display-manager gdm gdm3 lightdm sddm lxdm)
declare -a STOPPED_DISPLAY_SERVICES=()
declare -a GPU_HELPER_SERVICES=(nvidia-dcgm nvidia-powerd nvidia-fabricmanager nvidia-vgpu-mgr nvidia-vgpud nvidia-gridd nvidia-vgpu-dma)
declare -a STOPPED_GPU_HELPERS=()

discover_gpus() {
  mapfile -t GPU_IDS < <(
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null |
      awk '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); if (length) print}'
  )

  if (( ${#GPU_IDS[@]} == 0 )); then
    warn "No NVIDIA GPUs detected"
    exit 1
  fi

  while IFS=',' read -r idx state; do
    idx=$(awk '{$1=$1};1' <<<"$idx")
    state=$(awk '{$1=$1};1' <<<"$state")
    [[ -n "$idx" ]] && GPU_PERSISTENCE_STATE["$idx"]="$state"
  done < <(nvidia-smi --query-gpu=index,persistence_mode --format=csv,noheader 2>/dev/null || true)
}

stop_persistenced() {
  if command -v systemctl >/dev/null 2>&1; then
    if systemctl is-active --quiet nvidia-persistenced 2>/dev/null; then
      PERSISTENCED_WAS_ACTIVE=1
      if systemctl stop nvidia-persistenced >/dev/null 2>&1; then
        log "Stopped nvidia-persistenced service"
      else
        warn "Failed to stop nvidia-persistenced service"
      fi
    fi
  fi
}

start_persistenced_if_needed() {
  if (( PERSISTENCED_WAS_ACTIVE )); then
    if systemctl start nvidia-persistenced >/dev/null 2>&1; then
      log "Restarted nvidia-persistenced service"
    else
      warn "Failed to restart nvidia-persistenced service"
    fi
  fi
}

stop_gpu_helper_services() {
  local svc svc_unit
  if ! command -v systemctl >/dev/null 2>&1; then
    return
  fi

  for svc in "${GPU_HELPER_SERVICES[@]}"; do
    svc_unit=""
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
      svc_unit="$svc"
    elif systemctl is-active --quiet "${svc}.service" 2>/dev/null; then
      svc_unit="${svc}.service"
    else
      continue
    fi

    if systemctl stop "$svc_unit" >/dev/null 2>&1; then
      STOPPED_GPU_HELPERS+=("$svc_unit")
      log "Stopped NVIDIA helper service $svc_unit"
    else
      warn "Failed to stop NVIDIA helper service $svc_unit"
    fi
  done
}

restart_gpu_helper_services() {
  local svc
  if ! command -v systemctl >/dev/null 2>&1; then
    return
  fi

  for svc in "${STOPPED_GPU_HELPERS[@]}"; do
    if systemctl start "$svc" >/dev/null 2>&1; then
      log "Restarted NVIDIA helper service $svc"
    else
      warn "Failed to restart NVIDIA helper service $svc"
    fi
  done
}

disable_persistence_mode() {
  for id in "${GPU_IDS[@]}"; do
    if nvidia-smi -i "$id" -pm 0 >/dev/null 2>&1; then
      log "Disabled persistence mode on GPU $id"
    else
      warn "Unable to disable persistence mode on GPU $id"
    fi
  done
}

restore_persistence_mode() {
  for id in "${GPU_IDS[@]}"; do
    local original="${GPU_PERSISTENCE_STATE[$id]:-Enabled}"
    local target=1
    if [[ "$original" =~ [Dd]isabled ]]; then
      target=0
    fi
    if nvidia-smi -i "$id" -pm "$target" >/dev/null 2>&1; then
      log "Restored persistence mode ($original) on GPU $id"
    else
      warn "Failed to restore persistence mode on GPU $id"
    fi
  done
}

terminate_residual_display_processes() {
  local -a targets=(Xorg X Xwayland gnome-shell kwin_x11 kwin_wayland sway weston)
  local proc

  if ! command -v pgrep >/dev/null 2>&1; then
    return
  fi

  for proc in "${targets[@]}"; do
    mapfile -t pids < <(pgrep -x "$proc" 2>/dev/null || true)
    if (( ${#pids[@]} == 0 )); then
      continue
    fi
    log "Terminating lingering display process $proc (PIDs: ${pids[*]})"
    for pid in "${pids[@]}"; do
      kill -TERM "$pid" >/dev/null 2>&1 || true
    done
  done

  sleep 1

  for proc in "${targets[@]}"; do
    mapfile -t pids < <(pgrep -x "$proc" 2>/dev/null || true)
    if (( ${#pids[@]} == 0 )); then
      continue
    fi
    warn "Force killing stubborn display process $proc (PIDs: ${pids[*]})"
    for pid in "${pids[@]}"; do
      kill -KILL "$pid" >/dev/null 2>&1 || true
    done
  done
}

stop_display_stack() {
  local svc svc_unit
  if ! command -v systemctl >/dev/null 2>&1; then
    return
  fi

  for svc in "${DISPLAY_MANAGER_CANDIDATES[@]}"; do
    svc_unit=""
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
      svc_unit="$svc"
    elif systemctl is-active --quiet "${svc}.service" 2>/dev/null; then
      svc_unit="${svc}.service"
    else
      continue
    fi

    if systemctl stop "$svc_unit" >/dev/null 2>&1; then
      STOPPED_DISPLAY_SERVICES+=("$svc_unit")
      log "Stopped display manager service $svc_unit"
    else
      warn "Failed to stop display manager service $svc_unit"
    fi
  done

  terminate_residual_display_processes
}

restart_display_stack() {
  local svc
  if ! command -v systemctl >/dev/null 2>&1; then
    return
  fi

  for svc in "${STOPPED_DISPLAY_SERVICES[@]}"; do
    if systemctl start "$svc" >/dev/null 2>&1; then
      log "Restarted display manager service $svc"
    else
      warn "Failed to restart display manager service $svc"
    fi
  done
}

collect_gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
    awk '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); if ($0 ~ /^[0-9]+$/) print $0}'
}

drain_gpu_processes() {
  local max_attempts=3
  local attempt
  local -a pids=()

  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    mapfile -t pids < <(collect_gpu_pids || true)
    if (( ${#pids[@]} == 0 )); then
      log "No compute processes are holding the GPU"
      return 0
    fi

    local signal="TERM"
    if (( attempt == max_attempts )); then
      signal="KILL"
    fi

    log "Attempt $attempt: sending SIG${signal} to GPU processes ${pids[*]}"
    for pid in "${pids[@]}"; do
      if kill "-${signal}" "$pid" >/dev/null 2>&1; then
        :
      else
        warn "Failed to signal PID $pid"
      fi
    done
    sleep 2
  done

  mapfile -t pids < <(collect_gpu_pids || true)
  if (( ${#pids[@]} > 0 )); then
    warn "GPU is still busy with PIDs: ${pids[*]}"
    return 1
  fi
}

gpu_reset_via_nvml() {
  local result=1
  for id in "${GPU_IDS[@]}"; do
    if nvidia-smi --gpu-reset -i "$id" >/dev/null 2>&1; then
      log "nvidia-smi GPU reset succeeded for GPU $id"
      result=0
    else
      warn "nvidia-smi GPU reset is unsupported or failed for GPU $id"
    fi
  done
  return "$result"
}

module_loaded() {
  local module="$1"
  grep -q "^${module} " /proc/modules 2>/dev/null
}

reload_nvidia_modules() {
  local modules_to_remove=(nvidia_drm nvidia_modeset nvidia_uvm nvidia)
  local removed_any=0

  for module in "${modules_to_remove[@]}"; do
    if module_loaded "$module"; then
      removed_any=1
      if modprobe -r "$module" >/dev/null 2>&1; then
        log "Removed kernel module $module"
      else
        warn "Failed to remove kernel module $module"
      fi
    fi
  done

  if (( removed_any == 0 )); then
    log "No NVIDIA kernel modules were loaded prior to reload step"
  fi

  local modules_to_add=(nvidia nvidia_modeset nvidia_uvm nvidia_drm)
  for module in "${modules_to_add[@]}"; do
    if modprobe "$module" >/dev/null 2>&1; then
      log "Loaded kernel module $module"
    else
      warn "Kernel module $module could not be loaded (continuing)"
    fi
  done
}

normalize_bus_id() {
  local bdf="$1"
  if [[ "$bdf" != *:*:*.* ]]; then
    return 1
  fi
  local domain="${bdf%%:*}"
  local rest="${bdf#*:}"
  domain="${domain: -4}"
  local normalized="${domain}:${rest}"
  printf '%s\n' "${normalized,,}"
}

pci_function_level_reset() {
  mapfile -t bus_ids < <(
    nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null |
      awk '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); if (length) print}'
  )

  for raw_id in "${bus_ids[@]}"; do
    local normalized
    if ! normalized=$(normalize_bus_id "$raw_id"); then
      warn "Unable to normalize PCI bus id: $raw_id"
      continue
    fi

    local device_path="/sys/bus/pci/devices/$normalized"
    if [[ ! -d "$device_path" ]]; then
      warn "PCI device path not found: $device_path"
      continue
    fi

    local driver_name=""
    if [[ -L "$device_path/driver" ]]; then
      driver_name=$(basename "$(readlink "$device_path/driver")")
    fi

    if [[ -n "$driver_name" && -w "/sys/bus/pci/drivers/$driver_name/unbind" ]]; then
      if printf '%s\n' "$normalized" >"/sys/bus/pci/drivers/$driver_name/unbind"; then
        log "Unbound $normalized from driver $driver_name"
      else
        warn "Failed to unbind $normalized from driver $driver_name"
      fi
    fi

    if [[ -w "$device_path/reset" ]]; then
      if printf '1\n' >"$device_path/reset"; then
        log "Issued PCIe function-level reset for $normalized"
      else
        warn "Failed to trigger PCIe reset for $normalized"
      fi
    else
      warn "PCI reset interface not available for $normalized"
    fi

    if [[ -n "$driver_name" && -w "/sys/bus/pci/drivers/$driver_name/bind" ]]; then
      if printf '%s\n' "$normalized" >"/sys/bus/pci/drivers/$driver_name/bind"; then
        log "Rebound $normalized to driver $driver_name"
      else
        warn "Failed to rebind $normalized to driver $driver_name"
      fi
    fi
  done
}

cleanup() {
  local rc="$1"
  if (( ${#GPU_IDS[@]} )); then
    restore_persistence_mode || true
  fi
  restart_gpu_helper_services || true
  restart_display_stack || true
  start_persistenced_if_needed || true
  exit "$rc"
}

main() {
  ensure_root "$@"
  require_cmd nvidia-smi
  discover_gpus

  trap 'rc=$?; trap - EXIT; cleanup "$rc"' EXIT

  stop_persistenced
  stop_gpu_helper_services
  stop_display_stack
  disable_persistence_mode
  drain_gpu_processes || warn "Some processes could not be stopped; GPU reset may fail"
  gpu_reset_via_nvml || warn "NVML reset did not complete; attempting kernel module reload"
  reload_nvidia_modules
  pci_function_level_reset

  log "GPU reset sequence completed"
}

main "$@"
