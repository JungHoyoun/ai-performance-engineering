#!/usr/bin/env bash
set -euo pipefail

# Dashboard setup: install Node.js 18+ and frontend dependencies.
# Usage:
#   ./setup-dashboard.sh
# Env:
#   NODE_INSTALL_METHOD=auto|apt|nvm (default: auto)
#   SKIP_NODE_INSTALL=1 to skip installing Node.js
#   SKIP_NPM_INSTALL=1 to skip running npm install

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="${ROOT_DIR}/dashboard/web"
NODE_MIN_MAJOR=18
NODE_INSTALL_METHOD="${NODE_INSTALL_METHOD:-auto}"

echo "AI Performance Engineering Dashboard Setup"
echo "=========================================="

get_node_major() {
  if ! command -v node >/dev/null 2>&1; then
    echo ""
    return 0
  fi
  node -v | sed 's/^v//' | cut -d. -f1
}

install_node_apt() {
  if ! command -v sudo >/dev/null 2>&1; then
    echo "[error] sudo is required to install Node.js via apt." >&2
    return 1
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[error] apt-get not found. Cannot install Node.js via apt." >&2
    return 1
  fi
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "${ID:-}" != "ubuntu" ] && [ "${ID:-}" != "debian" ]; then
      echo "[warn] Detected ${ID:-unknown}; apt packages may not provide Node.js ${NODE_MIN_MAJOR}+." >&2
    fi
  fi
  echo "[setup] Installing Node.js via apt..."
  sudo apt-get update
  sudo apt-get install -y nodejs npm
}

install_node_nvm() {
  if ! command -v curl >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
      echo "[setup] Installing curl..."
      sudo apt-get update
      sudo apt-get install -y curl
    else
      echo "[error] curl not found. Install curl to bootstrap nvm." >&2
      return 1
    fi
  fi

  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if [ ! -s "${NVM_DIR}/nvm.sh" ]; then
    echo "[setup] Installing nvm..."
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
  fi

  # shellcheck disable=SC1090
  . "${NVM_DIR}/nvm.sh"
  echo "[setup] Installing Node.js ${NODE_MIN_MAJOR} via nvm..."
  nvm install "${NODE_MIN_MAJOR}"
  nvm alias default "${NODE_MIN_MAJOR}"
  nvm use "${NODE_MIN_MAJOR}" >/dev/null
}

node_major="$(get_node_major)"
if [ -n "${node_major}" ] && [ "${node_major}" -ge "${NODE_MIN_MAJOR}" ]; then
  echo "[ok] Node.js ${node_major} detected"
else
  if [ "${SKIP_NODE_INSTALL:-0}" -eq 1 ]; then
    echo "[warn] Node.js ${NODE_MIN_MAJOR}+ required. Install manually or rerun without SKIP_NODE_INSTALL=1." >&2
    exit 1
  fi

  case "${NODE_INSTALL_METHOD}" in
    apt)
      install_node_apt || exit 1
      ;;
    nvm)
      install_node_nvm || exit 1
      ;;
    auto)
      if command -v apt-get >/dev/null 2>&1; then
        install_node_apt || true
      fi
      node_major="$(get_node_major)"
      if [ -z "${node_major}" ] || [ "${node_major}" -lt "${NODE_MIN_MAJOR}" ]; then
        echo "[info] apt did not provide Node.js ${NODE_MIN_MAJOR}+; switching to nvm." >&2
        install_node_nvm || exit 1
      fi
      ;;
    *)
      echo "[error] Unknown NODE_INSTALL_METHOD=${NODE_INSTALL_METHOD}. Use auto, apt, or nvm." >&2
      exit 1
      ;;
  esac
fi

node_major="$(get_node_major)"
if [ -z "${node_major}" ] || [ "${node_major}" -lt "${NODE_MIN_MAJOR}" ]; then
  echo "[error] Node.js ${NODE_MIN_MAJOR}+ not available after install. Please install manually." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[error] npm not found after Node.js install. Please install npm manually." >&2
  exit 1
fi

if [ "${SKIP_NPM_INSTALL:-0}" -eq 1 ]; then
  echo "[ok] Skipping npm install (SKIP_NPM_INSTALL=1)"
  exit 0
fi

if [ ! -d "${APP_DIR}" ]; then
  echo "[error] Dashboard app directory not found: ${APP_DIR}" >&2
  exit 1
fi

echo "[setup] Installing dashboard npm dependencies..."
(cd "${APP_DIR}" && npm install)
echo "[ok] Dashboard dependencies installed."
