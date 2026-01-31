#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MCP_JSON="${PROJECT_ROOT}/mcp.json"
CODEX_BIN_ARG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --codex-bin)
            if [ $# -lt 2 ]; then
                echo "ERROR: --codex-bin requires a value" >&2
                exit 1
            fi
            CODEX_BIN_ARG="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ ! -f "${MCP_JSON}" ]; then
    echo "ERROR: mcp.json not found at ${MCP_JSON}" >&2
    exit 1
fi

CODEX_CMD=""
if [ -n "${CODEX_BIN_ARG}" ]; then
    CODEX_CMD="${CODEX_BIN_ARG}"
elif [ -n "${CODEX_BIN:-}" ]; then
    CODEX_CMD="${CODEX_BIN}"
elif command -v codex >/dev/null 2>&1; then
    CODEX_CMD="codex"
elif [ -n "${SUDO_USER:-}" ]; then
    user_home="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
    if [ -n "${user_home}" ] && [ -x "${user_home}/.local/bin/codex" ]; then
        CODEX_CMD="${user_home}/.local/bin/codex"
    fi
fi

if [ -z "${CODEX_CMD}" ]; then
    echo "ERROR: codex CLI not found in PATH. Install Codex CLI to configure MCP tools." >&2
    exit 1
fi

if [ "${CODEX_CMD}" != "codex" ] && [ ! -x "${CODEX_CMD}" ]; then
    echo "ERROR: codex binary is not executable at ${CODEX_CMD}" >&2
    exit 1
fi

RUN_PREFIX=()
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
    RUN_PREFIX=(sudo -H -u "${SUDO_USER}")
fi

mapfile -t CONFIG_LINES < <(MCP_JSON="${MCP_JSON}" PROJECT_ROOT="${PROJECT_ROOT}" python3 - <<'PY'
import json
import os
import sys

path = os.environ["MCP_JSON"]
root = os.path.abspath(os.environ["PROJECT_ROOT"])

with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

servers = data.get("mcpServers", {})
cfg = servers.get("aisp")
if not cfg:
    print("ERROR: mcp.json missing mcpServers.aisp", file=sys.stderr)
    sys.exit(1)

def substitute(value):
    if isinstance(value, str):
        return value.replace("${workspaceFolder}", root)
    return value

cwd = substitute(cfg.get("cwd"))
if cwd:
    cwd = os.path.abspath(cwd)
    if cwd != root:
        print(
            f"ERROR: mcp.json aisp.cwd resolves to {cwd}, expected {root}. "
            "Codex MCP does not support custom cwd.",
            file=sys.stderr,
        )
        sys.exit(1)

command = substitute(cfg.get("command"))
if not command:
    print("ERROR: mcp.json missing mcpServers.aisp.command", file=sys.stderr)
    sys.exit(1)

args = [substitute(a) for a in cfg.get("args", [])]
env = cfg.get("env") or {}
env = {k: substitute(v) for k, v in env.items()}

print(f"cmd\t{command}")
for arg in args:
    print(f"arg\t{arg}")
for key in sorted(env.keys()):
    print(f"env\t{key}={env[key]}")
PY
)

EXPECTED_CMD=""
EXPECTED_ARGS=()
EXPECTED_ENVS=()
for line in "${CONFIG_LINES[@]}"; do
    kind="${line%%$'\t'*}"
    value="${line#*$'\t'}"
    case "${kind}" in
        cmd) EXPECTED_CMD="${value}" ;;
        arg) EXPECTED_ARGS+=("${value}") ;;
        env) EXPECTED_ENVS+=("${value}") ;;
    esac
done

if [ -z "${EXPECTED_CMD}" ]; then
    echo "ERROR: Failed to read command from mcp.json" >&2
    exit 1
fi

LIST_JSON="$("${RUN_PREFIX[@]}" "${CODEX_CMD}" mcp list --json)"
HAS_AISP="$(LIST_JSON="${LIST_JSON}" python3 - <<'PY'
import json
import os
import sys

data = json.loads(os.environ["LIST_JSON"])
names = []
if isinstance(data, list):
    for entry in data:
        if isinstance(entry, dict):
            names.append(entry.get("name"))
print("1" if "aisp" in names else "0")
PY
)"

if [ "${HAS_AISP}" -eq 0 ]; then
    echo "Configuring Codex MCP server 'aisp'..."
    add_cmd=("${RUN_PREFIX[@]}" "${CODEX_CMD}" mcp add aisp)
    for env_entry in "${EXPECTED_ENVS[@]}"; do
        add_cmd+=(--env "${env_entry}")
    done
    add_cmd+=(-- "${EXPECTED_CMD}")
    add_cmd+=("${EXPECTED_ARGS[@]}")
    "${add_cmd[@]}"
else
    echo "Codex MCP server 'aisp' already configured; validating..."
fi

ACTUAL_JSON="$("${RUN_PREFIX[@]}" "${CODEX_CMD}" mcp get aisp --json)"
EXPECTED_JSON="$(MCP_JSON="${MCP_JSON}" PROJECT_ROOT="${PROJECT_ROOT}" python3 - <<'PY'
import json
import os
import sys

path = os.environ["MCP_JSON"]
root = os.path.abspath(os.environ["PROJECT_ROOT"])

with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

cfg = data.get("mcpServers", {}).get("aisp")
if not cfg:
    print("ERROR: mcp.json missing mcpServers.aisp", file=sys.stderr)
    sys.exit(1)

def substitute(value):
    if isinstance(value, str):
        return value.replace("${workspaceFolder}", root)
    return value

expected = {
    "command": substitute(cfg.get("command")),
    "args": [substitute(a) for a in cfg.get("args", [])],
    "env": {k: substitute(v) for k, v in (cfg.get("env") or {}).items()},
}

print(json.dumps(expected))
PY
)"

EXPECTED_JSON="${EXPECTED_JSON}" ACTUAL_JSON="${ACTUAL_JSON}" python3 - <<'PY'
import json
import os
import shlex
import sys

expected = json.loads(os.environ["EXPECTED_JSON"])
actual = json.loads(os.environ["ACTUAL_JSON"])

mismatches = []
if actual.get("enabled") is not True:
    mismatches.append("aisp server is disabled")

transport = actual.get("transport") or {}
if transport.get("type") != "stdio":
    mismatches.append(f"transport.type is {transport.get('type')}, expected stdio")
if transport.get("command") != expected["command"]:
    mismatches.append(
        f"command is {transport.get('command')}, expected {expected['command']}"
    )
if transport.get("args") != expected["args"]:
    mismatches.append(f"args are {transport.get('args')}, expected {expected['args']}")

actual_env = transport.get("env") or {}
if actual_env != expected["env"]:
    mismatches.append(f"env is {actual_env}, expected {expected['env']}")

if mismatches:
    print(
        "ERROR: Codex MCP configuration for 'aisp' does not match mcp.json.",
        file=sys.stderr,
    )
    for mismatch in mismatches:
        print(f"  - {mismatch}", file=sys.stderr)

    env_flags = []
    for key in sorted(expected["env"].keys()):
        env_flags.extend(["--env", f"{key}={expected['env'][key]}"])

    add_cmd = [
        "codex",
        "mcp",
        "add",
        "aisp",
        *env_flags,
        "--",
        expected["command"],
        *expected["args"],
    ]

    print("Guidance:", file=sys.stderr)
    print("  codex mcp remove aisp", file=sys.stderr)
    print(f"  {' '.join(shlex.quote(part) for part in add_cmd)}", file=sys.stderr)
    sys.exit(1)

print("Codex MCP configuration verified.")
PY
