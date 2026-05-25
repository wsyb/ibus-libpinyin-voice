#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY_PATH="/usr/libexec/ibus-engine-libpinyin.voice"
STOCK_PATH="/usr/libexec/ibus-engine-libpinyin.stock"
ALTERNATIVE_NAME="ibus-engine-libpinyin"

if [ "$(id -u)" -ne 0 ]; then
    log_error "Please run with sudo: sudo $0"
    exit 1
fi

log_info "=== ibus-libpinyin-voice uninstaller ==="
echo ""

log_info "[1/4] Removing update-alternatives entry..."
update-alternatives --remove "$ALTERNATIVE_NAME" "$BINARY_PATH" > /dev/null 2>&1 || true
log_info "Alternatives entry removed."

log_info "[2/4] Removing voice binary..."
if [ -f "$BINARY_PATH" ]; then
    rm -f "$BINARY_PATH"
    log_info "Removed $BINARY_PATH"
else
    log_warn "$BINARY_PATH not found."
fi

log_info "[3/4] Restoring original binary..."
if [ -f "$STOCK_PATH" ]; then
    if [ -L "/usr/libexec/ibus-engine-libpinyin" ]; then
        rm -f "/usr/libexec/ibus-engine-libpinyin"
    fi
    cp "$STOCK_PATH" "/usr/libexec/ibus-engine-libpinyin"
    rm -f "$STOCK_PATH"
    log_info "Original binary restored."
else
    if [ ! -f "/usr/libexec/ibus-engine-libpinyin" ]; then
        log_warn "No original binary found. You may need to reinstall ibus-libpinyin:"
        log_warn "  sudo apt install --reinstall ibus-libpinyin"
    else
        log_info "Original binary already in place."
    fi
fi

read -p "Remove ASR model (~230MB)? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SUDO_USER_HOME=$(getent passwd "$SUDO_USER" 2>/dev/null | cut -d: -f6 || echo "$HOME")
    MODEL_DIR="$SUDO_USER_HOME/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
    if [ -d "$MODEL_DIR" ]; then
        rm -rf "$MODEL_DIR"
        log_info "Model removed."
    else
        log_warn "Model directory not found."
    fi
else
    log_info "Model kept."
fi

log_info "[4/4] Restarting ibus-daemon..."
SUDO_UID="$(id -u "$SUDO_USER")"
export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$SUDO_UID/bus"
if [ -n "$SUDO_USER" ]; then
    sudo -u "$SUDO_USER" DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
        ibus exit 2>/dev/null || true
    sleep 2
    sudo -u "$SUDO_USER" DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
        nohup ibus-daemon -drx > /dev/null 2>&1 &
    sleep 4
fi

echo ""
log_info "=== Uninstall complete! ==="
echo ""
echo "  Original ibus-libpinyin has been restored."
echo ""
