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
PROJECT_DIR="$SCRIPT_DIR"
BINARY_PATH="/usr/libexec/ibus-engine-libpinyin.voice"
STOCK_PATH="/usr/libexec/ibus-engine-libpinyin.stock"
ALTERNATIVE_NAME="ibus-engine-libpinyin"

MODEL_DIR="$HOME/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
MODEL_BASE_URL="https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx/resolve/master"
MODEL_FILES=("model_quant.onnx" "am.mvn" "tokens.json")

BUILD_DEPS="build-essential autoconf automake libtool pkg-config"
RUNTIME_DEPS="libonnxruntime1.23 libpulse0 libibus-1.0-0 libsqlite3-0 libpinyin15 libopencc1.1 libnotify4"
BUILD_LIBS="libibus-1.0-dev libsqlite3-dev libpinyin-dev libopencc-dev libpulse-dev libonnxruntime-dev libnotify-dev"
OPTIONAL_DEPS="libboost-all-dev"

if [ "$(id -u)" -ne 0 ]; then
    log_error "Please run with sudo: sudo $0"
    exit 1
fi

log_info "=== ibus-libpinyin-voice installer ==="
log_info "Project dir: $PROJECT_DIR"
echo ""

log_info "[1/8] Installing build dependencies..."
apt-get update -qq
apt-get install -y -qq $BUILD_DEPS $BUILD_LIBS > /dev/null 2>&1
log_info "Build dependencies installed."

log_info "[2/8] Initializing submodules..."
cd "$PROJECT_DIR"
if [ -d "third_party/kaldi-native-fbank/.git" ]; then
    git submodule update --init third_party/kaldi-native-fbank > /dev/null 2>&1
fi
if [ -d "third_party/kissfft/.git" ]; then
    git submodule update --init third_party/kissfft > /dev/null 2>&1
fi
if [ ! -d "third_party/kissfft" ]; then
    mkdir -p third_party/kissfft
    git clone --depth 1 https://github.com/mborber/kissfft.git third_party/kissfft > /dev/null 2>&1
fi
log_info "Submodules ready."

log_info "[3/8] Building..."
cd "$PROJECT_DIR"
if [ -f "configure" ]; then
    make clean > /dev/null 2>&1 || true
    find . -name "*.o" -delete 2>/dev/null || true
else
    autoreconf -fi > /dev/null 2>&1
fi
./configure --enable-onnxruntime > /dev/null 2>&1
make -j"$(nproc)" > /dev/null 2>&1
log_info "Build successful."

log_info "[4/8] Stopping ibus processes..."
pkill -9 -f "ibus-engine-libpinyin" 2>/dev/null || true
sleep 1

log_info "Installing binary..."
cp "$PROJECT_DIR/src/ibus-engine-libpinyin" "$BINARY_PATH"
chmod 755 "$BINARY_PATH"
log_info "Binary installed to $BINARY_PATH"

log_info "[5/8] Setting up update-alternatives..."
if [ -f "$STOCK_PATH" ]; then
    log_warn "Stock binary backup already exists at $STOCK_PATH"
elif [ -f "/usr/libexec/ibus-engine-libpinyin" ] && [ ! -L "/usr/libexec/ibus-engine-libpinyin" ]; then
    cp "/usr/libexec/ibus-engine-libpinyin" "$STOCK_PATH"
    log_info "Original binary backed up to $STOCK_PATH"
fi

update-alternatives --install \
    /usr/libexec/ibus-engine-libpinyin \
    "$ALTERNATIVE_NAME" \
    "$BINARY_PATH" \
    100 \
    > /dev/null 2>&1 || true

update-alternatives --set "$ALTERNATIVE_NAME" "$BINARY_PATH" > /dev/null 2>&1 || true

if [ -L "/usr/libexec/ibus-engine-libpinyin" ]; then
    current_target=$(readlink -f /usr/libexec/ibus-engine-libpinyin)
    if [ "$current_target" != "$BINARY_PATH" ]; then
        rm /usr/libexec/ibus-engine-libpinyin
        ln -s "$BINARY_PATH" /usr/libexec/ibus-engine-libpinyin
    fi
elif [ -f "/usr/libexec/ibus-engine-libpinyin" ]; then
    rm /usr/libexec/ibus-engine-libpinyin
    ln -s "$BINARY_PATH" /usr/libexec/ibus-engine-libpinyin
fi

log_info "update-alternatives configured."

log_info "[6/8] Downloading ASR model..."
SUDO_USER_HOME=$(getent passwd "$SUDO_USER" 2>/dev/null | cut -d: -f6 || echo "$HOME")
MODEL_DIR_USER="$SUDO_USER_HOME/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
mkdir -p "$MODEL_DIR_USER"

for f in "${MODEL_FILES[@]}"; do
    if [ -f "$MODEL_DIR_USER/$f" ]; then
        log_info "  $f already exists, skipping."
    else
        log_info "  Downloading $f ..."
        curl -L -C - --progress-bar \
            "$MODEL_BASE_URL/$f" \
            -o "$MODEL_DIR_USER/$f" 2>&1 || {
            log_error "  Failed to download $f"
            exit 1
        }
        chown "$(id -u "$SUDO_USER")":"$(id -g "$SUDO_USER")" "$MODEL_DIR_USER/$f" 2>/dev/null || true
    fi
done
log_info "Model ready."

log_info "[7/8] Configuring as default input method..."
im-config -n ibus > /dev/null 2>&1 || true
SUDO_UID="$(id -u "$SUDO_USER")"
export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$SUDO_UID/bus"
sudo -u "$SUDO_USER" DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
    gsettings set org.freedesktop.ibus.general engines-order "['libpinyin']" > /dev/null 2>&1 || true
sudo -u "$SUDO_USER" DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
    gsettings set org.freedesktop.ibus.general preload-engines "['libpinyin']" > /dev/null 2>&1 || true
log_info "Input method configured."

log_info "[8/8] Restarting ibus-daemon..."
SUDO_UID="$(id -u "$SUDO_USER")"
export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$SUDO_UID/bus"
if pgrep -x ibus-daemon > /dev/null; then
    sudo -u "$SUDO_USER" DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
        ibus exit 2>/dev/null || true
    sleep 2
    sudo -u "$SUDO_USER" DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
        nohup ibus-daemon -drx > /dev/null 2>&1 &
    sleep 4
fi
if pgrep -f "ibus-engine-libpinyin" > /dev/null; then
    log_info "ibus-engine-libpinyin-voice is running."
else
    log_warn "Please log out and log back in to activate Voice Pinyin."
fi

echo ""
log_info "=== Installation complete! ==="
echo ""
echo "  Voice Pinyin has been installed and set as default input method."
echo ""
echo "  Usage:"
echo "    Double-press Right Ctrl  ->  Start recording"
echo "    Hold to speak, release   ->  Stop & recognize"
echo ""
echo "  Switch back to original:"
echo "    sudo update-alternatives --set ibus-engine-libpinyin /usr/libexec/ibus-engine-libpinyin.stock"
echo ""
echo "  Or restore original completely:"
echo "    sudo $SCRIPT_DIR/uninstall.sh"
echo ""
