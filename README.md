# ibus-libpinyin

> 基于 libpinyin 的 IBus 智能拼音输入法引擎

ibus-libpinyin 为 IBus 框架提供智能拼音和注音输入法，内置离线语音输入功能（基于 ONNX Runtime + Paraformer 模型）。

---


无需联网，无需外接服务，**离线运行**的语音识别输入。

- ⚡ **快如闪电** — 松开右侧 Control 键后文字即刻上屏，推理通常在 **20~80ms** 内完成，比其他 Linux 语音输入方案（Whisper、Vosk 等）快一个数量级
- 🌐 **中英文混合识别** — 流畅支持中英混杂语音，如「今天天气怎么样 hello world」
- 💻 **不需要 GPU** — 纯 CPU 运行，使用量化 ONNX 模型，普通笔记本也能跑
- 🔌 **完全离线** — 本地 ONNX 推理，不向任何服务器发送音频数据

### 与其他 Linux 语音输入方案的对比

| 特性 | ibus-libpinyin (本方案) | Whisper (本地) | Vosk |
|------|------------------------|----------------|------|
| 推理速度 | **20~80ms** | 1~10s | 200~500ms |
| 是否需要 GPU | **不需要，纯 CPU** | 推荐 GPU | 不需要 |
| 中英文混合 | ✅ 原生支持 | ✅ 支持 | ⚠️ 需要额外模型 |
| 实时显示 | ❌ 说完后出字 | ❌ | ✅ |
| 标点符号 | ❌ 暂不支持自动标点 | ✅ | ❌ |
| 集成方式 | IBus 输入法引擎 | 独立程序/管道 | 独立程序 |
| 离线运行 | ✅ | ✅ | ✅ |

> **核心优势**：在纯 CPU 环境下，推理速度比 Whisper 快 **10~50 倍**，比 Vosk 快 **3~10 倍**，同时保持中英文混合识别能力。

### 架构概览

```
键盘（长按右侧 Control）
    ↓
ibus-libpinyin 引擎
    ↓
PulseAudio 录音 ─→ FBank 特征提取 ─→ ONNX Runtime ─→ 文本候选
                      (kaldi-native-fbank)    (Paraformer 模型)
```

### 触发方式

| 操作 | 行为 |
|------|------|
| **长按键盘右侧 Control 键** | 提示音后开始录音（来自麦克风） |
| **松开右侧 Control 键** | 录音结束 → 本地识别 → 上屏 |
| 录音中松开再按下 Control | 忽略（防止重复触发） |

> **注意**：长按键盘右侧的 Control 键即可开始录音，松开后结束录音。按住的时长即为录音时长。

### 已知限制

- **不支持实时显示** — 不是边说边出字，而是松开右侧 Control 键后一次性输出结果
- **不支持标点符号自动识别** — 目前输出标点全部都是逗号（，），不会自动识别句号、问号等
- **需要手动下载模型** — 语音模型需从 ModelScope 下载（约 238MB），首次使用需手动放置

### 运行流程

1. **按键检测** — 引擎检测到右侧 Control 键长按时启动录音
2. **PulseAudio 录音** — `startRecording()` 启动 PulseAudio 异步采集，16kHz 16bit 单声道
3. **特征提取** — `extractFeatures()` 使用 kaldi-native-fbank 计算 80 维 FBank → LFR(7,6) 拼接 → CMVN 归一化
4. **ONNX 推理（极快）** — `transcribe()` 将特征送入 Paraformer 量化模型（`session.Run`），输出 logits → argmax 解码 → token 合并。量化模型推理通常在 **20~80ms** 内完成，松开右侧 Control 键后文本即刻上屏
5. **提交文本** — 识别结果直接上屏，若末尾无标点则自动补「，」

### 模型文件

语音模型和标点模型需从 ModelScope 下载，`install.sh` 会自动下载。

**ASR 语音识别模型**（Paraformer，必需）：

```
~/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx/
```

| 文件 | 说明 |
|------|------|
| `model_quant.onnx` | Paraformer 量化模型（约 228MB） |
| `am.mvn` | CMVN 均值和方差文件 |
| `tokens.json` | 词汇表（8404 字符） |

**SenseVoice 模型**（可选，更好的中英混合识别）：

```
~/.cache/modelscope/hub/models/iic/SenseVoiceSmall-onnx/
```

| 文件 | 说明 |
|------|------|
| `model_quant.onnx` | SenseVoice 量化模型（约 231MB） |
| `am.mvn` | CMVN 均值和方差文件 |
| `tokens.json` | 词汇表（25055 字符） |

**标点符号模型**（可选，自动添加标点）：

```
~/.cache/modelscope/hub/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx/
```

| 文件 | 说明 |
|------|------|
| `model_quant.onnx` | 标点模型量化版（约 270MB） |
| `tokens.json` | 词汇表 |

> **注意**：如果 SenseVoice 或标点模型不存在，引擎会自动回退到 Paraformer + 默认逗号模式。

### 诊断方法

如果语音功能无反应，请按以下步骤排查：

```bash
# 1. 查看调试日志
tail -f /tmp/vocotype-voice.log

# 2. 用 wev 检测右侧 Control 按键事件（Wayland）
wev | grep -A3 "key:"

# 3. 用 evtest 查看内核键码（需要 sudo）
sudo evtest
```

日志输出示例：
```
VoiceInput: initializing
VoiceInput: ONNX model loaded from /home/user/.cache/.../model_quant.onnx
VoiceInput: keyval=65508(0xffe3) keycode=29 modifiers=0x9
VoiceInput: recording started
VoiceInput: recording stopped, 48000 samples, join=0ms
VoiceInput: features extracted, frames=250, dim=560
VoiceInput: Session::Run took 123ms
VoiceInput: result='今天天气怎么样 hello world，'
```

如果右侧 Control 键事件异常，可检查键盘布局或通过 evtest 确认 Control 键的 keycode。

### 编译依赖

语音功能需要额外安装：

```bash
# ONNX Runtime（系统包或预编译库）
sudo apt install libonnxruntime-dev

# PulseAudio 音频库
sudo apt install libpulse-dev

# 编译时需 --enable-onnxruntime 选项
./configure --enable-onnxruntime
```

> 语音功能对应的编译条件在 `configure.ac` 中通过 `PKG_CHECK_MODULES(ONNXRUNTIME, [onnxruntime >= 1.17.0])` 检测，`Makefile.am` 中使用 `ENABLE_ONNXRUNTIME` 条件开关。

---

## ⌨️ 输入法

ibus-libpinyin 内置多种输入模式，通过输入法属性切换：

| 模式 | 说明 |
|------|------|
| **全拼** | 标准汉语拼音输入 |
| **双拼** | 支持自然码、微软双拼等多种方案 |
| **注音** | 注音符号（Bopomofo）输入 |
| **英文** | 智能英文输入（含英文单词候选） |
| **表形码** | Table 输入模式 |
| **云输入** | 联网候选补全（需编译 `--enable-cloud-input-mode`） |

### Lua 扩展

支持 Lua 脚本扩展输入功能（需编译 `--enable-lua-extension`），可实现自定义转换器、触发器。

---

## 🔧 编译安装

### 依赖

```bash
sudo apt install ibus libpinyin-dev libpinyin-utils libsqlite3-dev \
                 libglib2.0-dev libgtk-3-dev libibus-1.0-dev \
                 python3 lua5.1 liblua5.1-dev gettext

# 语音功能额外依赖（可选）
sudo apt install libonnxruntime-dev libpulse-dev
```

### 编译

```bash
git clone https://github.com/libpinyin/ibus-libpinyin.git
cd ibus-libpinyin
./autogen.sh

# 不含语音功能
./configure --prefix=/usr

# 含语音功能
./configure --prefix=/usr --enable-onnxruntime

make -j$(nproc)
sudo make install
ibus restart
# 然后在 IBus 首选项中添加「智能拼音」或「LibPinyin」
```

### 编译选项

| 选项 | 功能 |
|------|------|
| `--enable-onnxruntime` | 启用语音输入 |
| `--enable-cloud-input-mode` | 启用云端候选 |
| `--enable-lua-extension` | 启用 Lua 脚本扩展 |
| `--enable-opencc` | 繁简转换（OpenCC） |
| `--enable-libnotify` | 通知提示 |
| `--enable-boost` | 使用 Boost 替代 C++0x |

### 卸载

编译安装后，使用以下命令卸载：

```bash
cd ibus-libpinyin
sudo make uninstall
ibus restart
```

> 如果 `make uninstall` 不可用（例如 `Makefile` 已丢失），可手动删除安装的文件。安装文件通常位于 `/usr/lib/ibus-engine-libpinyin`、`/usr/lib/ibus-setup-libpinyin`、`/usr/share/ibus-libpinyin/` 和 `/usr/share/ibus/component/` 下。

### 恢复原版 ibus-libpinyin

如果系统原本通过包管理器安装了 ibus-libpinyin，手动编译安装后想恢复原版：

```bash
# 1. 卸载手动编译的版本
sudo make uninstall  # 在编译目录中执行

# 2. 重新安装发行版的原版包
sudo apt install --reinstall ibus-libpinyin   # Ubuntu / Debian
sudo dnf reinstall ibus-libpinyin              # Fedora
sudo pacman -S ibus-libpinyin                  # Arch Linux

# 3. 重启 IBus
ibus restart
```

---

## ⚙️ 配置

通过 `ibus-setup-libpinyin` 图形界面配置：

```bash
ibus-setup-libpinyin
```

可配置项包括：
- 拼音方案（全拼 / 双拼 / 注音）
- 候选词数量
- 简繁切换
- 模糊音
- 云输入开关
- Lua 扩展管理

---

## 📁 项目结构

```
ibus-libpinyin/
├── src/                    # 核心引擎
│   ├── PYPPinyinEngine.*   # 拼音引擎主入口
│   ├── PYVoiceInput.*      # 语音输入模块
│   ├── FeatureExtractor.h  # FBank 特征提取
│   ├── PYFullPinyinEditor.*| 全拼编辑器
│   ├── PYPDoublePinyinEditor.*| 双拼编辑器
│   ├── PYPBopomofoEditor.* | 注音编辑器
│   ├── PYPEmojiCandidates.*| emoji 候选
│   ├── PYPSuggestionEditor.* | 联想候选
│   └── ...
├── third_party/
│   ├── kaldi-native-fbank/ # 音频特征提取库
│   └── kissfft/            # FFT 库
├── setup/                  # 配置界面
├── lua/                    # Lua 扩展
├── data/                   # 数据文件
├── po/                     # 翻译文件
├── configure.ac            # Autotools 构建配置
└── Makefile.am             # 顶层构建文件
```

---

## 📜 许可

GNU General Public License v2 或更高版本（GPLv2+）。

## 👤 作者

- Felix Yin \<ybkk1027@gmail.com\>
- Peng Huang \<shawn.p.huang@gmail.com\>
- BYVoid \<byvoid1@gmail.com\>
- Peng Wu \<alexepico@gmail.com\>

---

## 🔗 资源

- 上游仓库：<https://github.com/libpinyin/ibus-libpinyin>
- 问题反馈：<https://github.com/libpinyin/ibus-libpinyin/issues>
- 语音模型：ModelScope — [speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx)
