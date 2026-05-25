# ibus-libpinyin

> 基于 libpinyin 的 IBus 智能拼音输入法引擎

ibus-libpinyin 为 IBus 框架提供智能拼音和注音输入法，内置离线语音输入功能（基于 ONNX Runtime + Paraformer 模型）。

---

## 🎤 语音输入功能（核心特色）

无需联网，无需外接服务，**离线运行**的语音识别输入。

- ⚡ **毫秒级响应** — 松开 Ctrl 后即刻上屏，几乎无感知等待
- 🌐 **中英文混合识别** — 流畅支持中英混杂语音，如「今天天气怎么样 hello world」
- 🔌 **完全离线** — 本地 ONNX 推理，不向任何服务器发送音频数据

### 架构概览

```
键盘（双击 Ctrl 后按住）
    ↓
ibus-libpinyin 引擎
    ↓
PulseAudio 录音 ─→ FBank 特征提取 ─→ ONNX Runtime ─→ 文本候选
                      (kaldi-native-fbank)    (Paraformer 模型)
```

### 触发方式

| 操作 | 行为 |
|------|------|
| **双击 Ctrl：快速点按一次，第二次按下并保持** | 提示音后开始录音（来自麦克风） |
| **松开 Ctrl** | 录音结束 → 本地识别 → 上屏 |
| 录音中再次操作 Ctrl | 忽略（防止重复触发） |

> **注意**：触发方式是对 Ctrl 键双击 + 按住，不是双击后立刻启动，也不是单按。两次点按需要在 400ms 内完成，第二次按下的保持时长即为录音时长。

### 运行流程

1. **按键检测** — 引擎检测到 400ms 内第二次 Ctrl 按下时启动录音
2. **PulseAudio 录音** — `startRecording()` 启动 PulseAudio 异步采集，16kHz 16bit 单声道
3. **特征提取** — `extractFeatures()` 使用 kaldi-native-fbank 计算 80 维 FBank → LFR(7,6) 拼接 → CMVN 归一化
4. **ONNX 推理（毫秒级）** — `transcribe()` 将特征送入 Paraformer 模型（`session.Run`），输出 logits → argmax 解码 → token 合并 → 标点预测。量化模型推理通常在 **100~200ms** 内完成，松开 Ctrl 后文本即刻上屏
5. **提交文本** — 识别结果回填到输入法候选，若末尾无标点则自动补「，」

### 模型文件

语音模型需从 ModelScope 下载，放置到：

```
~/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx/
```

所需文件：

| 文件 | 说明 |
|------|------|
| `model_quant.onnx` / `model.int8.onnx` | Paraformer 量化模型（约 70MB） |
| `am.mvn` | CMVN 均值和方差文件 |
| `tokens.json` / `tokens.txt` | 词汇表（8404 字符） |

### 诊断方法

如果语音功能无反应，请按以下步骤排查：

```bash
# 1. 查看调试日志
tail -f /tmp/vocotype-voice.log

# 2. 用 wev 检测 Ctrl 按键事件（Wayland）
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

如果 Ctrl 键事件异常，可检查键盘布局或通过 evtest 确认 Ctrl 键的 keycode。

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
