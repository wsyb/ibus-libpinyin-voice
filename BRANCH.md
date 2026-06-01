# feat/punctuation-model 分支说明

## 相对于 main 分支的改动

### 新增功能

- **标点符号自动识别** — 接入 `punc_ct-transformer` 标点模型，中文语音识别后自动添加 `，。？、` 标点
- **中英文混合优化** — 标点模型仅对中文字符生效，英文部分不会被错误插入标点
- **移除 Ctrl+J 换行** — 语音识别后不再自动发送换行符

### 文档更新

- README 新增与 Whisper、Vosk 的速度对比表格
- README 新增"已知限制"章节（不支持实时显示、标点符号）
- README 更新模型文件说明（Paraformer + SenseVoice + 标点模型）
- README 更新推理速度数据（实际 20~80ms，非之前写的 100~200ms）

### 安装脚本更新

- `install.sh` 新增标点模型自动下载
- 修正使用说明（"双击"改为"长按"）

### 模型信息

| 模型 | 用途 | 大小 | 状态 |
|------|------|------|------|
| Paraformer | 语音识别 | 228MB | ✅ 使用中 |
| punc_ct-transformer | 标点符号 | 270MB | ✅ 使用中 |
| SenseVoice | 更好的中英文识别 | 231MB | ❌ 暂时禁用（解码问题） |

### 未解决的问题

- SenseVoice ONNX 解码只输出标点不输出文字，原因待查
- Paraformer 英文单词之间没有空格（tokenizer 限制）
