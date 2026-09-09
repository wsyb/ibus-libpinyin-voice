# ibus-libpinyin-voice 更新指南

## 仓库结构

- **upstream** = `libpinyin/ibus-libpinyin`（上游原始项目）
- **origin** = `wsyb/ibus-libpinyin-voice`（你的 fork）
- **main** 分支：基础代码 + 语音功能核心提交
- **feat/punctuation-model** 分支：语音输入 + 标点模型功能（当前开发分支）

## 更新步骤

在 ibus-libpinyin-voice 项目目录下执行：

### 1. 拉取上游最新代码

```bash
git fetch upstream
```

### 2. 合并上游到 main

```bash
git checkout main
git merge upstream/main
```

如果有冲突，解决冲突后：
```bash
# 解决冲突后
git add .
git commit
```

### 3. 将更新后的 main 合并到功能分支

```bash
git checkout feat/punctuation-model
git merge main
```

同样，有冲突就解决后 commit。

### 4. 推送到你的 fork

```bash
git push origin main feat/punctuation-model
```

### 5. 重新编译安装

```bash
sudo bash install.sh
```

安装完成后如果语音输入不好用，重新登录或手动重启 ibus：

```bash
ibus exit && sleep 2 && ibus-daemon -drx &
```

## 注意事项

- 合并时重点关注 `src/PYEngine.cc`、`src/PYPPinyinEngine.cc`、`src/PinyinEngine.h` 这些文件的冲突，因为上游可能修改了按键处理或引擎逻辑，而你的分支在这些文件里加了语音功能代码
- `src/PYVoiceInput.cc` 和 `src/PYVoiceInput.h` 是你独有的文件，上游不会改动，一般不会有冲突
- 每次合并后建议编译测试一下，确保语音功能正常
