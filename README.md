# clone_narration_video

这是独立的影视解说克隆子项目。当前已实现八个主模块，并支持可选的 5A 原声片段识别：

1. `1_reference_analyzer` 参考视频解析
2. `2_narration_segmenter` 解说段落切分
3. `3_movie_shot_parser` 原电影镜头拆分
4. `4_visual_alignment_engine` 画面定位
5. `5_script_visual_binder` 解说画面绑定
6. `5_audio_role_classifier` 可选：识别哪些片段应保留原片原声
7. `6_rewrite_engine` 仿稿
8. `7_timeline_composer` 生成新视频脚本时间轴
9. `8_generate_video` 生成剪映草稿 / 直接生成视频

## 环境

```powershell
cd .\movie-recap-clone
.\setup_venv.ps1 -TorchWheel cu128
.\.venv\Scripts\Activate.ps1
```

如本机 CUDA/PyTorch 版本不匹配，可改用 `-TorchWheel cu124`、`-TorchWheel cu121` 或 `-TorchWheel cpu`。
**RTX 50 系列（Blackwell，sm_120）必须使用 `cu128`**，否则 PyTorch 会报 GPU 不兼容警告且无法使用 CUDA。
镜头分割会优先使用 `model\transnetv2-weights\transnetv2-pytorch-weights.pth`，GPU 可用时自动走 GPU；ffmpeg 解码会在检测到 CUDA hwaccel 时优先尝试 GPU。

第 6 步默认使用 OpenAI 兼容接口。可通过环境变量配置：

```powershell
$env:CLONE_AI_API_KEY = "你的 key"
$env:CLONE_AI_BASE_URL = "https://api.openai.com/v1"
$env:CLONE_AI_MODEL = "gpt-4o-mini"
```

第 6 步没有本地兜底策略；缺少 API Key、AI 调用失败或返回结构不完整时会直接报错。

## 前后端开发运行

后端是 Python 流水线，前端是 `frontend` 目录下的 Electron + React + Vite 桌面应用。开发时建议先准备后端虚拟环境，再启动前端桌面壳。

### 1. 准备后端环境

```powershell
cd .\movie-recap-clone
.\setup_venv.ps1 -TorchWheel cu128
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如本机 CUDA/PyTorch 版本不匹配，可把 `cu128` 换成 `cu124`、`cu121` 或 `cpu`。RTX 50 系列请使用 `cu128`。

### 2. 启动前端开发桌面应用

```powershell
cd .\frontend
npm install
$env:ELECTRON_RUN_AS_NODE = $null
npm run dev
```

`npm run dev` 会同时启动 Vite 前端服务和 Electron 桌面窗口。桌面窗口里点击“开始生成”时，会调用项目根目录的 `main.py`，并把日志回传到前端界面。

如果只想在浏览器里预览页面：

```powershell
cd .\frontend
npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

浏览器预览模式不会真正调用后端，只用于检查界面。

### 3. 单独运行后端完整流水线

```powershell
cd .\clone_narration_video
.\.venv\Scripts\Activate.ps1

python .\main.py `
  --ref-video-path .\data\ref.mp4 `
  --movie-path .\data\movie.mp4 `
  --subtitle-srt .\data\ref.srt `
  --enable-audio-role-classifier `
  --movie-subtitle-srt .\data\movie.srt `
  --ai-provider custom_openai `
  --render-mode both
```

### 4. 打包 Windows exe

```powershell
cd .\frontend
$env:ELECTRON_RUN_AS_NODE = $null
npm run dist
```

生成安装包：

```text
frontend\release\Clone Narration Studio-Setup-1.0.0.exe
```

生成免安装目录版：

```powershell
cd .\frontend
$env:ELECTRON_RUN_AS_NODE = $null
npm run pack:dir
```

输出位置：

```text
frontend\release\win-unpacked\Clone Narration Studio.exe
```

注意：当前打包会把后端脚本资源放入安装包，但目标机器仍需要可用的 Python 环境和依赖。若要做完整离线交付，需要后续把 Python runtime 或 `.venv` 一起纳入安装流程。

## 单步运行

下面命令默认已经在 `clone_narration_video` 目录下，并已激活 `.venv`。

也可以通过总入口只执行某一步，前置产物需要已经存在：

```powershell
python .\main.py --only-step 5 --output-root .\outputs
python .\main.py --only-step audio --output-root .\outputs --movie-subtitle-srt .\data\movie.srt
python .\main.py --only-step 6 --output-root .\outputs --enable-audio-role-classifier --ai-provider custom_openai
```

如果已经启用过 5A，后续单独重跑第 6、7、8 步时也要继续带上增强后的 mapping。否则第 6 步会读取普通 `script_mapping.json`，第 7 步只能得到默认的 `audio_mode=voiceover`。

`--only-step` 支持 `1` 到 `8`，以及原声片段识别的 `audio` / `5a`。

### 1. 参考视频解析

有现成字幕时推荐这样跑，速度最快：

```powershell
python .\1_reference_analyzer\run.py `
  --ref-video-path .\data\ref.mp4 `
  --subtitle-srt .\data\ref.srt `
  --output-dir .\outputs\1_reference_analyzer `
  --backend auto
```

没有字幕、需要自动识别时：

```powershell
python .\1_reference_analyzer\run.py `
  --ref-video-path .\data\ref.mp4 `
  --output-dir .\outputs\1_reference_analyzer `
  --asr-provider bcut `
  --backend auto
```

无 GPU 机器上运行同一步时，先禁用 CUDA 探测：

```powershell
$env:TRANSNETV2_DEVICE = "cpu"
$env:CLONE_FFMPEG_HWACCEL = "cpu"

python .\1_reference_analyzer\run.py `
  --ref-video-path .\data\ref.mp4 `
  --subtitle-srt .\data\ref.srt `
  --output-dir .\outputs\1_reference_analyzer `
  --backend auto
```

如果没有现成字幕，再把 `--subtitle-srt .\data\ref.srt` 换成 `--asr-provider bcut`。

输出：

```text
outputs\1_reference_analyzer\ref_analysis.json
outputs\1_reference_analyzer\ref_subtitle.srt
outputs\1_reference_analyzer\keyframes\
```

### 2. 解说段落切分

```powershell
python .\2_narration_segmenter\run.py `
  --input .\outputs\1_reference_analyzer\ref_analysis.json `
  --output-dir .\outputs\2_narration_segmenter
```

输出：

```text
outputs\2_narration_segmenter\narration_segments.json
```

### 3. 原电影镜头拆分

```powershell
python .\3_movie_shot_parser\run.py `
  --movie-path .\data\movie.mp4 `
  --output-dir .\outputs\3_movie_shot_parser `
  --backend auto
```

输出：

```text
outputs\3_movie_shot_parser\movie_shots.json
outputs\3_movie_shot_parser\keyframes\
```

### 4. 画面定位

```powershell
python .\4_visual_alignment_engine\run.py `
  --ref-analysis .\outputs\1_reference_analyzer\ref_analysis.json `
  --movie-shots .\outputs\3_movie_shot_parser\movie_shots.json `
  --output-dir .\outputs\4_visual_alignment_engine
```

输出：

```text
outputs\4_visual_alignment_engine\ref_to_movie_timeline.json
```

### 5. 解说画面绑定

```powershell
python .\5_script_visual_binder\run.py `
  --narration-segments .\outputs\2_narration_segmenter\narration_segments.json `
  --timeline .\outputs\4_visual_alignment_engine\ref_to_movie_timeline.json `
  --output-dir .\outputs\5_script_visual_binder
```

输出：

```text
outputs\5_script_visual_binder\script_mapping.json
```

### 5A. 原声片段识别

该步骤会在 `script_mapping.json` 基础上补充原声决策字段，包括：

```text
movie_time_ranges[*].audio_role
movie_time_ranges[*].audio_action
movie_time_ranges[*].audio_confidence
movie_time_ranges[*].audio_reason
text_units[*].role / action / related_range_ids
```

没有原电影字幕时，可传 `--movie-path` 自动识别。该模式只会裁剪并识别 `script_mapping.json` 中已匹配到的电影镜头时间段，不会识别整部电影：

```powershell
python .\5_audio_role_classifier\run.py `
  --script-mapping .\outputs\5_script_visual_binder\script_mapping.json `
  --ref-analysis .\outputs\1_reference_analyzer\ref_analysis.json `
  --movie-path .\data\movie.mp4 `
  --output-dir .\outputs\5_audio_role_classifier
```

自动识别依赖 Bcut 网络服务，模块会默认重试 3 次；如果网络不稳定，直接传 `--movie-subtitle-srt` 更稳。可用 `--movie-asr-padding` 调整匹配镜头前后的额外识别秒数，用 `--movie-asr-merge-gap` 合并相邻小片段。

输出：

```text
outputs\5_audio_role_classifier\script_mapping_with_audio.json
```

启用 5A 后，后续第 6、7 步都应继续使用这个增强后的 `script_mapping_with_audio.json`，否则原声标记会在后续产物中丢失。

### 6. 仿稿

使用 AI：

```powershell
python .\6_rewrite_engine\run.py `
  --script-mapping .\outputs\5_audio_role_classifier\script_mapping_with_audio.json `
  --output-dir .\outputs\6_rewrite_engine `
  --provider custom_openai
```

如果没有启用 5A，才使用普通 mapping：

```text
.\outputs\5_script_visual_binder\script_mapping.json
```

使用增强后的 mapping 时，仿稿会读取 `text_units`：只处理 `role=narration` 的单元，`original_dialogue` 单元会在 `rewritten_units` 中保留为空文案并标记 `keep_original_audio=true`。

输出：

```text
outputs\6_rewrite_engine\rewritten_script.json
```

### 7. 生成新视频脚本时间轴

```powershell
python .\7_timeline_composer\run.py `
  --rewritten-script .\outputs\6_rewrite_engine\rewritten_script.json `
  --script-mapping .\outputs\5_audio_role_classifier\script_mapping_with_audio.json `
  --movie-shots .\outputs\3_movie_shot_parser\movie_shots.json `
  --movie-source .\data\movie.mp4 `
  --output-dir .\outputs\7_timeline_composer
```

如果没有启用 5A，才使用普通 mapping：

```text
.\outputs\5_script_visual_binder\script_mapping.json
```

当 `rewritten_script.json` 包含 `rewritten_units` 时，第 7 步会把 mixed segment 拆成多个 timeline item，并写入 `audio_mode`、`OST`、`audio_decision`。其中 `audio_mode=original` 的 item 时长直接取原片 clip 时长，不按字数估算 TTS。

排查提示：如果 `final_timeline.json` 里全部都是 `"audio_mode": "voiceover"`，优先检查第 6 步是否使用了 `script_mapping_with_audio.json`。第 5A 输出中应能看到 `audio_action=play_original_audio`，第 6 步输出中应保留 `rewritten_units` 或带音频字段的 `movie_time_ranges`。

输出：

```text
outputs\7_timeline_composer\final_timeline.json
```

## 完整流水线 / 可选第 8 步渲染

```powershell
python .\main.py `
  --ref-video-path .\data\ref.mp4 `
  --movie-path .\data\movie.mp4 `
  --subtitle-srt .\data\ref.srt `
  --enable-audio-role-classifier `
  --movie-subtitle-srt .\data\movie.srt `
  --ai-provider custom_openai
```

最终输出位于 `outputs\7_timeline_composer\final_timeline.json`。

## 第 8 步：生成剪映草稿 / 直接生成视频

第 8 步会读取 `final_timeline.json` 中的 `audio_mode`：

```text
audio_mode=voiceover => 裁视频时去掉原片声音，合成 TTS 后 mux 到视频
audio_mode=original  => 不生成 TTS，不生成静音音频，裁视频时保留原片音轨
```

生成剪映草稿时，`voiceover` 片段的视频轨音量为 0，并额外添加 TTS 音频轨；`original` 片段的视频轨音量为 1，不额外添加配音轨。

同时生成剪映草稿和 mp4：
```powershell
python .\8_generate_video\run.py `
  --timeline .\outputs\7_timeline_composer\final_timeline.json `
  --output-dir .\outputs\8_generate_video `
  --mode both `
  --voice-id zh-CN-XiaoxiaoNeural
```

只生成剪映草稿：
```powershell
python .\8_generate_video\run.py `
  --timeline .\outputs\7_timeline_composer\final_timeline.json `
  --output-dir .\outputs\8_generate_video `
  --mode draft `
  --voice-id zh-CN-XiaoxiaoNeural
```

只直接生成视频：
```powershell
python .\8_generate_video\run.py `
  --timeline .\outputs\7_timeline_composer\final_timeline.json `
  --output-dir .\outputs\8_generate_video `
  --mode video `
  --voice-id zh-CN-XiaoxiaoNeural `
  --video-output-name clone_narration_output.mp4 `
  --video-encoder auto
```

输出：
```text
outputs\8_generate_video\audio\
outputs\8_generate_video\clone_narration_output.mp4
outputs\8_generate_video\generate_video_result.json
```

剪映草稿默认写入本机剪映/CapCut 草稿目录；也可以通过 `--jianying-draft-dir` 指定草稿根目录。

`generate_video_result.json` 中的 `audio_results` 会标记原声片段：

```json
{
  "item_003": {
    "path": "",
    "duration": 3.215,
    "original_audio": true,
    "silent": false
  }
}
```

完整流水线最后也可以追加渲染：
```powershell
python .\main.py `
  --ref-video-path .\data\ref.mp4 `
  --movie-path .\data\movie.mp4 `
  --subtitle-srt .\data\ref.srt `
  --enable-audio-role-classifier `
  --movie-subtitle-srt .\data\movie.srt `
  --ai-provider custom_openai `
  --render-mode both `
  --edge-voice-id zh-CN-XiaoxiaoNeural `
  --video-encoder auto
```
