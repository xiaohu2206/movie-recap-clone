# clone_narration_video

这是独立的影视解说克隆子项目。当前已实现十个流水线步骤：

1. `1_reference_analyzer` 参考视频解析
2. `2_narration_segmenter` 解说段落切分
3. `3_movie_shot_parser` 原电影镜头拆分
4. `4_visual_alignment_engine` 画面定位
5. `5_script_visual_binder` 解说画面绑定
5.1. `5.1_movie_subtitle_filler` 原片字幕补充
5.2. `5.2_audio_role_classifier` 原声判定
6. `6_rewrite_engine` 仿稿
7. `7_timeline_composer` 生成新视频脚本时间轴
8. `8_generate_video` 生成剪映草稿 / 直接生成视频

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

### 5.1 原片字幕补充

有现成原片字幕时推荐这样跑，速度最快：

```powershell
python .\5.1_movie_subtitle_filler\run.py `
  --script-mapping .\outputs\5_script_visual_binder\script_mapping.json `
  --movie-path .\data\movie.mp4 `
  --movie-subtitle-srt .\data\movie.srt `
  --output-dir .\outputs\5.1_movie_subtitle_filler
```

没有原片字幕、需要自动识别时：

```powershell
python .\5.1_movie_subtitle_filler\run.py `
  --script-mapping .\outputs\5_script_visual_binder\script_mapping.json `
  --movie-path .\data\movie.mp4 `
  --output-dir .\outputs\5.1_movie_subtitle_filler
```

输出：

```text
outputs\5.1_movie_subtitle_filler\script_mapping_subtitled.json
outputs\5.1_movie_subtitle_filler\movie_subtitle.srt
```

### 5.2 原声判定

```powershell
python .\5.2_audio_role_classifier\run.py `
  --script-mapping .\outputs\5.1_movie_subtitle_filler\script_mapping_subtitled.json `
  --output-dir .\outputs\5.2_audio_role_classifier `
  --provider custom_openai
```

输出：

```text
outputs\5.2_audio_role_classifier\script_mapping_with_audio.json
```

### 6. 仿稿

使用 AI：

```powershell
python .\6_rewrite_engine\run.py `
  --script-mapping .\outputs\5.2_audio_role_classifier\script_mapping_with_audio.json `
  --output-dir .\outputs\6_rewrite_engine `
  --provider custom_openai
```

输出：

```text
outputs\6_rewrite_engine\rewritten_script.json
```

### 7. 生成新视频脚本时间轴

```powershell
python .\7_timeline_composer\run.py `
  --rewritten-script .\outputs\6_rewrite_engine\rewritten_script.json `
  --script-mapping .\outputs\5.2_audio_role_classifier\script_mapping_with_audio.json `
  --movie-shots .\outputs\3_movie_shot_parser\movie_shots.json `
  --movie-source .\data\movie.mp4 `
  --output-dir .\outputs\7_timeline_composer `
  --output-root .\outputs
```

输出：

```text
outputs\7_timeline_composer\final_timeline.json
outputs\7_timeline_composer\shot_breakdown.json
```

## 完整流水线 / 可选第 8 步渲染

```powershell
python .\main.py `
  --ref-video-path .\data\ref.mp4 `
  --movie-path .\data\movie.mp4 `
  --subtitle-srt .\data\ref.srt `
  --movie-subtitle-srt .\data\movie.srt `
  --ai-provider custom_openai
```

最终输出位于 `outputs\7_timeline_composer\final_timeline.json`。

## 第 8 步：生成剪映草稿 / 直接生成视频

同时生成剪映草稿和 mp4：
```powershell
python .\8_generate_video\run.py `
  --timeline .\outputs\7_timeline_composer\final_timeline.json `
  --output-dir .\outputs\8_generate_video `
  --script-mapping .\outputs\5.2_audio_role_classifier\script_mapping_with_audio.json `
  --output-root .\outputs `
  --mode both `
  --voice-id zh-CN-XiaoxiaoNeural
```

只生成剪映草稿：
```powershell
python .\8_generate_video\run.py `
  --timeline .\outputs\7_timeline_composer\final_timeline.json `
  --output-dir .\outputs\8_generate_video `
  --script-mapping .\outputs\5.2_audio_role_classifier\script_mapping_with_audio.json `
  --output-root .\outputs `
  --mode draft `
  --voice-id zh-CN-XiaoxiaoNeural
```

只直接生成视频：
```powershell
python .\8_generate_video\run.py `
  --timeline .\outputs\7_timeline_composer\final_timeline.json `
  --output-dir .\outputs\8_generate_video `
  --script-mapping .\outputs\5.2_audio_role_classifier\script_mapping_with_audio.json `
  --output-root .\outputs `
  --mode video `
  --voice-id zh-CN-XiaoxiaoNeural `
  --video-output-name clone_narration_output.mp4 `
  --video-encoder auto
```

输出：
```text
outputs\8_generate_video\audio\
outputs\8_generate_video\jianying_drafts\
outputs\8_generate_video\clone_narration_output.mp4
outputs\8_generate_video\generate_video_result.json
```

完整流水线最后也可以追加渲染：
```powershell
python .\main.py `
  --ref-video-path .\data\ref.mp4 `
  --movie-path .\data\movie.mp4 `
  --subtitle-srt .\data\ref.srt `
  --movie-subtitle-srt .\data\movie.srt `
  --ai-provider custom_openai `
  --render-mode both `
  --edge-voice-id zh-CN-XiaoxiaoNeural `
  --video-encoder auto
```
