# clone_narration_video

这是独立的影视解说克隆子项目。当前已实现八个模块：

1. `1_reference_analyzer` 参考视频解析
2. `2_narration_segmenter` 解说段落切分
3. `3_movie_shot_parser` 原电影镜头拆分
4. `4_visual_alignment_engine` 画面定位
5. `5_script_visual_binder` 解说画面绑定
6. `6_rewrite_engine` 仿稿
7. `7_timeline_composer` 生成新视频脚本时间轴
8. `8_generate_video` 生成剪映草稿 / 直接生成视频

## 环境

```powershell
cd .\clone_narration_video
.\setup_venv.ps1 -TorchWheel cu124
.\.venv\Scripts\Activate.ps1
```

如本机 CUDA/PyTorch 版本不匹配，可改用 `-TorchWheel cu121` 或 `-TorchWheel cpu`。
镜头分割会优先使用 `model\transnetv2-weights\transnetv2-pytorch-weights.pth`，`torch.cuda.is_available()` 为 true 时自动走 GPU；ffmpeg 解码会在检测到 CUDA hwaccel 时优先尝试 GPU。

第 6 步默认使用 OpenAI 兼容接口。可通过环境变量配置：

```powershell
$env:CLONE_AI_API_KEY = "你的 key"
$env:CLONE_AI_BASE_URL = "https://api.openai.com/v1"
$env:CLONE_AI_MODEL = "gpt-4o-mini"
```

第 6 步没有本地兜底策略；缺少 API Key、AI 调用失败或返回结构不完整时会直接报错。

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

### 6. 仿稿

使用 AI：

```powershell
python .\6_rewrite_engine\run.py `
  --script-mapping .\outputs\5_script_visual_binder\script_mapping.json `
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
  --script-mapping .\outputs\5_script_visual_binder\script_mapping.json `
  --movie-shots .\outputs\3_movie_shot_parser\movie_shots.json `
  --movie-source .\data\movie.mp4 `
  --output-dir .\outputs\7_timeline_composer
```

输出：

```text
outputs\7_timeline_composer\final_timeline.json
```

## 七模块流水线 / 可选第 8 步渲染

```powershell
python .\main.py `
  --ref-video-path .\data\ref.mp4 `
  --movie-path .\data\movie.mp4 `
  --subtitle-srt .\data\ref.srt `
  --ai-provider custom_openai
```

最终输出位于 `outputs\7_timeline_composer\final_timeline.json`。

## 第 8 步：生成剪映草稿 / 直接生成视频

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
  --ai-provider custom_openai `
  --render-mode both `
  --edge-voice-id zh-CN-XiaoxiaoNeural `
  --video-encoder auto
```
