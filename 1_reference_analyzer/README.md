# 参考视频解析模块

输入参考解说视频，输出 `ref_analysis.json`、`ref_subtitle.srt` 和关键帧。

```powershell
python .\clone_narration_video\1_reference_analyzer\run.py `
  --ref-video-path .\data\ref.mp4 `
  --subtitle-srt .\data\ref.srt `
  --output-dir .\clone_narration_video\outputs\1_reference_analyzer
```

不传 `--subtitle-srt` 时默认用 Bcut ASR；只测试镜头拆分可加 `--asr-provider none`。

