# 画面定位模块

```powershell
python .\clone_narration_video\4_visual_alignment_engine\run.py `
  --ref-analysis .\clone_narration_video\outputs\1_reference_analyzer\ref_analysis.json `
  --movie-shots .\clone_narration_video\outputs\3_movie_shot_parser\movie_shots.json
```

输出 `ref_to_movie_timeline.json`，包含每个参考镜头的原电影候选与最终匹配。

