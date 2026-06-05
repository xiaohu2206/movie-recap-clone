# 画面定位模块

```powershell
python .\clone_narration_video\4_visual_alignment_engine\run.py `
  --ref-analysis .\clone_narration_video\outputs\1_reference_analyzer\ref_analysis.json `
  --movie-shots .\clone_narration_video\outputs\3_movie_shot_parser\movie_shots.json
```

输出 `ref_to_movie_timeline.json`，包含每个参考镜头的原电影候选与最终匹配。

新版默认使用多关键帧镜头级匹配、top-k 候选召回和全局路径优化。常用参数：

```powershell
--keyframes-per-shot 3
--recall-top-k 80
--rerank-top-k 20
--top-n 8
--min-score 0.35
--feature-mode classic
--disable-global-path
--manual-overrides .\overrides.json
--diagnostics-dir .\outputs\4_visual_alignment_engine\diagnostics
```

`classic_clip` 已作为 CLI 模式保留；未配置深度特征模型时会自动回退到 `classic`，并在 metadata 中标记 `embedding_status`。
