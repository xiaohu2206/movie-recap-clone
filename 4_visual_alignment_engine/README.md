# 画面定位模块

第 4 步由 `run.py` 衔接主流水线，并直接调用 `run_shot_match_localize.py` 完成镜头定位。定位器结合多帧全局描述子、ORB 候选召回和几何验证，`run.py` 再把结果映射回第 1、3 步 JSON 中的镜头 ID 与时间，写出下游兼容的 `ref_to_movie_timeline.json`。

```powershell
python .\4_visual_alignment_engine\run.py `
  --ref-analysis .\outputs\1_reference_analyzer\ref_analysis.json `
  --movie-shots .\outputs\3_movie_shot_parser\movie_shots.json `
  --output-dir .\outputs\4_visual_alignment_engine
```

默认 clip 目录会自动从 JSON 路径推导：

```text
outputs\1_reference_analyzer\shot_clips
outputs\3_movie_shot_parser\shot_clips
```

常用参数：

```powershell
--sample-count 6
--frame-size 384
--workers 4
--neighbor-radius 2
--candidate-count 30
--geometry-candidate-count 24
--top-k 3
--min-score 0.35
--min-geometry-inliers 20
--manual-overrides .\overrides.json
--diagnostics-dir .\outputs\4_visual_alignment_engine\diagnostics
```

输出：

```text
outputs\4_visual_alignment_engine\ref_to_movie_timeline.json
outputs\4_visual_alignment_engine\matches.csv
outputs\4_visual_alignment_engine\matches.json
outputs\4_visual_alignment_engine\pairs\
outputs\4_visual_alignment_engine\low_confidence_pairs\
outputs\4_visual_alignment_engine\diagnostics\low_confidence_report.json
```

`run.py` 是生产入口，`run_shot_match_localize.py` 是实际定位算法模块。
