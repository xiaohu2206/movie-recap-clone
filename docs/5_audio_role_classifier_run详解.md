# `5_audio_role_classifier/run.py` 详解

本文档解释 `5_audio_role_classifier/run.py` 当前具体做什么、依赖哪些输入、生成哪些字段，以及它在整条影视解说克隆流水线中的作用。

## 1. 模块定位

`5_audio_role_classifier/run.py` 是项目中的可选 5A 阶段，中文可以理解为：

```txt
原声片段识别 / 音频角色分类模块
```

它运行在：

```txt
5_script_visual_binder
  ↓
5_audio_role_classifier
  ↓
6_rewrite_engine
  ↓
7_timeline_composer
  ↓
8_generate_video
```

它的核心目标是：

```txt
在已经完成“参考视频字幕段落 -> 原电影画面片段”绑定之后，
判断每个片段应该重新生成解说配音，还是应该保留原电影对白原声。
```

它不负责：

1. 不重新做画面匹配。
2. 不修改 `movie_time_ranges` 的起止时间。
3. 不生成新文案。
4. 不合成 TTS。
5. 不剪辑或渲染视频。

它只在第 5 步的 `script_mapping.json` 基础上追加音频决策字段，输出增强版 `script_mapping_with_audio.json`。

## 2. 为什么需要这个模块

影视解说视频里通常混合了两类字幕：

```txt
解说旁白：
  用第三方视角概括剧情、制造悬念、串联画面。

原片对白：
  直接来自电影角色的台词，最好保留原电影声音。
```

第 5 步 `5_script_visual_binder` 只能告诉后续模块“这段解说对应哪些原电影画面”，但它不知道这些画面里的声音应该怎么处理。

如果没有本模块，后续默认会把所有片段都当成解说覆盖：

```txt
裁原电影画面
  → 去掉原音轨
  → 用新文案生成 TTS
  → 把 TTS mux 到视频里
```

这样会导致一个问题：原本应该听到角色对白的地方，也被新解说盖掉。

本模块解决的就是这个问题：在进入仿稿和时间轴之前，提前标出哪些单元是解说，哪些单元是原片对白。

## 3. 输入文件

### 3.1 必需输入

#### `--script-mapping`

来源：

```txt
outputs/5_script_visual_binder/script_mapping.json
```

用途：

1. 读取每个 `segment_id`。
2. 读取每段的 `old_text`。
3. 读取 `ref_time_range`，用于从参考视频字幕中切出字幕单元。
4. 读取 `movie_time_ranges`，用于从原电影字幕中切出对应片段的字幕。

关键结构示例：

```json
{
  "script_mapping": [
    {
      "segment_id": "seg_001",
      "old_text": "职场如战场而他是那个最没用的炮灰...",
      "ref_time_range": {
        "start": 0.0,
        "end": 17.69
      },
      "movie_time_ranges": [
        {
          "start": 149.643,
          "end": 153.01,
          "source_ref_shot_id": "ref_shot_001",
          "movie_shot_ids": ["movie_shot_000038"]
        }
      ]
    }
  ]
}
```

#### `--ref-analysis`

来源：

```txt
outputs/1_reference_analyzer/ref_analysis.json
```

用途：

1. 读取 `subtitle_srt`，也就是参考解说视频对应的 SRT 文件。
2. 读取 `ref_shots`，用于把参考字幕单元和画面 range 建立关联。

关键字段：

```json
{
  "subtitle_srt": "outputs/1_reference_analyzer/ref_subtitle.srt",
  "ref_shots": [
    {
      "ref_shot_id": "ref_shot_001",
      "start": 0.0,
      "end": 3.2
    }
  ]
}
```

### 3.2 原电影字幕输入二选一

#### 方式一：传入已有原电影字幕

```powershell
--movie-subtitle-srt .\data\movie.srt
```

这种方式最稳定。传入后模块会直接解析该 SRT，不再调用 ASR。

#### 方式二：传入原电影视频，让模块自动识别字幕

```powershell
--movie-path .\data\movie.mp4
```

如果没有 `--movie-subtitle-srt`，模块会使用原电影视频自动生成：

```txt
outputs/5_audio_role_classifier/movie_subtitle.srt
```

当前实现不是对整部电影做 ASR，而是只对 `script_mapping.json` 中已经匹配到的 `movie_time_ranges` 做局部 ASR。

这样做的好处：

1. ASR 范围小，速度更快。
2. 只识别后续会用到的画面片段。
3. 减少长电影 ASR 失败或超时的概率。

## 4. 输出文件

主输出：

```txt
outputs/5_audio_role_classifier/script_mapping_with_audio.json
```

它保持原来的 `script_mapping` 主结构不变，只在每个 segment 和 range 上增加音频决策字段。

整体结构：

```json
{
  "script_mapping": [],
  "audio_role_backend": {
    "movie_subtitle_srt": "outputs/5_audio_role_classifier/movie_subtitle.srt",
    "decision_mode": "rules_only",
    "version": "audio_role_classifier_v1",
    "original_threshold": 0.82,
    "review_threshold": 0.65,
    "asr_retries": 3,
    "movie_asr_scope": "matched_movie_time_ranges",
    "movie_asr_padding": 0.2,
    "movie_asr_merge_gap": 0.5
  }
}
```

### 4.1 `movie_time_ranges` 增强字段

每个原电影时间段会新增：

```json
{
  "range_id": "seg_001_range_001",
  "movie_subtitles": [
    {
      "start": 149.443,
      "end": 152.043,
      "text": "试用期三个月的时候是刚及格"
    }
  ],
  "audio_role": "narration_overlay",
  "audio_action": "rewrite_and_voiceover",
  "audio_confidence": 0.917,
  "audio_reason": "参考字幕和原电影字幕相似度不足，按解说覆盖处理",
  "visual_match_locked": true
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `range_id` | 给原本没有 ID 的 movie range 补一个稳定 ID |
| `movie_subtitles` | 该电影时间段内匹配到的原电影字幕 |
| `audio_role` | 该片段的音频角色 |
| `audio_action` | 后续模块应该采取的音频动作 |
| `audio_confidence` | 规则判断置信度 |
| `audio_reason` | 可读的判定理由 |
| `visual_match_locked` | 表示音频判断不应反向修改画面匹配 |

### 4.2 `text_units` 增强字段

每个 segment 会新增 `text_units`。它不是从 `old_text` 硬切出来的，而是从参考视频 SRT 中按 `ref_time_range` 截取出来的字幕行。

示例：

```json
{
  "unit_id": "seg_001_unit_001",
  "source_ref_subtitle_index": 1,
  "ref_start": 0.0,
  "ref_end": 2.0,
  "text": "职场如战场而他是那个最没用的炮灰",
  "role": "narration",
  "action": "rewrite",
  "related_range_ids": ["seg_001_range_001"],
  "matched_movie_subtitles": [],
  "confidence": 0.9,
  "reason": "第三方叙述或未在原电影字幕中找到相似对白"
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `unit_id` | 字幕单元 ID |
| `source_ref_subtitle_index` | 来源参考视频 SRT 的字幕序号 |
| `ref_start` / `ref_end` | 参考视频中的字幕时间 |
| `text` | 字幕文本 |
| `role` | 字幕单元角色，当前主要是 `narration` / `original_dialogue` / `unknown` |
| `action` | 后续动作，当前主要是 `rewrite` / `play_original_audio` / `manual_review` |
| `related_range_ids` | 该字幕单元关联到哪些原电影 range |
| `matched_movie_subtitles` | 若判断为原片对白，记录匹配到的原电影字幕 |
| `confidence` | 字幕单元判断置信度 |
| `reason` | 判断原因 |

### 4.3 `segment_audio_role`

每个 segment 会根据内部 range 的动作聚合出一个整体角色：

| 条件 | `segment_audio_role` |
| --- | --- |
| 全部 `rewrite_and_voiceover` | `narration_overlay` |
| 全部 `play_original_audio` | `original_dialogue` |
| 同时存在解说和原声 | `mixed` |
| 存在 `manual_review` | `manual_review` |
| 无法判断 | `unknown` |

## 5. 核心执行流程

入口函数是：

```python
classify_audio_roles(...)
```

CLI 入口 `main()` 只是负责解析参数、读取 JSON、调用 `classify_audio_roles()`，最后写出 `script_mapping_with_audio.json`。

整体流程如下：

```txt
1. 读取 ref_analysis.subtitle_srt。
2. parse_srt() 解析参考视频字幕。
3. 读取 script_mapping.script_mapping。
4. 准备原电影字幕：
   - 如果传了 --movie-subtitle-srt，直接使用；
   - 否则根据 movie_time_ranges 局部抽音频并调用 Bcut ASR。
5. parse_srt() 解析原电影字幕。
6. 为每个 segment 生成 text_units。
7. 为每个 movie_time_range 补 movie_subtitles。
8. 对每个 movie_time_range 做音频角色判断。
9. 对每个 text_unit 做音频角色判断。
10. 聚合 segment_audio_role。
11. 返回增强后的 script_mapping。
```

## 6. 原电影字幕生成逻辑

如果没有传 `--movie-subtitle-srt`，模块会自动生成原电影字幕。

实现入口：

```python
_extract_matched_movie_srt_with_retries(...)
```

它的流程：

```txt
1. 从 script_mapping[*].movie_time_ranges 收集所有原电影时间段。
2. 给每个时间段前后增加 padding，默认 0.2 秒。
3. 如果相邻时间段间隔小于 merge_gap，默认 0.5 秒，则合并识别。
4. 用 ffmpeg 从原电影中抽出这些局部音频片段。
5. 对每个音频片段调用 Bcut ASR。
6. 把识别结果按原电影时间偏移还原为全局时间。
7. 写成 movie_subtitle.srt。
```

相关函数：

| 函数 | 作用 |
| --- | --- |
| `_movie_ranges_from_mapping()` | 从 `script_mapping` 收集所有电影时间段 |
| `_merge_time_ranges()` | 对时间段做 padding 和合并 |
| `_extract_audio_clip_mp3()` | 用 ffmpeg 抽取局部 mp3 |
| `_bcut_entries_for_audio()` | 调用 Bcut ASR 并转成字幕条目 |
| `_run_with_retries()` | 为 ASR 增加失败重试 |
| `_extract_matched_movie_srt_with_retries()` | 组织局部 ASR 并写出 SRT |

注意：

1. 抽取音频时会输出到 `outputs/5_audio_role_classifier/movie_subtitle_clips/`。
2. 如果局部 mp3 已存在且大小大于 0，会复用它，不重复抽取。
3. Bcut ASR 使用 `use_cache=True`，底层会复用缓存。
4. 如果 ASR 多次失败，错误提示会建议用户改为手动传入 `--movie-subtitle-srt`。

## 7. 音频角色判定规则

当前版本的决策模式是：

```txt
rules_only
```

也就是说，它没有调用 LLM，而是用字幕文本相似度做规则判断。

### 7.1 文本归一化

函数：

```python
_normalize_text(text)
```

处理方式：

```txt
1. 使用 unicodedata.normalize("NFKC", text) 统一全角半角等字符形态。
2. 转小写。
3. 只保留 isalnum() 字符。
4. 去掉标点、空格、引号等非字母数字字符。
```

示例：

```txt
“我不努力？我只是没能力！”
  ↓
我不努力我只是没能力
```

### 7.2 相似度计算

函数：

```python
_similarity(a, b)
```

它使用 Python 标准库：

```python
SequenceMatcher(None, left, right).ratio()
```

相似度范围：

```txt
0.0 完全不像
1.0 完全一致
```

### 7.3 range 级别判断

函数：

```python
_decide_range(...)
```

输入：

1. 当前 range 关联到的参考字幕单元 `related_units`。
2. 当前电影时间段内的原电影字幕 `movie_subtitles`。
3. 阈值参数：
   - `original_threshold`，默认 `0.82`。
   - `review_threshold`，默认 `0.65`。
   - `min_dialogue_chars`，默认 `3`。

判断逻辑：

| 条件 | `audio_role` | `audio_action` | 说明 |
| --- | --- | --- | --- |
| 没有原电影字幕 | `narration_overlay` | `rewrite_and_voiceover` | 默认按解说覆盖 |
| 没有可比较的参考字幕 | `narration_overlay` | `rewrite_and_voiceover` | 无法证明是原声 |
| 原电影字幕长度足够，且相似度 >= `original_threshold` | `original_dialogue` | `play_original_audio` | 保留原片对白 |
| 相似度 >= `review_threshold` | `unknown` | `manual_review` | 部分相似，需要人工复查 |
| 其他情况 | `narration_overlay` | `rewrite_and_voiceover` | 按解说覆盖 |

当前默认策略偏保守：只有文本相似度足够高才会判为原片对白。

### 7.4 text unit 级别判断

函数：

```python
_unit_decision(...)
```

它根据该字幕单元关联到的 range 决定自身角色：

| 关联 range 情况 | `role` | `action` |
| --- | --- | --- |
| 存在 `play_original_audio` | `original_dialogue` | `play_original_audio` |
| 存在 `manual_review` | `unknown` | `manual_review` |
| 其他情况 | `narration` | `rewrite` |

如果判断为 `original_dialogue`，还会在 `matched_movie_subtitles` 里记录和该字幕单元相似度超过 `0.65` 的原电影字幕。

## 8. 字幕单元和画面 range 如何关联

关键函数：

```python
_range_ref_overlap(unit, row, shots_by_id)
```

它不是直接用电影时间比较，而是通过 `source_ref_shot_id` 回到参考视频时间轴：

```txt
movie_time_range.source_ref_shot_id
  → ref_analysis.ref_shots 中对应的参考视频镜头
  → 和 text_unit 的 ref_start/ref_end 计算重叠
```

如果某个字幕单元和某个 range 在参考视频时间轴上有重叠，就认为它们相关。

如果找不到明确关联，则会使用兜底策略：

```txt
range 找不到 related_units：
  使用当前 segment 的全部 text_units。

text_unit 找不到 related ranges：
  如果当前 segment 有 ranges，则关联全部 ranges。
```

这个兜底能避免数据缺失时完全无法判定，但也意味着输入的 `ref_shots` 和 `source_ref_shot_id` 越准确，判断越可靠。

## 9. CLI 使用方式

### 9.1 使用已有原电影字幕

推荐方式：

```powershell
python .\5_audio_role_classifier\run.py `
  --script-mapping .\outputs\5_script_visual_binder\script_mapping.json `
  --ref-analysis .\outputs\1_reference_analyzer\ref_analysis.json `
  --movie-subtitle-srt .\data\movie.srt `
  --output-dir .\outputs\5_audio_role_classifier
```

### 9.2 自动识别原电影字幕

```powershell
python .\5_audio_role_classifier\run.py `
  --script-mapping .\outputs\5_script_visual_binder\script_mapping.json `
  --ref-analysis .\outputs\1_reference_analyzer\ref_analysis.json `
  --movie-path .\data\movie.mp4 `
  --output-dir .\outputs\5_audio_role_classifier
```

### 9.3 在完整流水线里启用

`main.py` 中已经支持可选启用：

```powershell
python .\main.py `
  --ref-video-path .\data\ref.mp4 `
  --movie-path .\data\movie.mp4 `
  --subtitle-srt .\data\ref.srt `
  --enable-audio-role-classifier `
  --movie-subtitle-srt .\data\movie.srt
```

启用后，下游第 6 步会读取：

```txt
outputs/5_audio_role_classifier/script_mapping_with_audio.json
```

未启用时，下游仍读取：

```txt
outputs/5_script_visual_binder/script_mapping.json
```

## 10. CLI 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--script-mapping` | 必填 | 第 5 步输出的 `script_mapping.json` |
| `--ref-analysis` | 必填 | 第 1 步输出的 `ref_analysis.json` |
| `--movie-path` | 无 | 原电影路径，未传 `--movie-subtitle-srt` 时用于自动 ASR |
| `--movie-subtitle-srt` | 无 | 已有原电影字幕，传入后跳过 ASR |
| `--output-dir` | `outputs/5_audio_role_classifier` | 输出目录 |
| `--original-threshold` | `0.82` | 判定为原片对白的相似度阈值 |
| `--review-threshold` | `0.65` | 判定为人工复查的相似度阈值 |
| `--min-dialogue-chars` | `3` | 原电影字幕归一化后至少多少字符才允许判为对白 |
| `--asr-retries` | `3` | 自动 ASR 失败时的重试次数 |
| `--asr-retry-delay` | `2.0` | ASR 重试间隔秒数 |
| `--movie-asr-padding` | `0.2` | 局部 ASR 时每个匹配片段前后额外识别秒数 |
| `--movie-asr-merge-gap` | `0.5` | 相邻片段间隔小于该秒数时合并识别 |

## 11. 核心函数速查

| 函数 | 作用 |
| --- | --- |
| `_round()` | 将时间数值统一四舍五入到 3 位小数 |
| `_overlap()` | 计算两个时间区间的重叠时长 |
| `_subtitle_rows()` | 从字幕列表中取出和指定时间段有重叠的字幕 |
| `_normalize_text()` | 文本归一化，去掉标点和空白 |
| `_similarity()` | 计算两个文本的相似度 |
| `_combined_text()` | 合并多行字幕文本 |
| `_shot_lookup()` | 把 `ref_shots` 转成按 ID 查询的字典 |
| `_range_ref_overlap()` | 判断 text unit 和 movie range 在参考时间轴上的关联程度 |
| `_range_id()` | 生成 `seg_001_range_001` 形式的 range ID |
| `_unit_id()` | 生成 `seg_001_unit_001` 形式的 unit ID |
| `_best_match()` | 在参考字幕和电影字幕之间找最佳文本匹配 |
| `_decide_range()` | 决定 range 的 `audio_role` / `audio_action` |
| `_aggregate_segment_role()` | 聚合 segment 级别的音频角色 |
| `_movie_ranges_from_mapping()` | 收集所有需要 ASR 的电影时间段 |
| `_merge_time_ranges()` | 对 ASR 时间段做 padding 和合并 |
| `_extract_audio_clip_mp3()` | 用 ffmpeg 从电影中抽取局部音频 |
| `_bcut_entries_for_audio()` | 对局部音频调用 Bcut ASR |
| `_run_with_retries()` | 通用重试封装 |
| `_extract_matched_movie_srt_with_retries()` | 局部生成原电影字幕 |
| `_unit_decision()` | 决定 text unit 的角色和动作 |
| `classify_audio_roles()` | 模块主逻辑 |
| `main()` | CLI 参数解析和文件读写 |

## 12. 下游模块如何使用这些字段

### 12.1 `6_rewrite_engine`

下游仿稿模块会优先读取 `text_units`：

```txt
role = narration:
  参与仿写，生成新解说。

role = original_dialogue:
  不仿写，new_text 为空，keep_original_audio = true。
```

这样可以避免把原片对白改写成解说文案。

### 12.2 `7_timeline_composer`

时间轴模块会根据 `rewritten_units` 和 `movie_time_ranges[*].audio_action` 拆分 item：

```txt
rewrite_and_voiceover:
  audio_mode = voiceover
  OST = 0
  后续生成 TTS。

play_original_audio:
  audio_mode = original
  OST = 1
  后续保留原片音频。
```

如果一个 segment 里同时存在解说和原声，它会变成多个 timeline item。

### 12.3 `8_generate_video`

最终渲染模块需要根据 `audio_mode` 决定音频处理方式：

```txt
voiceover:
  裁视频时去掉原音轨，使用 TTS。

original:
  裁视频时保留原音轨，不生成 TTS。
```

因此，`5_audio_role_classifier` 只是把“应该怎么处理音频”的决策写进数据里，真正能不能听到原片原声，还取决于第 8 步是否完整支持 `audio_mode=original`。

## 13. 测试覆盖

当前相关测试包括：

```txt
tests/test_audio_role_classifier.py
tests/test_rewrite_engine_audio_units.py
tests/test_timeline_audio_modes.py
```

其中 `tests/test_audio_role_classifier.py` 主要验证：

1. 能按 `ref_time_range` 从参考 SRT 切出 `text_units`。
2. 能按 `movie_time_ranges` 从原电影 SRT 切出 `movie_subtitles`。
3. 文本高度相似时判为 `original_dialogue / play_original_audio`。
4. 不相似或没有电影字幕时判为 `narration_overlay / rewrite_and_voiceover`。
5. 不会修改原本的 `movie_time_ranges.start/end`。
6. 自动 ASR 失败时会按配置重试。
7. 局部 ASR 会按默认 padding 把 `10.0-11.0` 扩为 `9.8-11.2`。

## 14. 常见问题

### 14.1 为什么不用 `old_text` 直接判断？

因为 `old_text` 往往是多个参考字幕直接拼接后的长文本，里面可能同时包含解说旁白和原片对白。

如果直接对整个 `old_text` 判断，很容易把一个 mixed segment 整段误判。

当前做法是重新解析参考 SRT，按字幕行拆成 `text_units`，粒度更细。

### 14.2 为什么相似度高才保留原片对白？

这是保守策略。

误把解说当原声会导致最终视频缺少新解说；而误把原声当解说，最多是继续走原有“解说覆盖”的旧流程。

因此当前逻辑宁愿少保留，也不轻易误保留。

### 14.3 为什么 `manual_review` 不直接判为原声？

`manual_review` 表示文本有一定相似度，但未达到高置信阈值。

例如只命中几个字、ASR 有错字、或者上下文不完整时，自动判定风险较高，所以交给人工复查。

### 14.4 自动 ASR 为什么只识别匹配片段？

因为本模块只需要判断已经被第 5 步选中的电影片段。

对整部电影做 ASR 成本更高，失败概率更大，也会产生大量后续用不到的字幕。

### 14.5 `visual_match_locked` 是什么？

它表示：

```txt
音频判断只能决定声音策略，不能反向修改画面匹配结果。
```

也就是说，即使某个 range 被判为 `play_original_audio`，也不应该在本模块里修改它的 `start/end`。

## 15. 当前限制

1. 当前决策是 `rules_only`，没有 LLM 兜底。
2. 主要识别“字幕文本能对上的原片对白”，不识别无字幕音效，例如尖叫、爆炸、哭声。
3. 判断质量依赖参考 SRT、原电影 SRT、`ref_shots` 和 `source_ref_shot_id` 的准确性。
4. 如果原电影 ASR 质量差，可能把真实对白误判为解说覆盖。
5. 如果一个电影片段内字幕过短，默认不会轻易判为原声。
6. 本模块只产出数据决策，最终音频效果仍由第 8 步渲染实现决定。

## 16. 一句话总结

`5_audio_role_classifier/run.py` 的作用是：

```txt
把第 5 步输出的“画面绑定结果”升级成“画面绑定 + 音频策略结果”，
让后续模块知道哪些文字要仿写配音，哪些片段要跳过 TTS 并保留原片对白。
```

