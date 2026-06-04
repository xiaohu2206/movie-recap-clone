你的设计方向是对的。这个功能的核心不是“生成视频”，而是做一个 **参考解说视频 → 原电影时间轴 → 新解说脚本** 的映射系统。

建议把它定义成：

> **影视解说克隆 Core = 参考视频拆解 + 原片定位 + 文案仿写 + 时间轴重组**
>

前提建议只用于你有授权的影视素材和参考视频，避免直接复刻他人创作表达。

---

# 一、整体 Core 架构
```plain
输入：
  1. 参考影视解说视频 ref_video.mp4
  2. 原电影/电视剧 movie.mp4

输出：
  1. rewritten_script.json     新文案
  2. script_mapping.json       新旧文案和画面绑定关系
  3. final_timeline.json       新视频时间轴脚本
```

核心流程：

```plain
参考视频解析
   ↓
参考视频镜头切分
   ↓
解说字幕提取
   ↓
以参考视频镜头为基准聚合解说段落
   ↓
原电影镜头切分
   ↓
参考镜头定位到原电影
   ↓
生成 ref_to_movie_timeline
   ↓
文案和画面绑定
   ↓
仿稿生成新文案
   ↓
根据新文案时长重新分配画面
   ↓
输出 final_timeline
```

---

# 二、模块设计
## 1. 参考视频解析模块
### 作用
解析别人做好的影视解说视频，拿到三个基础数据：

1. 解说字幕srt文本
2. 参考视频镜头列表，每个镜头的关键帧

### 输入
```json
{
  "ref_video_path": "ref_video.mp4"
}
```

### 输出
```json
{
  "ref_video_id": "ref_001",
  "duration": 180.5,
  "subtitle_srt": "ref_subtitle.srt",
  "ref_shots": [
    {
      "ref_shot_id": "ref_shot_001",
      "start": 0.0,
      "end": 2.8,
      "duration": 2.8,
      "keyframes": []
    }
  ]
}
```

### 注意点
这里的镜头拆分不需要特别复杂，建议直接以参考视频的画面变化为准。因为你最终要复刻的是参考视频的剪辑结构，而不是重新理解整部电影。

---

## 2. 解说段落切分模块
不要单纯按字幕句号切分，也不要单纯按 5 秒切分。应该以 **参考视频镜头组** 为主，字幕为辅。

### 核心逻辑
```plain
字幕时间轴 + 参考视频镜头时间轴
        ↓
找出每段解说覆盖了哪些镜头
        ↓
把连续语义相关的镜头聚合成一个 narration_segment
```

### 片段聚合规则
可以先做简单规则：

| 规则 | 说明 |
| --- | --- |
| 字幕时间重叠 | 字幕在哪些镜头时间范围内出现，就绑定哪些镜头 |
| 停顿切分 | 字幕之间停顿超过 1.2s，可以切一段 |
| 镜头数量限制 | 每段最多聚合 3～8 个参考镜头 |


### 输出结构
```json
{
  "narration_segments": [
    {
      "segment_id": "seg_001",
      "ref_start": 0.0,
      "ref_end": 8.6,
      "text": "这个男人怎么也没想到，自己平静的生活，会被一个神秘电话彻底打破。",
      "ref_shot_ids": [
        "ref_shot_001",
        "ref_shot_002",
        "ref_shot_003"
      ],
      "segment_type": "narration"
    }
  ]
}
```



## 3. 原电影镜头拆分模块
### 作用
把原电影拆成标准 movie_shots。

### 输入
```json
{
  "movie_path": "movie.mp4"
}
```

### 输出
```json
{
  "movie_id": "movie_001",
  "duration": 7200,
  "movie_shots": [
    {
      "movie_shot_id": "movie_shot_000001",
      "start": 125.3,
      "end": 128.9,
      "duration": 3.6,
      "keyframes": [],
    }
  ]
}
```



## 4. 画面定位模块
用 **画面指纹匹配 + 时间连续性校验**。



### 4.1 不用向量索引的定位方式
可以用三类特征：

| 特征 | 作用 |
| --- | --- |
| 感知哈希 pHash / dHash | 快速判断画面是否相似 |
| 颜色直方图 | 处理裁剪、压缩、调色后的相似画面 |
| 关键点匹配 ORB/SIFT | 处理画面缩放、裁剪、加字幕、加滤镜 |


建议简单版这样做：

```plain
参考镜头关键帧
   ↓
和原电影抽帧结果做相似度计算
   ↓
得到 topN 候选位置
   ↓
用前后镜头时间连续性修正
   ↓
输出最可信的 movie_shot
```

---

### 4.2 参考镜头定位结果
```json
{
  "ref_to_movie_timeline": [
    {
      "ref_shot_id": "ref_shot_001",
      "ref_start": 0.0,
      "ref_end": 2.8,
      "movie_start": 125.3,
      "movie_end": 128.1,
      "movie_shot_ids": [
        "movie_shot_000041"
      ],
      "match_score": 0.92,
      "match_type": "visual_hash",
      "confidence": "high",
      "status": "matched"
    },
    {
      "ref_shot_id": "ref_shot_002",
      "ref_start": 2.8,
      "ref_end": 5.6,
      "movie_start": 128.1,
      "movie_end": 131.4,
      "movie_shot_ids": [
        "movie_shot_000042"
      ],
      "match_score": 0.88,
      "match_type": "temporal_continuity",
      "confidence": "medium",
      "status": "matched"
    }
  ]
}
```

---

### 4.3 时间连续性修正
这是不用向量索引时非常重要的能力。

参考视频通常是从原电影连续片段中剪出来的，不会每个镜头都乱跳。所以可以这样做：

```plain
如果 ref_shot_001 匹配到 movie 125s
ref_shot_002 的候选有：
  A. movie 3000s，score 0.91
  B. movie 128s，score 0.84

虽然 A 分数更高，但 B 更符合时间连续性
所以选择 B
```

### 简单打分公式
```plain
final_score = visual_score * 0.7 + continuity_score * 0.3
```

如果你想更简单：

```plain
优先选视觉分最高的；
如果前后两个参考镜头已经定位成功，
则优先选原电影时间相邻的候选。
```

---

# 三、解说画面绑定模块
这个模块负责把：

```plain
narration_segment
   +
ref_to_movie_timeline
```

合并成：

```plain
script_mapping
```

也就是：

> 每一段旧解说文案，对应原电影中的哪些画面。
>

---

## 输入
```json
{
  "narration_segments": [],
  "ref_to_movie_timeline": []
}
```

## 输出
```json
{
  "script_mapping": [
    {
      "segment_id": "seg_001",
      "old_text": "这个男人怎么也没想到，自己平静的生活，会被一个神秘电话彻底打破。",
      "ref_time_range": {
        "start": 0.0,
        "end": 8.6
      },
      "movie_time_ranges": [
        {
          "start": 125.3,
          "end": 128.1,
          "source_ref_shot_id": "ref_shot_001",
          "confidence": "high"
        },
        {
          "start": 128.1,
          "end": 131.4,
          "source_ref_shot_id": "ref_shot_002",
          "confidence": "medium"
        }
      ],
      "text_role": "hook"
    }
  ]
}
```

---



# 四、仿稿模块
你的目标不是重新写一个完全不同的解说，而是：

> 基于旧文案结构，生成一个结构相似但表达不同的新文案。
>

这里建议仿稿输出不要只给纯文本，而要输出 **新旧文案映射**。

---

## 4.1 仿稿输入
```json
{
  "segment_id": "seg_001",
  "old_text": "这个男人怎么也没想到，自己平静的生活，会被一个神秘电话彻底打破。",
  "movie_time_ranges": [
    {
      "start": 125.3,
      "end": 131.4
    }
  ]
}
```

## 4.2 仿稿输出
```json
{
  "rewritten_script": [
    {
      "segment_id": "seg_001",
      "new_text": "他本以为这只是普通的一天，直到那通诡异的电话响起，一切开始失控。",
    }
  ]
}
```

---

## 4.3 仿稿需要控制的重点
| 控制点 | 说明 |
| --- | --- |
| 不改变剧情事实 | 新文案不能说画面里没有发生的事 |
| 不大幅改变文案长度 | 新文案时长最好接近旧文案 |
| 保留叙事功能 | 钩子还是钩子，转折还是转折，情绪铺垫还是情绪铺垫 |
| 保留画面绑定 | 不能写出和当前画面冲突的内容 |
| 避免逐字洗稿 | 不能只是近义词替换 |


---

# 五、生成新视频脚本时间轴
这个模块不负责生成剪映草稿，只负责生成一个标准 `final_timeline.json`。

核心逻辑：

```plain
new_text
   ↓
估算 TTS 时长
   ↓
根据原 movie_time_ranges 分配画面
   ↓
如果文案变长，则扩展相邻原片镜头
   ↓
如果文案变短，则裁掉部分画面
   ↓
输出最终时间轴
```

---

## 输出结构
```json
{
  "final_timeline": [
    {
      "item_id": "item_001",
      "segment_id": "seg_001",
      "narration": "他本以为这只是普通的一天，直到那通诡异的电话响起，一切开始失控。",
      "tts_duration": 6.2,
      "video_clips": [
        {
          "movie_start": 125.3,
          "movie_end": 128.1,
          "duration": 2.8,
          "source": "movie.mp4"
        },
        {
          "movie_start": 128.1,
          "movie_end": 131.5,
          "duration": 3.4,
          "source": "movie.mp4"
        }
      ],
      "ref_source": {
        "ref_start": 0.0,
        "ref_end": 8.6,
        "ref_shot_ids": [
          "ref_shot_001",
          "ref_shot_002"
        ]
      },
      "confidence": "high"
    }
  ]
}
```

---

# 生成视频
生成剪映草稿
生成视频

# 六、推荐的最终 Core 数据流
你可以把整个系统设计成 7 个核心 JSON。

## 1. `ref_subtitles.json`
```json
{
  "subtitles": [
    {
      "subtitle_id": "sub_001",
      "start": 0.2,
      "end": 2.5,
      "text": "这个男人怎么也没想到"
    }
  ]
}
```

## 2. `ref_shots.json`
```json
{
  "ref_shots": [
    {
      "ref_shot_id": "ref_shot_001",
      "start": 0.0,
      "end": 2.8,
      "keyframes": [],
    }
  ]
}
```

## 3. `narration_segments.json`
```json
{
  "narration_segments": [
    {
      "segment_id": "seg_001",
      "ref_start": 0.0,
      "ref_end": 8.6,
      "text": "",
      "ref_shot_ids": [],
      "text_role": "hook"
    }
  ]
}
```

## 4. `movie_shots.json`
```json
{
  "movie_shots": [
    {
      "movie_shot_id": "movie_shot_001",
      "start": 120.0,
      "end": 124.5,
      "keyframes": [],
    }
  ]
}
```

## 5. `ref_to_movie_timeline.json`
```json
{
  "mappings": [
    {
      "ref_shot_id": "ref_shot_001",
      "movie_start": 120.0,
      "movie_end": 124.5,
      "match_score": 0.91,
      "confidence": "high"
    }
  ]
}
```

## 6. `script_mapping.json`
```json
{
  "script_mapping": [
    {
      "segment_id": "seg_001",
      "old_text": "",
      "movie_time_ranges": [],
    }
  ]
}
```

## 7. `final_timeline.json`
```json
{
  "final_timeline": [
    {
      "item_id": "item_001",
      "segment_id": "seg_001",
      "narration": "",
      "video_clips": []
    }
  ]
}
```

---

# 七、简洁版架构图
```plain
                ┌────────────────────┐
                │ 参考解说视频        │
                └─────────┬──────────┘
                          ↓
        ┌────────────────────────────────┐
        │ 参考视频解析模块                │
        │ - 提取字幕                      │
        │ - 拆参考镜头                    │
        │ - 抽关键帧                      │
        └─────────┬──────────────────────┘
                  ↓
        ┌────────────────────────────────┐
        │ 解说段落切分模块                │
        │ - 字幕按参考镜头聚合            │
        │ - 生成 narration_segments       │
        └─────────┬──────────────────────┘
                  ↓
┌─────────────────┴──────────────────┐
│                                    │
↓                                    ↓
┌────────────────────┐       ┌────────────────────┐
│ 原电影镜头拆分模块  │       │ 参考镜头特征提取模块 │
│ movie_shots         │       │ ref_shot_features   │
└─────────┬──────────┘       └─────────┬──────────┘
          ↓                            ↓
          └────────────┬───────────────┘
                       ↓
        ┌────────────────────────────────┐
        │ 画面定位模块                    │
        │ - 参考镜头匹配原片位置          │
        │ - 时间连续性修正                │
        │ - 输出 ref_to_movie_timeline    │
        └─────────┬──────────────────────┘
                  ↓
        ┌────────────────────────────────┐
        │ 解说画面绑定模块                │
        │ - segment 绑定 movie clips      │
        │ - 输出 script_mapping           │
        └─────────┬──────────────────────┘
                  ↓
        ┌────────────────────────────────┐
        │ 仿稿模块                        │
        │ - 保留结构                      │
        │ - 重写表达                      │
        │ - 新旧文案映射                  │
        └─────────┬──────────────────────┘
                  ↓
        ┌────────────────────────────────┐
        │ 新视频时间轴生成模块            │
        │ - 根据新文案时长调整画面        │
        │ - 输出 final_timeline           │
        └────────────────────────────────┘
```

---

# 八、我建议你调整后的模块命名
你现在的模块已经很接近了，我建议稍微改成下面这样，更清晰：

| 原模块 | 建议模块名 | 说明 |
| --- | --- | --- |
| 参考视频解析模块 | Reference Analyzer | 解析参考视频 |
| 解说段落切分模块 | Narration Segmenter | 生成解说段落 |
| 原电影镜头拆分模块 | Movie Shot Parser | 生成原片镜头库 |
| 画面定位模块 | Visual Alignment Engine | 参考视频画面对齐原电影 |
| 解说画面绑定模块 | Script-Visual Binder | 文案和画面建立关系 |
| 仿稿模块 | Rewrite Engine | 生成新文案 |
| 生成新视频脚本时间轴 | Timeline Composer | 生成新时间轴 |


---

# 九、最小可行版本 MVP
不要一开始做太复杂。MVP 可以这样：

## MVP 版本只做 5 件事
```plain
1. 参考视频提取字幕
2. 参考视频拆镜头
3. 原电影拆镜头
4. 用关键帧相似度定位参考镜头在原电影的位置
5. 按原解说段落仿写文案，输出 final_timeline
```



# 十、核心难点和兜底策略
## 难点 1：参考视频被裁剪、加字幕、调色
解决：

```plain
不要只用单帧 hash。
使用：
- pHash
- 颜色直方图
- ORB 局部特征
- 多关键帧投票
```

---

## 难点 2：参考视频有素材不是原电影里的
比如博主加了表情包、素材图、转场、花字。

解决：

```json
{
  "ref_shot_id": "ref_shot_009",
  "status": "unmatched",
  "reason": "non_movie_material"
}
```

这类镜头不要强行匹配，直接跳过或者用相邻原片画面补齐。

---

## 难点 3：新文案比旧文案长
解决：

```plain
优先扩展当前 movie_time_range 的尾部；
不够时取相邻 movie_shot；
仍不够时降低语速或拆成两段。
```

---

## 难点 4：新文案比旧文案短
解决：

```plain
优先裁剪低信息量镜头；
保留开头和结尾关键画面；
不要强行把所有旧画面都塞进去。
```

---

## 难点 5：画面定位失败
兜底优先级：

```plain
1. 用相邻参考镜头推断位置
2. 用原电影相邻镜头补齐
3. 用 visual_desc 人工/模型判断
4. 标记 low_confidence，进入人工审核
```

---

# 十一、最终推荐架构
你可以最终整理成这样：

```plain
Core Pipeline：

1. Reference Analyzer
   输入参考解说视频
   输出 ref_shots、ref_subtitles

2. Narration Segmenter
   输入 ref_shots、ref_subtitles
   输出 narration_segments

3. Movie Shot Parser
   输入原电影
   输出 movie_shots

4. Visual Alignment Engine
   输入 ref_shots、movie_shots
   输出 ref_to_movie_timeline

5. Script-Visual Binder
   输入 narration_segments、ref_to_movie_timeline
   输出 script_mapping

6. Rewrite Engine
   输入 script_mapping
   输出 rewritten_script

7. Timeline Composer
   输入 rewritten_script、script_mapping
   输出 final_timeline
```

---

# 十二、一句话总结
这个功能的核心不是让 AI 重新理解电影，而是把参考视频当成“剪辑模板”：

```plain
参考视频 = 剪辑结构模板
原电影 = 画面素材来源
原解说文案 = 叙事结构模板
新文案 = 换一种表达但不改变画面语义
final_timeline = 新文案 + 原电影画面重新绑定
```

所以最关键的三个核心表是：

```plain
narration_segments.json
ref_to_movie_timeline.json
script_mapping.json
```

只要这三个数据稳定，后面的仿稿、配音、时间轴生成都会比较顺。
