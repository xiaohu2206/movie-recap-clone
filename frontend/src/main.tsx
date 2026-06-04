import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import {
  ArrowCounterClockwise,
  CheckCircle,
  CircleNotch,
  ClockCounterClockwise,
  Export,
  File,
  FilmSlate,
  Folder,
  GearSix,
  Info,
  Lightning,
  Play,
  Prohibit,
  Queue,
  SlidersHorizontal,
  Sparkle,
  Stop,
  TerminalWindow,
  UploadSimple,
  Video,
  WarningCircle,
} from "@phosphor-icons/react";
import "./styles.css";

type RenderMode = "none" | "draft" | "video" | "both";
type BackendMode = "auto" | "transnet" | "opencv";
type VideoEncoder = "auto" | "libx264" | "h264_nvenc";

type PipelineConfig = {
  refVideoPath: string;
  moviePath: string;
  subtitlePath: string;
  outputRoot: string;
  asrProvider: "bcut" | "none";
  threshold: number;
  backend: BackendMode;
  aiApiKey: string;
  aiBaseUrl: string;
  aiModel: string;
  aiTemperature: number;
  charsPerSecond: number;
  renderMode: RenderMode;
  edgeVoiceId: string;
  edgeTtsSpeed: number;
  jianyingDraftDir: string;
  videoOutputName: string;
  videoEncoder: VideoEncoder;
};

type StageState = "waiting" | "running" | "done" | "failed";

type Stage = {
  id: string;
  step: number;
  title: string;
  detail: string;
  output: string;
  patterns: string[];
};

type LogLine = {
  id: number;
  level: "info" | "error" | "system";
  text: string;
};

type BackendInfo = {
  root: string;
  python: string;
  packaged: boolean;
  hasLocalVenv: boolean;
};

const stages: Stage[] = [
  {
    id: "reference",
    step: 1,
    title: "参考视频解析",
    detail: "提取参考视频字幕、关键帧和节奏信息。",
    output: "outputs/1_reference_analyzer/ref_analysis.json",
    patterns: ["1_reference_analyzer", "ref_analysis.json"],
  },
  {
    id: "segments",
    step: 2,
    title: "解说段落切分",
    detail: "把参考解说拆成可改写、可对齐的语义段。",
    output: "outputs/2_narration_segmenter/narration_segments.json",
    patterns: ["2_narration_segmenter", "narration_segments.json"],
  },
  {
    id: "shots",
    step: 3,
    title: "原片镜头拆分",
    detail: "识别原片镜头边界，建立素材检索空间。",
    output: "outputs/3_movie_shot_parser/movie_shots.json",
    patterns: ["3_movie_shot_parser", "movie_shots.json"],
  },
  {
    id: "alignment",
    step: 4,
    title: "画面对齐",
    detail: "将参考节奏映射到原片镜头时间线。",
    output: "outputs/4_visual_alignment_engine/ref_to_movie_timeline.json",
    patterns: ["4_visual_alignment_engine", "ref_to_movie_timeline.json"],
  },
  {
    id: "binder",
    step: 5,
    title: "脚本画面绑定",
    detail: "为每段解说匹配可使用的原片画面。",
    output: "outputs/5_script_visual_binder/script_mapping.json",
    patterns: ["5_script_visual_binder", "script_mapping.json"],
  },
  {
    id: "rewrite",
    step: 6,
    title: "AI 仿写",
    detail: "调用 OpenAI 兼容接口生成新解说文案。",
    output: "outputs/6_rewrite_engine/rewritten_script.json",
    patterns: ["6_rewrite_engine", "rewritten_script.json"],
  },
  {
    id: "timeline",
    step: 7,
    title: "时间线合成",
    detail: "生成新视频脚本时间线和剪辑结构。",
    output: "outputs/7_timeline_composer/final_timeline.json",
    patterns: ["7_timeline_composer", "final_timeline.json"],
  },
  {
    id: "render",
    step: 8,
    title: "视频与剪映草稿",
    detail: "按渲染模式输出草稿、音频或 mp4 成片。",
    output: "outputs/8_generate_video/generate_video_result.json",
    patterns: ["8_generate_video", "generate_video_result.json", "clone_narration_output.mp4"],
  },
];

const defaultConfig: PipelineConfig = {
  refVideoPath: "",
  moviePath: "",
  subtitlePath: "",
  outputRoot: "",
  asrProvider: "bcut",
  threshold: 0.5,
  backend: "auto",
  aiApiKey: "",
  aiBaseUrl: "https://api.openai.com/v1",
  aiModel: "gpt-4o-mini",
  aiTemperature: 0.7,
  charsPerSecond: 4.2,
  renderMode: "both",
  edgeVoiceId: "zh-CN-XiaoxiaoNeural",
  edgeTtsSpeed: 1,
  jianyingDraftDir: "",
  videoOutputName: "clone_narration_output.mp4",
  videoEncoder: "auto",
};

const previewBridge = {
  selectFile: async () => "",
  selectDirectory: async () => "",
  startPipeline: async () => ({ ok: false, error: "当前是浏览器预览模式，请在 Electron 中运行生成任务。" }),
  stopPipeline: async () => ({ ok: true }),
  revealPath: async () => false,
  openPath: async () => false,
  getBackendInfo: async (): Promise<BackendInfo> => ({
    root: "browser-preview",
    python: "Electron preload 未连接",
    packaged: false,
    hasLocalVenv: false,
  }),
  onPipelineEvent: () => () => undefined,
};

const cloneBridge = window.cloneApp ?? previewBridge;

const videoFilters = [
  { name: "视频文件", extensions: ["mp4", "mov", "mkv", "avi", "webm"] },
  { name: "全部文件", extensions: ["*"] },
];

const subtitleFilters = [
  { name: "字幕文件", extensions: ["srt", "vtt", "ass"] },
  { name: "全部文件", extensions: ["*"] },
];

function cx(...items: Array<string | false | null | undefined>) {
  return items.filter(Boolean).join(" ");
}

function fileName(value: string) {
  if (!value) {
    return "未选择";
  }
  return value.split(/[\\/]/).pop() || value;
}

function outputPath(root: string, relative: string) {
  if (!root) {
    return relative;
  }
  return `${root.replace(/[\\/]$/, "")}\\${relative.replace("outputs/", "").replace(/\//g, "\\")}`;
}

function App() {
  const [config, setConfig] = useState<PipelineConfig>(defaultConfig);
  const [backendInfo, setBackendInfo] = useState<BackendInfo | null>(null);
  const [activeTab, setActiveTab] = useState<"setup" | "pipeline" | "outputs">("setup");
  const [isRunning, setIsRunning] = useState(false);
  const [startedAt, setStartedAt] = useState<Date | null>(null);
  const [finishedCode, setFinishedCode] = useState<number | null>(null);
  const [logLines, setLogLines] = useState<LogLine[]>([
    { id: 1, level: "system", text: "工作台已就绪。请选择参考视频和原片后启动流水线。" },
  ]);
  const [stageStates, setStageStates] = useState<Record<string, StageState>>(() =>
    Object.fromEntries(stages.map((stage) => [stage.id, "waiting" as StageState])),
  );
  const nextLogId = useRef(2);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    cloneBridge.getBackendInfo().then((info) => {
      setBackendInfo(info);
      setConfig((prev) => ({
        ...prev,
        outputRoot: prev.outputRoot || `${info.root}\\outputs`,
      }));
    });

    const unsubscribe = cloneBridge.onPipelineEvent((event) => {
      if (event.type === "started") {
        setIsRunning(true);
        setStartedAt(new Date());
        setFinishedCode(null);
        setStageStates(Object.fromEntries(stages.map((stage) => [stage.id, "waiting" as StageState])));
        pushLog("system", `启动命令: ${event.command}`);
        pushLog("system", `输出目录: ${event.outputRoot}`);
        markStageFromText(event.command);
      }

      if (event.type === "stdout") {
        appendChunk("info", event.text);
        markStageFromText(event.text);
      }

      if (event.type === "stderr") {
        appendChunk("error", event.text);
        markStageFromText(event.text);
      }

      if (event.type === "error") {
        setIsRunning(false);
        markCurrentStageFailed();
        pushLog("error", event.error);
      }

      if (event.type === "stopped") {
        setIsRunning(false);
        pushLog("system", "任务已停止。");
      }

      if (event.type === "finished") {
        setIsRunning(false);
        setFinishedCode(event.code ?? null);
        setStageStates((previous) => {
          const next = { ...previous };
          const hasFailure = event.code !== 0;
          for (const stage of stages) {
            if (next[stage.id] === "running") {
              next[stage.id] = hasFailure ? "failed" : "done";
            }
            if (!hasFailure && next[stage.id] === "waiting") {
              next[stage.id] = config.renderMode === "none" && stage.id === "render" ? "waiting" : "done";
            }
          }
          return next;
        });
        pushLog(event.code === 0 ? "system" : "error", event.code === 0 ? "流水线完成。" : `流水线退出，代码 ${event.code}`);
      }
    });

    return unsubscribe;
  }, [config.renderMode]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  function pushLog(level: LogLine["level"], text: string) {
    const clean = text.trim();
    if (!clean) {
      return;
    }
    setLogLines((previous) => [...previous.slice(-400), { id: nextLogId.current++, level, text: clean }]);
  }

  function appendChunk(level: LogLine["level"], text: string) {
    text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .forEach((line) => pushLog(level, line));
  }

  function markStageFromText(text: string) {
    const nextIndex = stages.findIndex((stage) => stage.patterns.some((pattern) => text.includes(pattern)));
    if (nextIndex < 0) {
      return;
    }
    setStageStates((previous) => {
      const next = { ...previous };
      stages.forEach((stage, index) => {
        if (index < nextIndex && next[stage.id] !== "failed") {
          next[stage.id] = "done";
        }
        if (index === nextIndex && next[stage.id] !== "done" && next[stage.id] !== "failed") {
          next[stage.id] = "running";
        }
      });
      return next;
    });
  }

  function markCurrentStageFailed() {
    setStageStates((previous) => {
      const next = { ...previous };
      const runningStage = stages.find((stage) => next[stage.id] === "running");
      if (runningStage) {
        next[runningStage.id] = "failed";
      }
      return next;
    });
  }

  function updateConfig<K extends keyof PipelineConfig>(key: K, value: PipelineConfig[K]) {
    setConfig((previous) => ({ ...previous, [key]: value }));
  }

  async function chooseVideo(key: "refVideoPath" | "moviePath", title: string) {
    const selected = await cloneBridge.selectFile({ title, filters: videoFilters });
    if (selected) {
      updateConfig(key, selected);
    }
  }

  async function chooseSubtitle() {
    const selected = await cloneBridge.selectFile({ title: "选择字幕文件", filters: subtitleFilters });
    if (selected) {
      updateConfig("subtitlePath", selected);
    }
  }

  async function chooseDirectory(key: "outputRoot" | "jianyingDraftDir") {
    const selected = await cloneBridge.selectDirectory();
    if (selected) {
      updateConfig(key, selected);
    }
  }

  async function startPipeline() {
    if (!config.refVideoPath || !config.moviePath) {
      pushLog("error", "请先选择参考视频和原片。");
      setActiveTab("setup");
      return;
    }

    setActiveTab("pipeline");
    const result = await cloneBridge.startPipeline(config);
    if (!result.ok) {
      pushLog("error", result.error || "启动失败。");
    }
  }

  async function stopPipeline() {
    await cloneBridge.stopPipeline();
  }

  const completedCount = useMemo(
    () => Object.values(stageStates).filter((state) => state === "done").length,
    [stageStates],
  );
  const progress = Math.round((completedCount / stages.length) * 100);
  const canStart = config.refVideoPath && config.moviePath && !isRunning;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark">
            <FilmSlate weight="duotone" />
          </div>
          <div>
            <strong>Clone Studio</strong>
            <span>视频解说克隆工作台</span>
          </div>
        </div>

        <nav className="nav-stack" aria-label="主导航">
          <button className={cx("nav-item", activeTab === "setup" && "active")} onClick={() => setActiveTab("setup")}>
            <SlidersHorizontal /> 项目配置
          </button>
          <button className={cx("nav-item", activeTab === "pipeline" && "active")} onClick={() => setActiveTab("pipeline")}>
            <Queue /> 流水线
          </button>
          <button className={cx("nav-item", activeTab === "outputs" && "active")} onClick={() => setActiveTab("outputs")}>
            <Export /> 产物
          </button>
        </nav>

        <div className="sidebar-status">
          <div className="progress-ring" style={{ "--progress": `${progress}%` } as React.CSSProperties}>
            <span>{progress}</span>
          </div>
          <div>
            <strong>{isRunning ? "正在生成" : finishedCode === 0 ? "已完成" : "待启动"}</strong>
            <span>{completedCount} / {stages.length} 阶段完成</span>
          </div>
        </div>

        <div className="backend-card">
          <div className="backend-card-title">
            <Info /> 后端
          </div>
          <p>{backendInfo?.python || "检测中"}</p>
          <p>{backendInfo?.root || "正在读取项目路径"}</p>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <p className="kicker">Production pipeline</p>
            <h1>从参考解说到可交付视频，一次跑完整条链路。</h1>
          </div>
          <div className="topbar-actions">
            <button className="ghost-button" onClick={() => setConfig(defaultConfig)} disabled={isRunning}>
              <ArrowCounterClockwise /> 重置
            </button>
            {isRunning ? (
              <button className="danger-button" onClick={stopPipeline}>
                <Stop weight="fill" /> 停止
              </button>
            ) : (
              <button className="primary-button" onClick={startPipeline} disabled={!canStart}>
                <Play weight="fill" /> 开始生成
              </button>
            )}
          </div>
        </header>

        {activeTab === "setup" && (
          <section className="content-grid setup-grid">
            <div className="panel wide-panel">
              <div className="panel-heading">
                <div>
                  <p className="section-label">输入素材</p>
                  <h2>建立一个可复现的生成任务</h2>
                </div>
              </div>

              <div className="asset-grid">
                <AssetPicker
                  icon={<Video />}
                  title="参考视频"
                  value={config.refVideoPath}
                  description="用于学习解说节奏、字幕和镜头参考。"
                  actionLabel="选择视频"
                  onChoose={() => chooseVideo("refVideoPath", "选择参考视频")}
                />
                <AssetPicker
                  icon={<FilmSlate />}
                  title="原片素材"
                  value={config.moviePath}
                  description="用于重新匹配画面并生成新时间线。"
                  actionLabel="选择原片"
                  onChoose={() => chooseVideo("moviePath", "选择原片素材")}
                />
                <AssetPicker
                  icon={<File />}
                  title="参考字幕"
                  value={config.subtitlePath}
                  description="可选。提供 srt 会跳过 ASR，速度更快。"
                  actionLabel="选择字幕"
                  onChoose={chooseSubtitle}
                  onClear={() => updateConfig("subtitlePath", "")}
                />
              </div>
            </div>

            <div className="panel">
              <div className="panel-heading compact">
                <div>
                  <p className="section-label">生成模式</p>
                  <h2>输出目标</h2>
                </div>
              </div>
              <SegmentedControl
                value={config.renderMode}
                options={[
                  ["none", "时间线"],
                  ["draft", "剪映草稿"],
                  ["video", "MP4"],
                  ["both", "全部"],
                ]}
                onChange={(value) => updateConfig("renderMode", value as RenderMode)}
              />
              <div className="form-stack">
                <label>
                  输出目录
                  <div className="input-with-action">
                    <input value={config.outputRoot} onChange={(event) => updateConfig("outputRoot", event.target.value)} />
                    <button onClick={() => chooseDirectory("outputRoot")}><Folder /></button>
                  </div>
                </label>
                <label>
                  视频文件名
                  <input value={config.videoOutputName} onChange={(event) => updateConfig("videoOutputName", event.target.value)} />
                </label>
              </div>
            </div>

            <div className="panel">
              <div className="panel-heading compact">
                <div>
                  <p className="section-label">AI 设置</p>
                  <h2>OpenAI 兼容接口</h2>
                </div>
              </div>
              <div className="form-stack">
                <label>
                  API Key
                  <input
                    type="password"
                    value={config.aiApiKey}
                    placeholder="可留空，使用环境变量"
                    onChange={(event) => updateConfig("aiApiKey", event.target.value)}
                  />
                </label>
                <label>
                  Base URL
                  <input value={config.aiBaseUrl} onChange={(event) => updateConfig("aiBaseUrl", event.target.value)} />
                </label>
                <label>
                  模型
                  <input value={config.aiModel} onChange={(event) => updateConfig("aiModel", event.target.value)} />
                </label>
              </div>
            </div>

            <div className="panel wide-panel">
              <div className="panel-heading">
                <div>
                  <p className="section-label">高级参数</p>
                  <h2>性能、镜头识别与配音</h2>
                </div>
              </div>
              <div className="settings-grid">
                <SelectField label="镜头后端" value={config.backend} onChange={(value) => updateConfig("backend", value as BackendMode)} options={["auto", "transnet", "opencv"]} />
                <SelectField label="视频编码" value={config.videoEncoder} onChange={(value) => updateConfig("videoEncoder", value as VideoEncoder)} options={["auto", "libx264", "h264_nvenc"]} />
                <NumberField label="镜头阈值" value={config.threshold} min={0.1} max={0.9} step={0.05} onChange={(value) => updateConfig("threshold", value)} />
                <NumberField label="AI 温度" value={config.aiTemperature} min={0} max={1.5} step={0.1} onChange={(value) => updateConfig("aiTemperature", value)} />
                <NumberField label="每秒字数" value={config.charsPerSecond} min={2} max={8} step={0.1} onChange={(value) => updateConfig("charsPerSecond", value)} />
                <NumberField label="配音速度" value={config.edgeTtsSpeed} min={0.5} max={2} step={0.05} onChange={(value) => updateConfig("edgeTtsSpeed", value)} />
                <label>
                  Edge Voice ID
                  <input value={config.edgeVoiceId} onChange={(event) => updateConfig("edgeVoiceId", event.target.value)} />
                </label>
                <label>
                  剪映草稿目录
                  <div className="input-with-action">
                    <input value={config.jianyingDraftDir} placeholder="可选" onChange={(event) => updateConfig("jianyingDraftDir", event.target.value)} />
                    <button onClick={() => chooseDirectory("jianyingDraftDir")}><Folder /></button>
                  </div>
                </label>
              </div>
            </div>
          </section>
        )}

        {activeTab === "pipeline" && (
          <section className="pipeline-layout">
            <div className="stage-list">
              {stages.map((stage) => (
                <StageRow
                  key={stage.id}
                  stage={stage}
                  state={stageStates[stage.id]}
                  targetPath={outputPath(config.outputRoot, stage.output)}
                />
              ))}
            </div>

            <div className="log-panel">
              <div className="panel-heading compact">
                <div>
                  <p className="section-label">实时日志</p>
                  <h2>运行反馈</h2>
                </div>
                <TerminalWindow />
              </div>
              <div className="log-body" ref={logRef}>
                {logLines.map((line) => (
                  <div key={line.id} className={cx("log-line", line.level)}>
                    <span>{line.level}</span>
                    <p>{line.text}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {activeTab === "outputs" && (
          <section className="content-grid outputs-grid">
            <div className="panel wide-panel output-hero">
              <div>
                <p className="section-label">交付检查</p>
                <h2>{finishedCode === 0 ? "最近一次任务已完成" : "产物会在流水线完成后出现在这里"}</h2>
                <p>可以直接打开输出目录，或定位到关键 JSON、剪映草稿和最终 mp4。</p>
              </div>
              <button className="primary-button" onClick={() => cloneBridge.openPath(config.outputRoot)} disabled={!config.outputRoot}>
                <Folder weight="fill" /> 打开输出目录
              </button>
            </div>

            {stages.map((stage) => (
              <button
                key={stage.id}
                className="output-tile"
                onClick={() => cloneBridge.revealPath(outputPath(config.outputRoot, stage.output))}
              >
                <span>{String(stage.step).padStart(2, "0")}</span>
                <strong>{stage.title}</strong>
                <small>{outputPath(config.outputRoot, stage.output)}</small>
              </button>
            ))}
          </section>
        )}
      </main>
    </div>
  );
}

function AssetPicker(props: {
  icon: React.ReactNode;
  title: string;
  value: string;
  description: string;
  actionLabel: string;
  onChoose: () => void;
  onClear?: () => void;
}) {
  return (
    <div className={cx("asset-picker", props.value && "selected")}>
      <div className="asset-icon">{props.icon}</div>
      <div className="asset-copy">
        <strong>{props.title}</strong>
        <span>{props.description}</span>
        <p title={props.value}>{fileName(props.value)}</p>
      </div>
      <div className="asset-actions">
        <button onClick={props.onChoose}>
          <UploadSimple /> {props.actionLabel}
        </button>
        {props.value && props.onClear && (
          <button className="icon-only" onClick={props.onClear} aria-label="清除">
            <Prohibit />
          </button>
        )}
      </div>
    </div>
  );
}

function SegmentedControl(props: {
  value: string;
  options: Array<[string, string]>;
  onChange: (value: string) => void;
}) {
  return (
    <div className="segmented-control">
      {props.options.map(([value, label]) => (
        <button key={value} className={props.value === value ? "active" : ""} onClick={() => props.onChange(value)}>
          {label}
        </button>
      ))}
    </div>
  );
}

function SelectField(props: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label>
      {props.label}
      <select value={props.value} onChange={(event) => props.onChange(event.target.value)}>
        {props.options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}

function NumberField(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label>
      <span className="range-label">
        {props.label}
        <strong>{props.value}</strong>
      </span>
      <input
        type="range"
        value={props.value}
        min={props.min}
        max={props.max}
        step={props.step}
        onChange={(event) => props.onChange(Number(event.target.value))}
      />
    </label>
  );
}

function StageRow({ stage, state, targetPath }: { stage: Stage; state: StageState; targetPath: string }) {
  const icon = {
    waiting: <ClockCounterClockwise />,
    running: <CircleNotch className="spin" />,
    done: <CheckCircle weight="fill" />,
    failed: <WarningCircle weight="fill" />,
  }[state];

  return (
    <article className={cx("stage-row", state)}>
      <div className="stage-index">{String(stage.step).padStart(2, "0")}</div>
      <div className="stage-main">
        <div>
          <strong>{stage.title}</strong>
          <span>{stage.detail}</span>
        </div>
        <button onClick={() => cloneBridge.revealPath(targetPath)}>
          <Export /> 定位产物
        </button>
      </div>
      <div className="stage-state">
        {icon}
        <span>{stateText(state)}</span>
      </div>
    </article>
  );
}

function stateText(state: StageState) {
  if (state === "running") {
    return "运行中";
  }
  if (state === "done") {
    return "完成";
  }
  if (state === "failed") {
    return "失败";
  }
  return "等待";
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
