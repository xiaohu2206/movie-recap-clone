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
  MagnifyingGlass,
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
type PipelineMode = "clone" | "ref_audio_rebuild";
type BackendMode = "auto" | "transnet" | "opencv";
type VideoEncoder = "auto" | "libx264" | "h264_nvenc";

type PipelineConfig = {
  pipelineMode: PipelineMode;
  refVideoPath: string;
  moviePath: string;
  subtitlePath: string;
  movieSubtitlePath: string;
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

type PipelineRunMode = "normal" | "resume" | "restart";

type PipelineResumeState = {
  outputRoot: string;
  completedStages: string[];
};

type StageState = "waiting" | "running" | "done" | "failed" | "skipped";

type StageProgress = {
  percent: number;
  message: string;
};

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
    id: "ref_audio_rebuild",
    step: 4.1,
    title: "参考音频重组",
    detail: "复用参考视频原音频，并替换为对齐后的原片画面。",
    output: "outputs/4.1_ref_audio_rebuild_composer/ref_audio_rebuild_timeline.json",
    patterns: ["4.1_ref_audio_rebuild_composer", "ref_audio_rebuild_timeline.json"],
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
    id: "subtitle",
    step: 5.1,
    title: "原片字幕补充",
    detail: "为绑定画面补充原片台词字幕。",
    output: "outputs/5.1_movie_subtitle_filler/script_mapping_subtitled.json",
    patterns: ["5.1_movie_subtitle_filler", "script_mapping_subtitled.json"],
  },
  {
    id: "audio_role",
    step: 5.2,
    title: "原声判定",
    detail: "判断每段使用 AI 配音还是保留原片原声。",
    output: "outputs/5.2_audio_role_classifier/script_mapping_with_audio.json",
    patterns: ["5.2_audio_role_classifier", "script_mapping_with_audio.json"],
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
  pipelineMode: "clone",
  refVideoPath: "",
  moviePath: "",
  subtitlePath: "",
  movieSubtitlePath: "",
  outputRoot: "",
  asrProvider: "bcut",
  threshold: 0.5,
  backend: "auto",
  aiApiKey: "",
  aiBaseUrl: "https://api.openai.com/v1",
  aiModel: "gpt-4o-mini",
  aiTemperature: 0.7,
  charsPerSecond: 4.2,
  renderMode: "draft",
  edgeVoiceId: "zh-CN-XiaoxiaoNeural",
  edgeTtsSpeed: 1,
  jianyingDraftDir: "",
  videoOutputName: "clone_narration_output.mp4",
  videoEncoder: "auto",
};

const previewBridge = {
  selectFile: async () => "",
  selectDirectory: async () => "",
  detectJianyingDraftDir: async () => "",
  loadConfig: async () => null,
  saveConfig: async () => ({ ok: true }),
  testAi: async () => ({ ok: false, error: "当前是浏览器预览模式，请在 Electron 中测试连接。" }),
  startPipeline: async () => ({ ok: false, error: "当前是浏览器预览模式，请在 Electron 中运行生成任务。" }),
  getPipelineState: async () => ({
    ok: true,
    outputRoot: "",
    completed: false,
    canResume: false,
    completedStages: [],
  }),
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

function isStageEnabled(stage: Stage, renderMode: RenderMode, pipelineMode: PipelineMode) {
  if (stage.id === "render") {
    return renderMode !== "none";
  }
  if (pipelineMode === "ref_audio_rebuild") {
    return ["reference", "shots", "alignment", "ref_audio_rebuild"].includes(stage.id);
  }
  return stage.id !== "ref_audio_rebuild";
}

function createStageStates(renderMode: RenderMode, pipelineMode: PipelineMode) {
  return Object.fromEntries(
    stages.map((stage) => [stage.id, isStageEnabled(stage, renderMode, pipelineMode) ? "waiting" : "skipped"]),
  ) as Record<string, StageState>;
}

function renderModeFromTargets(outputDraft: boolean, outputVideo: boolean): RenderMode {
  if (outputDraft && outputVideo) {
    return "both";
  }
  if (outputDraft) {
    return "draft";
  }
  if (outputVideo) {
    return "video";
  }
  return "none";
}

function renderTargetsFromMode(renderMode: RenderMode) {
  return {
    outputDraft: renderMode === "draft" || renderMode === "both",
    outputVideo: renderMode === "video" || renderMode === "both",
  };
}

function App() {
  const [config, setConfig] = useState<PipelineConfig>(defaultConfig);
  const [backendInfo, setBackendInfo] = useState<BackendInfo | null>(null);
  const [activeTab, setActiveTab] = useState<"setup" | "pipeline" | "outputs">("setup");
  const [isRunning, setIsRunning] = useState(false);
  const [startedAt, setStartedAt] = useState<Date | null>(null);
  const [finishedCode, setFinishedCode] = useState<number | null>(null);
  const [resumeState, setResumeState] = useState<PipelineResumeState | null>(null);
  const [pipelineLogPath, setPipelineLogPath] = useState("");
  const [logLines, setLogLines] = useState<LogLine[]>([
    { id: 1, level: "system", text: "工作台已就绪。请选择参考视频和原片后启动流水线。" },
  ]);
  const [stageStates, setStageStates] = useState<Record<string, StageState>>(() => createStageStates(defaultConfig.renderMode, defaultConfig.pipelineMode));
  const [stageProgress, setStageProgress] = useState<Record<string, StageProgress>>({});
  const [aiTest, setAiTest] = useState<{ status: "idle" | "testing" | "ok" | "error"; message: string }>({
    status: "idle",
    message: "",
  });
  const [draftDetect, setDraftDetect] = useState<{ status: "idle" | "detecting" | "found" | "missing"; message: string }>({
    status: "idle",
    message: "",
  });
  const nextLogId = useRef(2);
  const logRef = useRef<HTMLDivElement>(null);
  const configLoaded = useRef(false);

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
        setPipelineLogPath(event.logPath || `${event.outputRoot}\\logs\\pipeline.log`);
        setStageStates(createStageStates(config.renderMode, config.pipelineMode));
        setStageProgress({});
        pushLog("system", `启动命令: ${event.command}`);
        pushLog("system", `输出目录: ${event.outputRoot}`);
      }

      if (event.type === "stdout") {
        appendChunk("info", event.text);
      }

      if (event.type === "stderr") {
        appendChunk("error", event.text);
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
          const hasFailure = event.code !== 0;
          if (hasFailure) {
            return Object.fromEntries(
              stages.map((stage) => [stage.id, previous[stage.id] === "running" ? "failed" : previous[stage.id]]),
            ) as Record<string, StageState>;
          }
          return Object.fromEntries(
                stages.map((stage) => [stage.id, isStageEnabled(stage, config.renderMode, config.pipelineMode) ? "done" : "skipped"]),
          ) as Record<string, StageState>;
        });
        if (event.code === 0) {
          setStageProgress((previous) =>
            Object.fromEntries(
              stages.map((stage) => [
                stage.id,
                previous[stage.id] ?? { percent: isStageEnabled(stage, config.renderMode, config.pipelineMode) ? 100 : 0, message: "" },
              ]),
            ) as Record<string, StageProgress>,
          );
        }
        pushLog(event.code === 0 ? "system" : "error", event.code === 0 ? "流水线完成。" : `流水线退出，代码 ${event.code}`);
      }
    });

    return unsubscribe;
  }, [config.renderMode, config.pipelineMode]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  useEffect(() => {
    let cancelled = false;
    cloneBridge.loadConfig().then((saved) => {
      if (!cancelled && saved && typeof saved === "object") {
        setConfig((prev) => ({ ...prev, ...(saved as Partial<PipelineConfig>) }));
      }
      configLoaded.current = true;
      const savedDraftDir =
        saved && typeof saved === "object" ? String((saved as Partial<PipelineConfig>).jianyingDraftDir || "") : "";
      if (!cancelled && !savedDraftDir) {
        void detectDraftDir(false);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!configLoaded.current) {
      return;
    }
    const timer = setTimeout(() => {
      cloneBridge.saveConfig(config);
    }, 400);
    return () => clearTimeout(timer);
  }, [config]);

  useEffect(() => {
    if (isRunning) {
      return;
    }
    setStageStates(createStageStates(config.renderMode, config.pipelineMode));
    setStageProgress({});
  }, [config.renderMode, config.pipelineMode, isRunning]);

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
      .forEach((line) => {
        if (handleProgressLine(line)) {
          return;
        }
        pushLog(level, line);
        markStageFromText(line);
      });
  }

  function handleProgressLine(line: string) {
    const match = line.match(/^\[progress\]\s+(.+)$/);
    if (!match) {
      return false;
    }
    try {
      const payload = JSON.parse(match[1]) as Partial<StageProgress> & { stage?: string };
      const stageId = String(payload.stage || "");
      const stage = stages.find((item) => item.id === stageId);
      if (!stage) {
        return true;
      }
      const percent = Math.max(0, Math.min(100, Number(payload.percent ?? 0)));
      const message = String(payload.message || "");
      setStageProgress((previous) => ({
        ...previous,
        [stageId]: { percent, message },
      }));
      setStageStates((previous) => ({
        ...previous,
        [stageId]: percent >= 100 ? "done" : "running",
      }));
      pushLog("info", `${stage.title}: ${message}${Number.isFinite(percent) ? ` (${Math.round(percent)}%)` : ""}`);
    } catch {
      pushLog("info", line);
    }
    return true;
  }

  function markStageFromText(text: string) {
    const marker = text.match(/\[pipeline\]\s+([0-9]+(?:\.[0-9]+)?)_/);
    const step = marker ? Number(marker[1]) : NaN;
    const nextIndex = Number.isFinite(step) ? stages.findIndex((stage) => stage.step === step) : -1;
    if (nextIndex < 0) {
      return;
    }
    setStageStates((previous) => {
      return Object.fromEntries(
        stages.map((stage, index) => {
          if (!isStageEnabled(stage, config.renderMode, config.pipelineMode)) {
            return [stage.id, "skipped"];
          }
          if (previous[stage.id] === "failed") {
            return [stage.id, "failed"];
          }
          if (index < nextIndex) {
            return [stage.id, "done"];
          }
          if (index === nextIndex) {
            return [stage.id, "running"];
          }
          return [stage.id, "waiting"];
        }),
      ) as Record<string, StageState>;
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

  function updateRenderTarget(target: "draft" | "video", checked: boolean) {
    setConfig((previous) => {
      const targets = renderTargetsFromMode(previous.renderMode);
      const nextTargets = {
        ...targets,
        [target === "draft" ? "outputDraft" : "outputVideo"]: checked,
      };
      return {
        ...previous,
        renderMode: renderModeFromTargets(nextTargets.outputDraft, nextTargets.outputVideo),
      };
    });
  }

  async function chooseVideo(key: "refVideoPath" | "moviePath", title: string) {
    const selected = await cloneBridge.selectFile({ title, filters: videoFilters });
    if (selected) {
      updateConfig(key, selected);
    }
  }

  async function chooseSubtitle(key: "subtitlePath" | "movieSubtitlePath", title = "选择字幕文件") {
    const selected = await cloneBridge.selectFile({ title, filters: subtitleFilters });
    if (selected) {
      updateConfig(key, selected);
    }
  }

  async function chooseDirectory(key: "outputRoot" | "jianyingDraftDir") {
    const selected = await cloneBridge.selectDirectory();
    if (selected) {
      updateConfig(key, selected);
      if (key === "jianyingDraftDir") {
        setDraftDetect({ status: "found", message: "已设置剪映草稿目录" });
      }
    }
  }

  async function detectDraftDir(manual: boolean) {
    setDraftDetect({ status: "detecting", message: "正在查找本机剪映草稿目录…" });
    try {
      const found = await cloneBridge.detectJianyingDraftDir();
      if (found) {
        updateConfig("jianyingDraftDir", found);
        setDraftDetect({ status: "found", message: manual ? "已重新定位剪映草稿目录" : "已自动定位剪映草稿目录" });
      } else {
        setDraftDetect({ status: "missing", message: "未找到剪映草稿目录，请手动选择保存文件夹。" });
      }
    } catch (error) {
      setDraftDetect({ status: "missing", message: error instanceof Error ? error.message : String(error) });
    }
  }

  async function startPipeline(runMode: PipelineRunMode = "normal") {
    if (!config.refVideoPath || !config.moviePath) {
      pushLog("error", "请先选择参考视频和原片。");
      setActiveTab("setup");
      return;
    }

    setActiveTab("pipeline");
    setResumeState(null);
    const result = await cloneBridge.startPipeline({ ...config, runMode });
    if (!result.ok) {
      pushLog("error", result.error || "启动失败。");
    } else if (result.logPath) {
      setPipelineLogPath(result.logPath);
    }
  }

  async function handleStartClick() {
    if (!config.refVideoPath || !config.moviePath) {
      await startPipeline();
      return;
    }

    const state = await cloneBridge.getPipelineState(config);
    if (state.canResume) {
      setResumeState({
        outputRoot: state.outputRoot,
        completedStages: state.completedStages,
      });
      return;
    }

    await startPipeline("normal");
  }

  async function stopPipeline() {
    await cloneBridge.stopPipeline();
  }

  async function testAiConnection() {
    setAiTest({ status: "testing", message: "正在测试连接…" });
    try {
      const result = await cloneBridge.testAi({
        aiApiKey: config.aiApiKey,
        aiBaseUrl: config.aiBaseUrl,
        aiModel: config.aiModel,
      });
      if (result.ok) {
        setAiTest({ status: "ok", message: `连接成功，模型 ${result.model || config.aiModel}` });
      } else {
        setAiTest({ status: "error", message: result.error || "连接失败" });
      }
    } catch (error) {
      setAiTest({ status: "error", message: error instanceof Error ? error.message : String(error) });
    }
  }

  const renderTargets = renderTargetsFromMode(config.renderMode);
  const activeStages = useMemo(
    () => stages.filter((stage) => isStageEnabled(stage, config.renderMode, config.pipelineMode)),
    [config.renderMode, config.pipelineMode],
  );
  const activeStageCount = activeStages.length;
  const completedCount = useMemo(
    () => activeStages.filter((stage) => stageStates[stage.id] === "done").length,
    [activeStages, stageStates],
  );
  const progressUnits = useMemo(
    () =>
      stages.reduce((sum, stage) => {
        if (!isStageEnabled(stage, config.renderMode, config.pipelineMode)) {
          return sum;
        }
        if (stageStates[stage.id] === "done") {
          return sum + 1;
        }
        if (stageStates[stage.id] === "running") {
          return sum + Math.max(0, Math.min(100, stageProgress[stage.id]?.percent ?? 0)) / 100;
        }
        return sum;
      }, 0),
    [config.renderMode, config.pipelineMode, stageProgress, stageStates],
  );
  const progress = Math.round((progressUnits / Math.max(1, activeStageCount)) * 100);
  const canStart = config.refVideoPath && config.moviePath && !isRunning;

  const videoOutputFullPath = outputPath(config.outputRoot, "outputs/8_generate_video");
  const jianyingDraftFullPath = config.jianyingDraftDir
    ? config.jianyingDraftDir
    : outputPath(config.outputRoot, "outputs/8_generate_video/jianying_drafts");
  const isRefAudioRebuild = config.pipelineMode === "ref_audio_rebuild";

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
            <span>{completedCount} / {activeStageCount} 阶段完成</span>
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
              <button className="primary-button" onClick={handleStartClick} disabled={!canStart}>
                <Play weight="fill" /> 开始生成
              </button>
            )}
          </div>
        </header>

        {resumeState && (
          <div className="resume-dialog-backdrop" role="presentation">
            <div className="resume-dialog" role="dialog" aria-modal="true" aria-labelledby="resume-dialog-title">
              <div>
                <p className="section-label">检测到未完成任务</p>
                <h2 id="resume-dialog-title">要继续生成还是重新生成？</h2>
                <p>
                  已在当前输出目录找到 {resumeState.completedStages.length} 个阶段产物。继续生成会复用已有产物，重新生成会清理阶段产物并从头开始。
                </p>
                <small>{resumeState.outputRoot}</small>
              </div>
              <div className="resume-dialog-actions">
                <button className="ghost-button" onClick={() => setResumeState(null)}>
                  取消
                </button>
                <button className="danger-button" onClick={() => startPipeline("restart")}>
                  <ArrowCounterClockwise /> 重新生成
                </button>
                <button className="primary-button" onClick={() => startPipeline("resume")}>
                  <Play weight="fill" /> 继续生成
                </button>
              </div>
            </div>
          </div>
        )}

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
                  onChoose={() => chooseSubtitle("subtitlePath", "选择参考字幕")}
                  onClear={() => updateConfig("subtitlePath", "")}
                />
                <AssetPicker
                  icon={<File />}
                  title="原片字幕"
                  value={config.movieSubtitlePath}
                  description="可选。提供 srt 可跳过原片 ASR，并用于原声判定。"
                  actionLabel="选择字幕"
                  onChoose={() => chooseSubtitle("movieSubtitlePath", "选择原片字幕")}
                  onClear={() => updateConfig("movieSubtitlePath", "")}
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
              <div className="render-options pipeline-mode-options">
                <label className={cx("check-option", config.pipelineMode === "clone" && "active")}>
                  <input
                    type="radio"
                    name="pipelineMode"
                    checked={config.pipelineMode === "clone"}
                    onChange={() => updateConfig("pipelineMode", "clone")}
                  />
                  <span>
                    <strong>仿写克隆</strong>
                    <small>重写文案并生成新解说视频</small>
                  </span>
                </label>
                <label className={cx("check-option", isRefAudioRebuild && "active")}>
                  <input
                    type="radio"
                    name="pipelineMode"
                    checked={isRefAudioRebuild}
                    onChange={() => updateConfig("pipelineMode", "ref_audio_rebuild")}
                  />
                  <span>
                    <strong>参考音频画面重组</strong>
                    <small>保留参考原声并替换为原片画面</small>
                  </span>
                </label>
              </div>
              <div className="render-options">
                <label className={cx("check-option", renderTargets.outputVideo && "active")}>
                  <input
                    type="checkbox"
                    checked={renderTargets.outputVideo}
                    onChange={(event) => updateRenderTarget("video", event.target.checked)}
                  />
                  <span>
                    <strong>直接输出视频</strong>
                    <small>生成最终 MP4 成片</small>
                  </span>
                </label>
                <label className={cx("check-option", renderTargets.outputDraft && "active")}>
                  <input
                    type="checkbox"
                    checked={renderTargets.outputDraft}
                    onChange={(event) => updateRenderTarget("draft", event.target.checked)}
                  />
                  <span>
                    <strong>输出剪映草稿</strong>
                    <small>需确保剪映草稿路径正确</small>
                  </span>
                </label>
              </div>
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

            <div className="panel full-panel">
              <div className="panel-heading">
                <div>
                  <p className="section-label">剪映草稿</p>
                  <h2>草稿保存目录</h2>
                </div>
                <button
                  className="ghost-button"
                  onClick={() => detectDraftDir(true)}
                  disabled={draftDetect.status === "detecting"}
                >
                  {draftDetect.status === "detecting" ? <CircleNotch className="spin" /> : <MagnifyingGlass weight="bold" />}
                  自动查找
                </button>
              </div>
              <p className="panel-hint">
                软件启动时会自动查找本机剪映 / CapCut 草稿目录并记住；找到后下次启动不再修改。若未找到，请手动选择要保存草稿的文件夹。
              </p>
              <div className="draft-dir-field">
                <div className="input-with-action">
                  <input
                    value={config.jianyingDraftDir}
                    placeholder="例如 C:\\Users\\你\\AppData\\Local\\JianyingPro\\User Data\\Projects\\com.lveditor.draft"
                    onChange={(event) => updateConfig("jianyingDraftDir", event.target.value)}
                  />
                  <button onClick={() => chooseDirectory("jianyingDraftDir")} title="选择文件夹">
                    <Folder />
                  </button>
                </div>
                {config.jianyingDraftDir && (
                  <button
                    className="ghost-button draft-clear"
                    onClick={() => {
                      updateConfig("jianyingDraftDir", "");
                      setDraftDetect({ status: "idle", message: "" });
                    }}
                  >
                    <Prohibit /> 清除
                  </button>
                )}
              </div>
              {draftDetect.status !== "idle" && (
                <p className={cx("draft-detect-status", draftDetect.status)}>
                  {draftDetect.status === "found" && <CheckCircle weight="fill" />}
                  {draftDetect.status === "missing" && <WarningCircle weight="fill" />}
                  {draftDetect.status === "detecting" && <CircleNotch className="spin" />}
                  <span>
                    {draftDetect.message}
                    {draftDetect.status === "found" && config.jianyingDraftDir ? `：${config.jianyingDraftDir}` : ""}
                  </span>
                </p>
              )}
            </div>

            {!isRefAudioRebuild && (
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
                <div className="ai-test-row">
                  <button className="ghost-button" onClick={testAiConnection} disabled={aiTest.status === "testing"}>
                    {aiTest.status === "testing" ? <CircleNotch className="spin" /> : <Lightning weight="fill" />} 测试连接
                  </button>
                  {aiTest.status !== "idle" && (
                    <p className={cx("ai-test-status", aiTest.status)}>
                      {aiTest.status === "ok" && <CheckCircle weight="fill" />}
                      {aiTest.status === "error" && <WarningCircle weight="fill" />}
                      <span>{aiTest.message}</span>
                    </p>
                  )}
                </div>
              </div>
            </div>
            )}

            <div className="panel wide-panel">
              <div className="panel-heading">
                <div>
                  <p className="section-label">高级参数</p>
                  <h2>{isRefAudioRebuild ? "性能与镜头识别" : "性能、镜头识别与配音"}</h2>
                </div>
              </div>
              <div className="settings-grid">
                <SelectField label="镜头后端" value={config.backend} onChange={(value) => updateConfig("backend", value as BackendMode)} options={["auto", "transnet", "opencv"]} />
                <SelectField label="视频编码" value={config.videoEncoder} onChange={(value) => updateConfig("videoEncoder", value as VideoEncoder)} options={["auto", "libx264", "h264_nvenc"]} />
                <NumberField label="镜头阈值" value={config.threshold} min={0.1} max={0.9} step={0.05} onChange={(value) => updateConfig("threshold", value)} />
                {!isRefAudioRebuild && (
                  <>
                <NumberField label="AI 温度" value={config.aiTemperature} min={0} max={1.5} step={0.1} onChange={(value) => updateConfig("aiTemperature", value)} />
                <NumberField label="每秒字数" value={config.charsPerSecond} min={2} max={8} step={0.1} onChange={(value) => updateConfig("charsPerSecond", value)} />
                <NumberField label="配音速度" value={config.edgeTtsSpeed} min={0.5} max={2} step={0.05} onChange={(value) => updateConfig("edgeTtsSpeed", value)} />
                <label>
                  Edge Voice ID
                  <input value={config.edgeVoiceId} onChange={(event) => updateConfig("edgeVoiceId", event.target.value)} />
                </label>
                  </>
                )}
              </div>
            </div>
          </section>
        )}

        {activeTab === "pipeline" && (
          <section className="pipeline-layout">
            <div className="stage-list">
              {activeStages.map((stage) => (
                <StageRow
                  key={stage.id}
                  stage={stage}
                  state={stageStates[stage.id]}
                  progress={stageProgress[stage.id]}
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
                <div className="log-actions">
                  <button title="打开日志" onClick={() => cloneBridge.openPath(pipelineLogPath)} disabled={!pipelineLogPath}>
                    <File />
                  </button>
                  <button title="定位日志" onClick={() => cloneBridge.revealPath(pipelineLogPath)} disabled={!pipelineLogPath}>
                    <Folder />
                  </button>
                  <TerminalWindow />
                </div>
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

            <button
              className="output-tile"
              onClick={() => cloneBridge.openPath(videoOutputFullPath)}
            >
              <span>视频</span>
              <strong>直出视频</strong>
              <small>{videoOutputFullPath}</small>
            </button>

            {(config.renderMode === "draft" || config.renderMode === "both") && (
              <button
                className="output-tile"
                onClick={() => cloneBridge.openPath(jianyingDraftFullPath)}
              >
                <span>草稿</span>
                <strong>剪映草稿</strong>
                <small>{jianyingDraftFullPath}</small>
              </button>
            )}
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

function StageRow({
  stage,
  state,
  progress,
  targetPath,
}: {
  stage: Stage;
  state: StageState;
  progress?: StageProgress;
  targetPath: string;
}) {
  const icon = {
    waiting: <ClockCounterClockwise />,
    running: <CircleNotch className="spin" />,
    done: <CheckCircle weight="fill" />,
    failed: <WarningCircle weight="fill" />,
    skipped: <Prohibit />,
  }[state];
  const percent = Math.round(Math.max(0, Math.min(100, progress?.percent ?? (state === "done" ? 100 : 0))));
  const showProgress = state === "running" || (progress && percent > 0 && state !== "skipped");

  return (
    <article className={cx("stage-row", state)}>
      <div className="stage-index">{String(stage.step).padStart(2, "0")}</div>
      <div className="stage-main">
        <div className="stage-copy">
          <strong>{stage.title}</strong>
          <span>{stage.detail}</span>
          {showProgress && (
            <div className="stage-progress">
              <div className="stage-progress-meta">
                <small>{progress?.message || stateText(state)}</small>
                <small>{percent}%</small>
              </div>
              <div className="stage-progress-track">
                <i style={{ width: `${percent}%` }} />
              </div>
            </div>
          )}
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
  if (state === "skipped") {
    return "已跳过";
  }
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
