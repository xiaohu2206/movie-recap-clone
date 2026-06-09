const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

let mainWindow = null;
let activeProcess = null;
let activeOutputRoot = "";

const isDev = !app.isPackaged;

function logMain(message) {
  try {
    const logPath = path.join(app.getPath("userData"), "main.log");
    fs.appendFileSync(logPath, `[${new Date().toISOString()}] ${message}\n`, "utf8");
  } catch {
    // Logging must never block application startup.
  }
}

process.on("uncaughtException", (error) => {
  logMain(`uncaughtException: ${error.stack || error.message}`);
});

process.on("unhandledRejection", (error) => {
  logMain(`unhandledRejection: ${error && error.stack ? error.stack : String(error)}`);
});

function projectRoot() {
  if (isDev) {
    return path.resolve(__dirname, "..", "..");
  }
  return path.join(process.resourcesPath, "backend");
}

function rendererUrl() {
  return "http://127.0.0.1:5173";
}

function bundledPython(root) {
  const portablePython = path.join(root, "python", "python.exe");
  if (fs.existsSync(portablePython)) {
    return portablePython;
  }
  const venvPython = path.join(root, ".venv", "Scripts", "python.exe");
  if (fs.existsSync(venvPython)) {
    return venvPython;
  }
  return "";
}

function hasBundledPython(root) {
  return Boolean(bundledPython(root));
}

function candidatePython(root) {
  const resolved = bundledPython(root);
  if (resolved) {
    return resolved;
  }
  if (app.isPackaged) {
    logMain("bundled python missing under backend/python or backend/.venv");
  }
  return process.platform === "win32" ? "python" : "python3";
}

function bundledFfmpegDir(root) {
  const binDir = path.join(root, "ffmpeg", "bin");
  if (fs.existsSync(path.join(binDir, "ffmpeg.exe")) && fs.existsSync(path.join(binDir, "ffprobe.exe"))) {
    return binDir;
  }
  return "";
}

function pythonRuntimeEnv(root, extra = {}) {
  const env = {
    ...process.env,
    PYTHONIOENCODING: "utf-8",
    ...extra,
  };
  const ffmpegDir = bundledFfmpegDir(root);
  if (ffmpegDir) {
    env.FFMPEG_DIR = ffmpegDir;
    env.PATH = env.PATH ? `${ffmpegDir}${path.delimiter}${env.PATH}` : ffmpegDir;
  }
  if (process.platform !== "win32") {
    return env;
  }
  const torchLibDirs = [
    path.join(root, "python", "Lib", "site-packages", "torch", "lib"),
    path.join(root, ".venv", "Lib", "site-packages", "torch", "lib"),
  ].filter((item) => fs.existsSync(item));
  if (torchLibDirs.length) {
    const prefix = [...new Set(torchLibDirs)].join(path.delimiter);
    env.PATH = env.PATH ? `${prefix}${path.delimiter}${env.PATH}` : prefix;
  }
  return env;
}

function sanitizeStoredPath(value) {
  if (!value || typeof value !== "string") {
    return "";
  }
  let fixed = value.replace(/\r/g, "");
  fixed = fixed.replace(/([A-Za-z]:)esources/g, "$1\\resources");
  return path.normalize(fixed);
}

function sanitizePipelineConfig(config = {}) {
  const next = { ...config };
  for (const key of ["refVideoPath", "moviePath", "subtitlePath", "outputRoot", "jianyingDraftDir"]) {
    if (typeof next[key] === "string" && next[key]) {
      next[key] = sanitizeStoredPath(next[key]);
    }
  }
  if (!next.outputRoot) {
    next.outputRoot = path.join(projectRoot(), "outputs");
  }
  return next;
}

function jianyingDraftCandidates() {
  const candidates = [];
  if (process.platform === "win32") {
    const localAppData =
      process.env.LOCALAPPDATA ||
      (process.env.USERPROFILE ? path.join(process.env.USERPROFILE, "AppData", "Local") : "");
    if (localAppData) {
      candidates.push(path.join(localAppData, "JianyingPro", "User Data", "Projects", "com.lveditor.draft"));
      candidates.push(path.join(localAppData, "CapCut", "User Data", "Projects", "com.lveditor.draft"));
    }
  } else if (process.platform === "darwin") {
    const home = process.env.HOME || app.getPath("home");
    if (home) {
      candidates.push(path.join(home, "Movies", "JianyingPro", "User Data", "Projects", "com.lveditor.draft"));
      candidates.push(path.join(home, "Movies", "CapCut", "User Data", "Projects", "com.lveditor.draft"));
    }
  }
  return candidates;
}

function detectJianyingDraftDir() {
  for (const candidate of jianyingDraftCandidates()) {
    try {
      if (fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()) {
        return candidate;
      }
    } catch {
      // Ignore inaccessible candidates and keep probing.
    }
  }
  return "";
}

const stageOutputs = [
  ["reference", "1_reference_analyzer", "ref_analysis.json"],
  ["segments", "2_narration_segmenter", "narration_segments.json"],
  ["shots", "3_movie_shot_parser", "movie_shots.json"],
  ["alignment", "4_visual_alignment_engine", "ref_to_movie_timeline.json"],
  ["binder", "5_script_visual_binder", "script_mapping.json"],
  ["subtitle", "5.1_movie_subtitle_filler", "script_mapping_subtitled.json"],
  ["audio_role", "5.2_audio_role_classifier", "script_mapping_with_audio.json"],
  ["rewrite", "6_rewrite_engine", "rewritten_script.json"],
  ["timeline", "7_timeline_composer", "final_timeline.json"],
  ["render", "8_generate_video", "generate_video_result.json"],
];

function hasValidJson(filePath) {
  try {
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      return false;
    }
    JSON.parse(fs.readFileSync(filePath, "utf8"));
    return true;
  } catch {
    return false;
  }
}

function pipelineStatePath(outputRoot) {
  return path.join(outputRoot, ".pipeline_state.json");
}

function pipelineLogPath(outputRoot) {
  return path.join(outputRoot, "logs", "pipeline.log");
}

function appendPipelineLog(logPath, level, text) {
  if (!logPath || !text) {
    return;
  }
  try {
    fs.mkdirSync(path.dirname(logPath), { recursive: true });
    const lines = String(text).split(/\r?\n/).filter(Boolean);
    const stamp = new Date().toISOString();
    const payload = lines.map((line) => `[${stamp}] [${level}] ${line}`).join("\n");
    if (payload) {
      fs.appendFileSync(logPath, `${payload}\n`, "utf8");
    }
  } catch (error) {
    logMain(`appendPipelineLog failed: ${error.message}`);
  }
}

function configSignature(config = {}) {
  return JSON.stringify({
    refVideoPath: config.refVideoPath || "",
    moviePath: config.moviePath || "",
    subtitlePath: config.subtitlePath || "",
    movieSubtitlePath: config.movieSubtitlePath || "",
    outputRoot: config.outputRoot || "",
    asrProvider: config.subtitlePath ? "none" : config.asrProvider,
    threshold: config.threshold,
    backend: config.backend,
    aiProvider: "custom_openai",
    aiBaseUrl: config.aiBaseUrl || "",
    aiModel: config.aiModel || "",
    aiTemperature: config.aiTemperature,
    charsPerSecond: config.charsPerSecond,
    renderMode: config.renderMode,
    edgeVoiceId: config.edgeVoiceId,
    edgeTtsSpeed: config.edgeTtsSpeed,
    jianyingDraftDir: config.jianyingDraftDir || "",
    videoOutputName: config.videoOutputName || "",
    videoEncoder: config.videoEncoder,
  });
}

function readPipelineMemory(outputRoot) {
  try {
    return JSON.parse(fs.readFileSync(pipelineStatePath(outputRoot), "utf8"));
  } catch {
    return null;
  }
}

function writePipelineMemory(outputRoot, payload) {
  try {
    fs.mkdirSync(outputRoot, { recursive: true });
    const previous = readPipelineMemory(outputRoot) || {};
    fs.writeFileSync(
      pipelineStatePath(outputRoot),
      JSON.stringify({ ...previous, ...payload, updatedAt: new Date().toISOString() }, null, 2),
      "utf8",
    );
  } catch (error) {
    logMain(`writePipelineMemory failed: ${error.message}`);
  }
}

function hasCompletedFinalOutput(outputRoot, renderMode) {
  if (renderMode === "none") {
    return hasValidJson(path.join(outputRoot, "7_timeline_composer", "final_timeline.json"));
  }

  const manifestPath = path.join(outputRoot, "8_generate_video", "generate_video_result.json");
  if (!hasValidJson(manifestPath)) {
    return false;
  }

  try {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    const mode = manifest.mode || renderMode;
    if (mode === "video" || mode === "both") {
      const videoPath = manifest.rendered_video?.output_video || "outputs\\8_generate_video\\clone_narration_output.mp4";
      const absVideoPath = path.isAbsolute(videoPath)
        ? videoPath
        : path.join(projectRoot(), videoPath.replace(/^outputs[\\/]/, "outputs\\"));
      if (!fs.existsSync(absVideoPath) || fs.statSync(absVideoPath).size <= 0) {
        return false;
      }
    }
    if (mode === "draft" || mode === "both") {
      const draftDir = manifest.jianying_draft?.draft_dir;
      const absDraftDir = draftDir
        ? path.isAbsolute(draftDir)
          ? draftDir
          : path.join(projectRoot(), draftDir.replace(/^outputs[\\/]/, "outputs\\"))
        : "";
      if (!absDraftDir || !fs.existsSync(absDraftDir)) {
        return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}

function sendPipelineEvent(payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("pipeline:event", payload);
  }
}

function createWindow() {
  logMain(`createWindow packaged=${app.isPackaged} dirname=${__dirname}`);
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 940,
    minWidth: 1180,
    minHeight: 760,
    title: "Clone Narration Studio",
    backgroundColor: "#f4f3ef",
    titleBarStyle: "hiddenInset",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  if (isDev) {
    mainWindow.loadURL(rendererUrl());
  } else {
    mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (activeProcess) {
    activeProcess.kill();
  }
});

ipcMain.handle("dialog:select-file", async (_event, options = {}) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: options.title || "选择文件",
    properties: ["openFile"],
    filters: options.filters || [{ name: "All files", extensions: ["*"] }],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return "";
  }
  return result.filePaths[0];
});

ipcMain.handle("jianying:detect-draft-dir", async () => {
  const found = detectJianyingDraftDir();
  logMain(`detect jianying draft dir -> ${found || "(not found)"}`);
  return found;
});

ipcMain.handle("dialog:select-directory", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "选择目录",
    properties: ["openDirectory", "createDirectory"],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return "";
  }
  return result.filePaths[0];
});

function userConfigPath() {
  return path.join(app.getPath("userData"), "studio-config.json");
}

function normalizeChatUrl(baseUrl) {
  let bu = String(baseUrl || "https://api.openai.com/v1").trim().replace(/`/g, "").replace(/\/+$/, "");
  if (!bu.endsWith("/chat/completions")) {
    bu += "/chat/completions";
  }
  return bu;
}

ipcMain.handle("config:load", async () => {
  try {
    const config = JSON.parse(fs.readFileSync(userConfigPath(), "utf8"));
    return sanitizePipelineConfig(config);
  } catch {
    return null;
  }
});

ipcMain.handle("config:save", async (_event, config = {}) => {
  try {
    fs.writeFileSync(userConfigPath(), JSON.stringify(sanitizePipelineConfig(config), null, 2), "utf8");
    return { ok: true };
  } catch (error) {
    logMain(`config:save failed: ${error.message}`);
    return { ok: false, error: error.message };
  }
});

ipcMain.handle("ai:test", async (_event, config = {}) => {
  const apiKey = String(config.aiApiKey || process.env.CLONE_AI_API_KEY || process.env.OPENAI_API_KEY || "").trim();
  const model = String(config.aiModel || process.env.CLONE_AI_MODEL || process.env.OPENAI_MODEL || "gpt-4o-mini").trim();
  const url = normalizeChatUrl(config.aiBaseUrl || process.env.CLONE_AI_BASE_URL || process.env.OPENAI_BASE_URL || "https://api.openai.com/v1");
  if (!apiKey) {
    return { ok: false, error: "未填写 API Key，且环境变量未提供。" };
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "ping" }],
        temperature: 0,
        max_tokens: 1,
        stream: false,
      }),
      signal: controller.signal,
    });
    const raw = await response.text();
    if (!response.ok) {
      let detail = raw;
      try {
        const parsed = JSON.parse(raw);
        detail = parsed.error?.message || parsed.message || raw;
      } catch {
        // keep raw text as detail
      }
      return { ok: false, error: `HTTP ${response.status}: ${String(detail).slice(0, 300)}` };
    }
    let resolvedModel = model;
    try {
      resolvedModel = JSON.parse(raw).model || model;
    } catch {
      // response not JSON; still treat as reachable
    }
    return { ok: true, model: resolvedModel };
  } catch (error) {
    const message = error.name === "AbortError" ? "请求超时（20 秒）。" : error.message || String(error);
    return { ok: false, error: message };
  } finally {
    clearTimeout(timer);
  }
});

ipcMain.handle("backend:info", async () => {
  const root = projectRoot();
  return {
    root,
    defaultOutputRoot: path.join(root, "outputs"),
    python: candidatePython(root),
    packaged: app.isPackaged,
    hasBundledPython: fs.existsSync(path.join(root, "python", "python.exe")),
    hasLocalVenv: fs.existsSync(path.join(root, ".venv", "Scripts", "python.exe")),
  };
});

ipcMain.handle("pipeline:state", async (_event, config = {}) => {
  const root = projectRoot();
  const normalized = sanitizePipelineConfig(config);
  const outputRoot = normalized.outputRoot || path.join(root, "outputs");
  const logPath = pipelineLogPath(outputRoot);
  const completedStages = stageOutputs
    .filter(([, dir, file]) => hasValidJson(path.join(outputRoot, dir, file)))
    .map(([id]) => id);
  const finalStage = normalized.renderMode === "none" ? "timeline" : "render";
  const finalOutputComplete = completedStages.includes(finalStage) && hasCompletedFinalOutput(outputRoot, normalized.renderMode);
  const memory = readPipelineMemory(outputRoot);
  const sameProject = !memory?.configSignature || memory.configSignature === configSignature({ ...normalized, outputRoot });
  const completed = finalOutputComplete && (!memory || memory.status === "completed");

  return {
    ok: true,
    outputRoot,
    logPath,
    completed,
    canResume: sameProject && completedStages.length > 0 && !completed,
    completedStages,
  };
});

ipcMain.handle("shell:reveal-path", async (_event, targetPath) => {
  if (!targetPath) {
    return false;
  }
  shell.showItemInFolder(targetPath);
  return true;
});

ipcMain.handle("shell:open-path", async (_event, targetPath) => {
  if (!targetPath) {
    return false;
  }
  const error = await shell.openPath(targetPath);
  return !error;
});

ipcMain.handle("pipeline:stop", async () => {
  if (!activeProcess) {
    return { ok: true };
  }
  activeProcess.kill();
  activeProcess = null;
  if (activeOutputRoot) {
    writePipelineMemory(activeOutputRoot, { status: "stopped" });
    activeOutputRoot = "";
  }
  sendPipelineEvent({ type: "stopped", at: new Date().toISOString() });
  return { ok: true };
});

ipcMain.handle("pipeline:start", async (_event, rawConfig) => {
  if (activeProcess) {
    return { ok: false, error: "已有任务正在运行" };
  }

  const config = sanitizePipelineConfig(rawConfig);
  const root = projectRoot();
  const mainScript = path.join(root, "main.py");
  if (!fs.existsSync(mainScript)) {
    return { ok: false, error: `找不到后端入口: ${mainScript}` };
  }
  if (app.isPackaged && !hasBundledPython(root)) {
    return {
      ok: false,
      error:
        "安装包未包含 Python 运行环境（backend/python）。请重新执行 cnpm run pack:dir 打包完整安装包。",
    };
  }

  const outputRoot = config.outputRoot || path.join(root, "outputs");
  const logPath = pipelineLogPath(outputRoot);
  const renderMode = config.renderMode || "draft";
  if ((renderMode === "draft" || renderMode === "both") && config.jianyingDraftDir && !fs.existsSync(config.jianyingDraftDir)) {
    return { ok: false, error: `剪映草稿路径不存在: ${config.jianyingDraftDir}` };
  }
  const aiApiKey = String(config.aiApiKey || process.env.CLONE_AI_API_KEY || process.env.OPENAI_API_KEY || "").trim();
  const aiBaseUrl = String(config.aiBaseUrl || process.env.CLONE_AI_BASE_URL || process.env.OPENAI_BASE_URL || "").trim();
  const aiModel = String(config.aiModel || process.env.CLONE_AI_MODEL || process.env.OPENAI_MODEL || "").trim();
  const args = [
    mainScript,
    "--ref-video-path",
    config.refVideoPath,
    "--movie-path",
    config.moviePath,
    "--output-root",
    outputRoot,
    "--log-file",
    logPath,
    "--asr-provider",
    config.subtitlePath ? "none" : config.asrProvider,
    "--threshold",
    String(config.threshold),
    "--backend",
    config.backend,
    "--ai-provider",
    "custom_openai",
    "--ai-temperature",
    String(config.aiTemperature),
    "--chars-per-second",
    String(config.charsPerSecond),
    "--render-mode",
    renderMode,
    "--edge-voice-id",
    config.edgeVoiceId,
    "--edge-tts-speed",
    String(config.edgeTtsSpeed),
    "--video-output-name",
    config.videoOutputName,
    "--video-encoder",
    config.videoEncoder,
  ];

  if (config.subtitlePath) {
    args.push("--subtitle-srt", config.subtitlePath);
  }
  if (config.movieSubtitlePath) {
    args.push("--movie-subtitle-srt", config.movieSubtitlePath);
  }
  if (aiBaseUrl) {
    args.push("--ai-base-url", aiBaseUrl);
  }
  if (aiModel) {
    args.push("--ai-model", aiModel);
  }
  if (config.jianyingDraftDir) {
    args.push("--jianying-draft-dir", config.jianyingDraftDir);
  }
  if (config.runMode === "resume") {
    args.push("--resume");
  }
  if (config.runMode === "restart") {
    args.push("--restart");
  }

  fs.mkdirSync(outputRoot, { recursive: true });
  fs.mkdirSync(path.dirname(logPath), { recursive: true });
  fs.writeFileSync(logPath, `[${new Date().toISOString()}] [system] pipeline log created\n`, "utf8");
  writePipelineMemory(outputRoot, {
    status: "running",
    runMode: config.runMode || "normal",
    configSignature: configSignature({ ...config, outputRoot, renderMode }),
    completedStages: [],
    exitCode: null,
    error: "",
  });
  activeOutputRoot = outputRoot;

  const python = candidatePython(root);
  activeProcess = spawn(python, args, {
    cwd: root,
    env: pythonRuntimeEnv(root, {
      CLONE_AI_API_KEY: aiApiKey,
      CLONE_AI_BASE_URL: aiBaseUrl,
      CLONE_AI_MODEL: aiModel,
    }),
    windowsHide: true,
  });

  sendPipelineEvent({
    type: "started",
    at: new Date().toISOString(),
    command: `${python} ${args.map((item) => (item.includes(" ") ? `"${item}"` : item)).join(" ")}`,
    outputRoot,
    logPath,
  });
  appendPipelineLog(logPath, "system", `${python} ${args.join(" ")}`);

  activeProcess.stdout.on("data", (chunk) => {
    const text = chunk.toString("utf8");
    sendPipelineEvent({ type: "stdout", text });
  });

  activeProcess.stderr.on("data", (chunk) => {
    const text = chunk.toString("utf8");
    sendPipelineEvent({ type: "stderr", text });
  });

  activeProcess.on("error", (error) => {
    activeProcess = null;
    if (activeOutputRoot) {
      writePipelineMemory(activeOutputRoot, { status: "error", error: error.message });
      activeOutputRoot = "";
    }
    appendPipelineLog(logPath, "error", error.message);
    sendPipelineEvent({ type: "error", error: error.message, at: new Date().toISOString() });
  });

  activeProcess.on("close", (code) => {
    activeProcess = null;
    writePipelineMemory(outputRoot, {
      status: code === 0 ? "completed" : "failed",
      exitCode: code,
      completedStages: stageOutputs
        .filter(([, dir, file]) => hasValidJson(path.join(outputRoot, dir, file)))
        .map(([id]) => id),
    });
    activeOutputRoot = "";
    appendPipelineLog(logPath, "system", `pipeline finished with code ${code}`);
    sendPipelineEvent({ type: "finished", code, outputRoot, at: new Date().toISOString() });
  });

  return { ok: true, outputRoot, logPath };
});
