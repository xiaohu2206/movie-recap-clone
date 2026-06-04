const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

let mainWindow = null;
let activeProcess = null;

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

function candidatePython(root) {
  const localPython = path.join(root, ".venv", "Scripts", "python.exe");
  if (fs.existsSync(localPython)) {
    return localPython;
  }
  return process.platform === "win32" ? "python" : "python3";
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

ipcMain.handle("backend:info", async () => {
  const root = projectRoot();
  return {
    root,
    python: candidatePython(root),
    packaged: app.isPackaged,
    hasLocalVenv: fs.existsSync(path.join(root, ".venv", "Scripts", "python.exe")),
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
  sendPipelineEvent({ type: "stopped", at: new Date().toISOString() });
  return { ok: true };
});

ipcMain.handle("pipeline:start", async (_event, config) => {
  if (activeProcess) {
    return { ok: false, error: "已有任务正在运行" };
  }

  const root = projectRoot();
  const mainScript = path.join(root, "main.py");
  if (!fs.existsSync(mainScript)) {
    return { ok: false, error: `找不到后端入口: ${mainScript}` };
  }

  const outputRoot = config.outputRoot || path.join(root, "outputs");
  const args = [
    mainScript,
    "--ref-video-path",
    config.refVideoPath,
    "--movie-path",
    config.moviePath,
    "--output-root",
    outputRoot,
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
    config.renderMode,
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
  if (config.aiApiKey) {
    args.push("--ai-api-key", config.aiApiKey);
  }
  if (config.aiBaseUrl) {
    args.push("--ai-base-url", config.aiBaseUrl);
  }
  if (config.aiModel) {
    args.push("--ai-model", config.aiModel);
  }
  if (config.jianyingDraftDir) {
    args.push("--jianying-draft-dir", config.jianyingDraftDir);
  }

  fs.mkdirSync(outputRoot, { recursive: true });

  const python = candidatePython(root);
  activeProcess = spawn(python, args, {
    cwd: root,
    env: {
      ...process.env,
      PYTHONIOENCODING: "utf-8",
      CLONE_AI_API_KEY: config.aiApiKey || process.env.CLONE_AI_API_KEY || "",
      CLONE_AI_BASE_URL: config.aiBaseUrl || process.env.CLONE_AI_BASE_URL || "",
      CLONE_AI_MODEL: config.aiModel || process.env.CLONE_AI_MODEL || "",
    },
    windowsHide: true,
  });

  sendPipelineEvent({
    type: "started",
    at: new Date().toISOString(),
    command: `${python} ${args.map((item) => (item.includes(" ") ? `"${item}"` : item)).join(" ")}`,
    outputRoot,
  });

  activeProcess.stdout.on("data", (chunk) => {
    sendPipelineEvent({ type: "stdout", text: chunk.toString("utf8") });
  });

  activeProcess.stderr.on("data", (chunk) => {
    sendPipelineEvent({ type: "stderr", text: chunk.toString("utf8") });
  });

  activeProcess.on("error", (error) => {
    activeProcess = null;
    sendPipelineEvent({ type: "error", error: error.message, at: new Date().toISOString() });
  });

  activeProcess.on("close", (code) => {
    activeProcess = null;
    sendPipelineEvent({ type: "finished", code, outputRoot, at: new Date().toISOString() });
  });

  return { ok: true, outputRoot };
});
