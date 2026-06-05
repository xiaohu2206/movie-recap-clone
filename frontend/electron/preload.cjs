const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("cloneApp", {
  selectFile: (options) => ipcRenderer.invoke("dialog:select-file", options),
  selectDirectory: () => ipcRenderer.invoke("dialog:select-directory"),
  detectJianyingDraftDir: () => ipcRenderer.invoke("jianying:detect-draft-dir"),
  loadConfig: () => ipcRenderer.invoke("config:load"),
  saveConfig: (config) => ipcRenderer.invoke("config:save", config),
  testAi: (config) => ipcRenderer.invoke("ai:test", config),
  startPipeline: (config) => ipcRenderer.invoke("pipeline:start", config),
  getPipelineState: (config) => ipcRenderer.invoke("pipeline:state", config),
  stopPipeline: () => ipcRenderer.invoke("pipeline:stop"),
  revealPath: (targetPath) => ipcRenderer.invoke("shell:reveal-path", targetPath),
  openPath: (targetPath) => ipcRenderer.invoke("shell:open-path", targetPath),
  getBackendInfo: () => ipcRenderer.invoke("backend:info"),
  onPipelineEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("pipeline:event", listener);
    return () => ipcRenderer.removeListener("pipeline:event", listener);
  },
});
