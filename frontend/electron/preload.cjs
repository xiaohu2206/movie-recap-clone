const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("cloneApp", {
  selectFile: (options) => ipcRenderer.invoke("dialog:select-file", options),
  selectDirectory: () => ipcRenderer.invoke("dialog:select-directory"),
  startPipeline: (config) => ipcRenderer.invoke("pipeline:start", config),
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
