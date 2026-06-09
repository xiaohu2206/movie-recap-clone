/// <reference types="vite/client" />

export type DialogFileOptions = {
  title?: string;
  filters?: Array<{ name: string; extensions: string[] }>;
};

export type PipelineEvent =
  | {
      type: "started";
      at: string;
      command: string;
      outputRoot: string;
      logPath?: string;
      runMode?: string;
      completedStages?: string[];
    }
  | { type: "stdout"; text: string }
  | { type: "stderr"; text: string }
  | { type: "error"; error: string; at: string }
  | { type: "finished"; code: number | null; outputRoot: string; at: string }
  | { type: "stopped"; at: string };

export type BackendInfo = {
  root: string;
  defaultOutputRoot: string;
  python: string;
  packaged: boolean;
  hasBundledPython: boolean;
  hasLocalVenv: boolean;
};

export type PipelineState = {
  ok: boolean;
  outputRoot: string;
  logPath?: string;
  completed: boolean;
  canResume: boolean;
  configChanged?: boolean;
  completedStages: string[];
};

declare global {
  interface Window {
    cloneApp: {
      selectFile: (options?: DialogFileOptions) => Promise<string>;
      selectDirectory: () => Promise<string>;
      detectJianyingDraftDir: () => Promise<string>;
      loadConfig: () => Promise<Record<string, unknown> | null>;
      saveConfig: (config: unknown) => Promise<{ ok: boolean; error?: string }>;
      testAi: (config: unknown) => Promise<{ ok: boolean; model?: string; error?: string }>;
      startPipeline: (config: unknown) => Promise<{ ok: boolean; error?: string; outputRoot?: string; logPath?: string }>;
      getPipelineState: (config: unknown) => Promise<PipelineState>;
      stopPipeline: () => Promise<{ ok: boolean }>;
      revealPath: (targetPath: string) => Promise<boolean>;
      openPath: (targetPath: string) => Promise<boolean>;
      getBackendInfo: () => Promise<BackendInfo>;
      onPipelineEvent: (callback: (event: PipelineEvent) => void) => () => void;
    };
  }
}
