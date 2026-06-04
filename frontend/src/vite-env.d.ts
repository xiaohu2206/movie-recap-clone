/// <reference types="vite/client" />

export type DialogFileOptions = {
  title?: string;
  filters?: Array<{ name: string; extensions: string[] }>;
};

export type PipelineEvent =
  | { type: "started"; at: string; command: string; outputRoot: string }
  | { type: "stdout"; text: string }
  | { type: "stderr"; text: string }
  | { type: "error"; error: string; at: string }
  | { type: "finished"; code: number | null; outputRoot: string; at: string }
  | { type: "stopped"; at: string };

export type BackendInfo = {
  root: string;
  python: string;
  packaged: boolean;
  hasLocalVenv: boolean;
};

declare global {
  interface Window {
    cloneApp: {
      selectFile: (options?: DialogFileOptions) => Promise<string>;
      selectDirectory: () => Promise<string>;
      startPipeline: (config: unknown) => Promise<{ ok: boolean; error?: string; outputRoot?: string }>;
      stopPipeline: () => Promise<{ ok: boolean }>;
      revealPath: (targetPath: string) => Promise<boolean>;
      openPath: (targetPath: string) => Promise<boolean>;
      getBackendInfo: () => Promise<BackendInfo>;
      onPipelineEvent: (callback: (event: PipelineEvent) => void) => () => void;
    };
  }
}
