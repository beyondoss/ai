/** Minimal Bun types so we can `bun compile` serve-http without @types/bun. */
declare namespace Bun {
  function serve(options: {
    hostname?: string;
    port?: number;
    fetch: (req: Request) => Response | Promise<Response>;
  }): {
    readonly port: number;
    stop: (closeActiveConnections?: boolean) => void | Promise<void>;
  };
}

declare const Bun: typeof Bun;
