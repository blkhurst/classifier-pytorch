export function SiteHeader({ connected }: { connected: boolean }) {
  return (
    <header className="border-b border-border">
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-6 py-6 sm:px-8">
        <h1 className="font-mono text-sm font-semibold tracking-tight sm:text-base">
          classifier-pytorch
        </h1>
        <span
          role="status"
          className="flex items-center gap-2 text-xs text-copy-secondary"
        >
          <span
            aria-hidden="true"
            className={`size-1.5 rounded-full ${connected ? "bg-success" : "bg-copy-muted"}`}
          />
          {connected ? "API connected" : "API disconnected"}
        </span>
      </div>
    </header>
  );
}
