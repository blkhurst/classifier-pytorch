import { useEffect, useRef, useState } from "react";
import {
  ReactSketchCanvas,
  type ReactSketchCanvasRef,
} from "react-sketch-canvas";
import { Button } from "@/components/button";
import { PixelPreview } from "@/features/prediction/pixel-preview";
import type { Prediction, PredictionInput } from "@/lib/types";

type DrawingPadProps = {
  input: PredictionInput;
  result: Prediction | null;
  connected: boolean;
  onInput: (input: PredictionInput) => void;
};

export function DrawingPad({
  input,
  result,
  connected,
  onInput,
}: DrawingPadProps) {
  const canvas = useRef<ReactSketchCanvasRef>(null);
  const revision = useRef(0);
  const [error, setError] = useState<string | null>(null);
  const [hasInk, setHasInk] = useState(false);
  const sample = input?.kind === "sample";

  useEffect(
    () => () => {
      revision.current++;
    },
    [],
  );

  async function readDrawing() {
    const current = ++revision.current;
    setError(null);
    try {
      // A Fixed Export Size Also Works On High-DPI Screens.
      const image = await canvas.current?.exportImage("png", {
        width: 280,
        height: 280,
      });
      if (image && current === revision.current)
        onInput({ kind: "drawing", image });
    } catch {
      if (current === revision.current)
        setError("Could not read the drawing. Try clearing it.");
    }
  }

  function clear() {
    revision.current++;
    canvas.current?.clearCanvas();
    setHasInk(false);
    setError(null);
    onInput(null);
  }

  function trySample() {
    clear();
    onInput({ kind: "sample", index: Math.floor(Math.random() * 60_000) });
  }

  return (
    <div>
      <h2 className="section-title">Draw a digit</h2>
      <div
        className="relative mt-6 aspect-square w-full max-w-70 touch-none overflow-hidden rounded-lg bg-canvas"
        onPointerDownCapture={() => {
          revision.current++;
          setHasInk(true);
          if (sample) onInput(null);
        }}
      >
        {/* Fixed image colours, independent of the interface theme. */}
        <ReactSketchCanvas
          ref={canvas}
          width="100%"
          height="100%"
          strokeWidth={18}
          strokeColor="white"
          canvasColor="black"
          style={{ border: "none", borderRadius: 0 }}
          onStroke={() => void readDrawing()}
        />
        {!hasInk && !sample && (
          <span
            className="pointer-events-none absolute inset-0 grid place-items-center text-5xl font-light text-ink/15"
            aria-hidden="true"
          >
            0–9
          </span>
        )}
        {sample && result?.sample === input.index && (
          <div className="pointer-events-none absolute inset-0">
            <PixelPreview pixels={result.pixels} />
          </div>
        )}
      </div>
      <div className="mt-4 flex max-w-70 justify-between gap-3">
        <Button variant="secondary" onClick={clear}>
          Clear
        </Button>
        <Button variant="secondary" onClick={trySample} disabled={!connected}>
          Try a sample
        </Button>
      </div>
      {error && (
        <p className="mt-3 text-xs text-danger" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
