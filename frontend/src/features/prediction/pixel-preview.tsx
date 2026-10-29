import { useEffect, useRef } from "react";

export function PixelPreview({ pixels }: { pixels: number[] }) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context) return;
    const image = context.createImageData(28, 28);
    pixels.forEach((value, index) => {
      image.data.set([value * 255, value * 255, value * 255, 255], index * 4);
    });
    context.putImageData(image, 0, 0);
  }, [pixels]);

  return (
    <canvas
      ref={canvas}
      width={28}
      height={28}
      className="h-full w-full [image-rendering:pixelated]"
      role="img"
      aria-label="MNIST sample"
    />
  );
}
