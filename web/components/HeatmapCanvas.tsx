"use client";

import { useEffect, useRef } from "react";

/** Renders the tamper heatmap (row-major [h][w] probabilities) as a
 *  blue→red overlay so a reviewer can see *where* the model is looking. */
export default function HeatmapCanvas({ heatmap }: { heatmap: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || heatmap.length === 0) return;
    const h = heatmap.length;
    const w = heatmap[0].length;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const p = Math.min(1, Math.max(0, heatmap[y][x]));
        const i = (y * w + x) * 4;
        img.data[i] = Math.round(255 * p); // red = suspicious
        img.data[i + 1] = 40;
        img.data[i + 2] = Math.round(255 * (1 - p));
        img.data[i + 3] = Math.round(60 + 160 * p);
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [heatmap]);

  return (
    <canvas
      ref={ref}
      style={{
        width: "100%",
        imageRendering: "pixelated",
        borderRadius: 8,
        border: "1px solid var(--border)",
      }}
    />
  );
}
