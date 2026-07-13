"use client";

import { useCallback, useRef, useState } from "react";

export default function UploadCard({
  title,
  hint,
  onFile,
}: {
  title: string;
  hint: string;
  onFile: (f: File | null) => void;
}) {
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = useCallback(
    (f: File | undefined) => {
      if (!f || !f.type.startsWith("image/")) return;
      setPreview(URL.createObjectURL(f));
      onFile(f);
    },
    [onFile]
  );

  return (
    <div className="card">
      <h2>{title}</h2>
      <div
        className="drop"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          accept(e.dataTransfer.files[0]);
        }}
      >
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={preview} alt={`${title} preview`} />
        ) : (
          <p>{hint}</p>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => accept(e.target.files?.[0])}
      />
    </div>
  );
}
