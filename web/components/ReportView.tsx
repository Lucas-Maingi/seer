"use client";

import type { VerifyResponse } from "@/lib/api";
import HeatmapCanvas from "./HeatmapCanvas";

const STAGE_COLORS: Record<string, string> = {
  localize: "#4f8cff",
  ocr: "#8a6ff0",
  forensics: "#e8b93c",
  face: "#2fbf71",
};

function Check({ ok }: { ok: boolean }) {
  return <span className={`badge ${ok ? "ok" : "bad"}`}>{ok ? "OK" : "FAIL"}</span>;
}

export default function ReportView({ report }: { report: VerifyResponse }) {
  const doc = report.document;
  const face = report.face;
  const forensics = report.forensics;
  const stages = Object.entries(report.timings_ms).filter(
    ([k]) => k !== "total"
  );
  const total = report.timings_ms.total ?? 0;

  return (
    <section>
      <div className={`banner ${report.verdict}`}>
        {report.verdict === "pass"
          ? "✓ Verified"
          : report.verdict === "review"
            ? "⚠ Needs review"
            : "✗ Rejected"}
      </div>
      <ul className="reasons">
        {report.reasons.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>

      <div className="grid">
        {doc && (
          <div className="card" style={{ gridColumn: "1 / -1" }}>
            <h2>
              Extracted fields — {doc.kind.replace("_", " ")} (mean confidence{" "}
              {(doc.mean_confidence * 100).toFixed(1)}%)
            </h2>
            <table>
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Value</th>
                  <th>Confidence</th>
                  <th>Format</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(doc.fields).map(([name, f]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td className="mono">{f.text || "—"}</td>
                    <td>
                      <div className="confbar">
                        <div style={{ width: `${f.confidence * 100}%` }} />
                      </div>
                    </td>
                    <td>
                      <Check ok={f.format_valid} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {doc.mrz_valid !== null && (
              <p style={{ fontSize: 14, marginTop: 12 }}>
                MRZ check digits: <Check ok={doc.mrz_valid} />{" "}
                {Object.entries(doc.mrz_checks).map(([k, ok]) => (
                  <span key={k} style={{ marginLeft: 10 }}>
                    {k} <Check ok={ok} />
                  </span>
                ))}
              </p>
            )}
            {Object.keys(doc.viz_mrz_consistency).length > 0 && (
              <p style={{ fontSize: 14 }}>
                Visual zone vs MRZ:{" "}
                {Object.entries(doc.viz_mrz_consistency).map(([k, ok]) => (
                  <span key={k} style={{ marginLeft: 10 }}>
                    {k} <Check ok={ok} />
                  </span>
                ))}
              </p>
            )}
          </div>
        )}

        {face && (
          <div className="card">
            <h2>Face verification</h2>
            <p style={{ fontSize: 14 }}>
              Cosine similarity{" "}
              <strong>{face.similarity.toFixed(3)}</strong> vs threshold{" "}
              <strong>{face.threshold.toFixed(3)}</strong> at FMR{" "}
              {face.fmr_level} — <Check ok={face.match} />
            </p>
            <div className="gauge">
              <div
                className="fill"
                style={{
                  width: `${Math.max(0, Math.min(1, (face.similarity + 1) / 2)) * 100}%`,
                  background: face.match ? "var(--pass)" : "var(--fail)",
                }}
              />
              <div
                className="thr"
                style={{ left: `${((face.threshold + 1) / 2) * 100}%` }}
              />
            </div>
            <p className="footnote">
              Threshold calibrated on LFW to the selected false-match rate.
            </p>
          </div>
        )}

        {forensics && (
          <div className="card">
            <h2>
              Tamper forensics — probability{" "}
              {(forensics.tamper_probability * 100).toFixed(1)}%
            </h2>
            {forensics.heatmap && <HeatmapCanvas heatmap={forensics.heatmap} />}
            <p className="footnote">
              Red regions drive the score: ELA + noise-residual evidence.
            </p>
          </div>
        )}

        <div className="card" style={{ gridColumn: "1 / -1" }}>
          <h2>Latency — {total.toFixed(0)} ms total (CPU)</h2>
          <div className="latency">
            {stages.map(([k, v]) => (
              <div
                key={k}
                style={{
                  width: `${total ? (v / total) * 100 : 0}%`,
                  background: STAGE_COLORS[k] ?? "#666",
                }}
                title={`${k}: ${v.toFixed(1)} ms`}
              >
                {k} {v.toFixed(0)}ms
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
