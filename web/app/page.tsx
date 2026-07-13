"use client";

import { useEffect, useState } from "react";
import ReportView from "@/components/ReportView";
import UploadCard from "@/components/UploadCard";
import {
  fetchHealth,
  verify,
  type HealthResponse,
  type VerifyResponse,
} from "@/lib/api";

export default function Home() {
  const [document, setDocument] = useState<File | null>(null);
  const [selfie, setSelfie] = useState<File | null>(null);
  const [fmr, setFmr] = useState("1e-3");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<VerifyResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  async function run() {
    if (!document) return;
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      setReport(await verify(document, selfie, fmr));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <h1>Seer — KYC document verification</h1>
      <p className="sub">
        Corner localization → homography rectification → field OCR with MRZ
        check digits → face verification → tamper forensics. All specimen
        data; all CPU.
        {health &&
          !health.ready &&
          " ⚠ API is up but models are not trained/exported yet."}
        {health === null && " ⚠ API unreachable — start the FastAPI service."}
      </p>

      <div className="grid">
        <UploadCard
          title="Document photo"
          hint="Click or drop a photo of the ID card / passport data page"
          onFile={setDocument}
        />
        <UploadCard
          title="Selfie (optional)"
          hint="Click or drop a live selfie to verify against the portrait"
          onFile={setSelfie}
        />
      </div>

      <div className="row">
        <button
          className="primary"
          onClick={run}
          disabled={!document || busy}
        >
          {busy ? "Verifying…" : "Verify"}
        </button>
        <label style={{ fontSize: 14, color: "var(--muted)" }}>
          Face operating point{" "}
          <select value={fmr} onChange={(e) => setFmr(e.target.value)}>
            <option value="1e-2">FMR 1e-2 (lenient)</option>
            <option value="1e-3">FMR 1e-3 (default)</option>
            <option value="1e-4">FMR 1e-4 (strict)</option>
          </select>
        </label>
      </div>

      {error && <p className="error">{error}</p>}
      {report && <ReportView report={report} />}

      <p className="footnote">
        Seer never stores uploads. Every model in this pipeline was trained on
        synthetic specimen documents — no real identity document was used.
      </p>
    </main>
  );
}
