/** Typed client for the Seer verification API. */

export const API_BASE =
  process.env.NEXT_PUBLIC_SEER_API ?? "http://localhost:8000";

export interface FieldOut {
  text: string;
  confidence: number;
  format_valid: boolean;
}

export interface DocumentOut {
  kind: string;
  fields: Record<string, FieldOut>;
  mean_confidence: number;
  mrz_valid: boolean | null;
  mrz_checks: Record<string, boolean>;
  viz_mrz_consistency: Record<string, boolean>;
}

export interface FaceOut {
  similarity: number;
  threshold: number;
  fmr_level: string;
  match: boolean;
}

export interface ForensicsOut {
  tamper_probability: number;
  heatmap: number[][] | null;
}

export interface VerifyResponse {
  verdict: "pass" | "review" | "fail";
  reasons: string[];
  corners: number[][] | null;
  document: DocumentOut | null;
  face: FaceOut | null;
  forensics: ForensicsOut | null;
  timings_ms: Record<string, number>;
  stages_available: Record<string, boolean>;
}

export interface HealthResponse {
  status: string;
  ready: boolean;
  stages_available: Record<string, boolean>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const r = await fetch(`${API_BASE}/health`);
  if (!r.ok) throw new Error(`health check failed: ${r.status}`);
  return r.json();
}

export async function verify(
  document: File,
  selfie: File | null,
  fmr: string
): Promise<VerifyResponse> {
  const form = new FormData();
  form.append("document", document);
  if (selfie) form.append("selfie", selfie);
  const r = await fetch(`${API_BASE}/verify?fmr=${encodeURIComponent(fmr)}`, {
    method: "POST",
    body: form,
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => null);
    throw new Error(detail?.detail ?? `verification failed: ${r.status}`);
  }
  return r.json();
}
