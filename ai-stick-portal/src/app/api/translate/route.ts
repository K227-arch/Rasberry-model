import { NextRequest, NextResponse } from "next/server";

const PYTHON_BACKEND = process.env.MODEL_API_URL || "http://127.0.0.1:8000";
const HF_SPACE_URL = "https://keithtwesigye-runyoro-translator-api.hf.space";

// How long to wait for each backend before giving up (ms)
const LOCAL_TIMEOUT_MS = 5000;
const HF_TIMEOUT_MS = 25000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fetchWithTimeout(url: string, options: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() =>
    clearTimeout(timer)
  );
}

/**
 * Normalise the translation response to { translation, direction }.
 * Handles both our new FastAPI format and the old HF Space format.
 */
function extractTranslation(data: Record<string, unknown>, direction: string): { translation: string; direction: string } {
  // Our new format: { translation: string, direction: string }
  if (typeof data.translation === "string" && data.translation.trim()) {
    return { translation: data.translation.trim(), direction };
  }
  // Old HF Space format: { translation_nllb, translation_marian, ... }
  const candidates = [
    data.translation_nllb,
    data.translation_marian,
    data.translated_text,
    data.result,
    data.output,
  ];
  for (const c of candidates) {
    if (typeof c === "string" && c.trim()) {
      return { translation: c.trim(), direction };
    }
  }
  throw new Error("No translation field found in response");
}

// ---------------------------------------------------------------------------
// Backend callers
// ---------------------------------------------------------------------------

async function translateViaLocal(text: string, direction: string) {
  const res = await fetchWithTimeout(
    `${PYTHON_BACKEND}/translate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, direction }),
    },
    LOCAL_TIMEOUT_MS
  );
  if (!res.ok) throw new Error(`Local backend error: ${res.status}`);
  const data = await res.json();
  return extractTranslation(data, direction);
}

async function translateViaHuggingFace(text: string, direction: string) {
  // The old HF Space uses "English -> Runyoro" (ASCII arrow)
  // Map our direction strings to what the old Space understands
  const directionMapped = direction
    .replace("→", "->")
    .replace(" to ", " -> ");

  const res = await fetchWithTimeout(
    `${HF_SPACE_URL}/translate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, direction: directionMapped }),
    },
    HF_TIMEOUT_MS
  );

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HuggingFace Space error: ${res.status} - ${err}`);
  }

  const data = await res.json();
  return extractTranslation(data, direction);
}

// ---------------------------------------------------------------------------
// Route handler
// ---------------------------------------------------------------------------

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { text, direction } = body;

    if (!text || !direction) {
      return NextResponse.json(
        { error: "Missing required fields: text, direction" },
        { status: 400 }
      );
    }

    // Try local backend first (fast, offline)
    try {
      const data = await translateViaLocal(text, direction);
      return NextResponse.json(data);
    } catch {
      // Local server not running — expected on Vercel, fall through to HF Space
    }

    // Fall back to HuggingFace Space
    try {
      const data = await translateViaHuggingFace(text, direction);
      return NextResponse.json(data);
    } catch (hfError) {
      console.error("HuggingFace fallback error:", hfError);
      return NextResponse.json(
        { error: "Translation service unavailable. Please try again in a moment." },
        { status: 503 }
      );
    }
  } catch (error) {
    console.error("Translation API error:", error);
    return NextResponse.json(
      { error: "Translation service unavailable." },
      { status: 503 }
    );
  }
}
