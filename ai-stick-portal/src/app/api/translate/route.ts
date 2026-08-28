import { NextRequest, NextResponse } from "next/server";

// Extend Vercel function timeout to 30s (requires Pro for >10s, but set it anyway)
export const maxDuration = 30;

const PYTHON_BACKEND = process.env.MODEL_API_URL || "http://127.0.0.1:8000";
const HF_SPACE_URL =
  process.env.HF_SPACE_URL ||
  "https://keithtwesigye-runyoro-translator-api.hf.space";

// On Vercel the local backend is never available — skip it to save time
const IS_VERCEL = !!process.env.VERCEL;
const LOCAL_TIMEOUT_MS = 3000;
const HF_TIMEOUT_MS = 28000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() =>
    clearTimeout(timer)
  );
}

/**
 * Normalise any backend response to { translation, direction }.
 * Handles both our FastAPI format and the old HF Space format.
 */
function extractTranslation(
  data: Record<string, unknown>,
  direction: string
): { translation: string; direction: string } {
  const candidates = [
    data.translation,
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
  throw new Error("No translation field found in response: " + JSON.stringify(data));
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
  const res = await fetchWithTimeout(
    `${HF_SPACE_URL}/translate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, direction }),
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

    // On Vercel skip local backend — it's never available there
    if (!IS_VERCEL) {
      try {
        const data = await translateViaLocal(text, direction);
        return NextResponse.json(data);
      } catch {
        // Local not running, fall through to HF Space
      }
    }

    // HuggingFace Space
    try {
      const data = await translateViaHuggingFace(text, direction);
      return NextResponse.json(data);
    } catch (hfError) {
      console.error("HuggingFace fallback error:", hfError);
      return NextResponse.json(
        {
          error:
            "Translation service unavailable. Please try again in a moment.",
        },
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
