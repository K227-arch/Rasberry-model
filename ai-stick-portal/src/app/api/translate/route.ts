import { NextRequest, NextResponse } from "next/server";

const PYTHON_BACKEND = process.env.MODEL_API_URL || "http://127.0.0.1:8000";

async function translateViaLocal(text: string, direction: string) {
  const res = await fetch(`${PYTHON_BACKEND}/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, direction }),
  });
  if (!res.ok) throw new Error(`Local backend error: ${res.status}`);
  return await res.json();
}

async function translateViaHuggingFace(text: string, direction: string) {
  // Use the HuggingFace Space API (same interface as local server)
  const HF_SPACE_URL = "https://kathay-runyoro-nmt-api.hf.space";

  const res = await fetch(`${HF_SPACE_URL}/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, direction }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HuggingFace Space error: ${res.status} - ${err}`);
  }

  return await res.json();
}

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

    // Try local backend first, fall back to HuggingFace Inference API
    try {
      const data = await translateViaLocal(text, direction);
      return NextResponse.json(data);
    } catch {
      // Local server not available, use HuggingFace
      try {
        const data = await translateViaHuggingFace(text, direction);
        return NextResponse.json(data);
      } catch (hfError) {
        console.error("HuggingFace fallback error:", hfError);
        return NextResponse.json(
          { error: "Translation service unavailable. Both local and cloud backends failed." },
          { status: 503 }
        );
      }
    }
  } catch (error) {
    console.error("Translation API error:", error);
    return NextResponse.json(
      { error: "Translation service unavailable." },
      { status: 503 }
    );
  }
}
