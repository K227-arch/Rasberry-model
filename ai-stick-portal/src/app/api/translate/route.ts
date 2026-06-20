import { NextRequest, NextResponse } from "next/server";

const PYTHON_BACKEND = process.env.MODEL_API_URL || "http://127.0.0.1:8000";

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

    const res = await fetch(`${PYTHON_BACKEND}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, direction }),
    });

    if (!res.ok) {
      const err = await res.text();
      return NextResponse.json(
        { error: `Backend error: ${err}` },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Translation API error:", error);
    return NextResponse.json(
      { error: "Translation service unavailable. Is the model server running?" },
      { status: 503 }
    );
  }
}
