import { NextRequest, NextResponse } from "next/server";

const PYTHON_BACKEND = process.env.MODEL_API_URL || "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { image, direction } = body;

    if (!image || !direction) {
      return NextResponse.json(
        { error: "Missing required fields: image, direction" },
        { status: 400 }
      );
    }

    const res = await fetch(`${PYTHON_BACKEND}/ocr`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image, direction }),
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
    console.error("OCR API error:", error);
    return NextResponse.json(
      { error: "OCR service unavailable. Is the model server running?" },
      { status: 503 }
    );
  }
}
