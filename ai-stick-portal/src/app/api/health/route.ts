import { NextResponse } from "next/server";

const PYTHON_BACKEND = process.env.MODEL_API_URL || "http://127.0.0.1:8000";

export async function GET() {
  try {
    const res = await fetch(`${PYTHON_BACKEND}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) {
      return NextResponse.json({ status: "error", model_loaded: false }, { status: 200 });
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { status: "offline", model_loaded: false, error: "Model server not reachable" },
      { status: 200 }
    );
  }
}
