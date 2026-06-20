import { NextResponse } from "next/server";
import { readFileSync } from "fs";
import path from "path";

export async function GET() {
  try {
    const filePath = path.join(
      process.cwd(),
      "..",
      "runyoro_nmt",
      "data",
      "tm",
      "glossary.json"
    );
    const raw = readFileSync(filePath, "utf-8");
    const data = JSON.parse(raw);
    return NextResponse.json(data);
  } catch (err) {
    console.error("Glossary read error:", err);
    // Return sample data if file not found
    return NextResponse.json([
      { runyoro: "oraire ota?", english: "How are you?", domain: "greetings" },
      { runyoro: "webale", english: "Thank you", domain: "general" },
      { runyoro: "abantu", english: "people", domain: "general" },
      { runyoro: "ente", english: "cows", domain: "agriculture" },
      { runyoro: "muhogo", english: "cassava", domain: "agriculture" },
      { runyoro: "omuceri", english: "rice", domain: "agriculture" },
      { runyoro: "amaizi", english: "water", domain: "general" },
      { runyoro: "eizooba", english: "sun / day", domain: "general" },
    ]);
  }
}
