"use client";

import { useState, useCallback } from "react";
import TopAppBar from "@/components/TopAppBar";
import BottomNavBar from "@/components/BottomNavBar";

export default function EditorPage() {
  const [content, setContent] = useState("");
  const [translatedContent, setTranslatedContent] = useState("");
  const [bold, setBold] = useState(false);
  const [italic, setItalic] = useState(false);
  const [direction, setDirection] = useState<"eng-rny" | "rny-eng">("eng-rny");
  const [translating, setTranslating] = useState(false);
  const [showTranslation, setShowTranslation] = useState(false);

  const translateDocument = useCallback(async () => {
    if (!content.trim()) return;
    setTranslating(true);
    setShowTranslation(true);

    const dirStr =
      direction === "eng-rny" ? "English → Runyoro" : "Runyoro → English";

    // Split into sentences/paragraphs and translate each
    const lines = content.split("\n");
    const translated: string[] = [];

    for (const line of lines) {
      if (!line.trim()) {
        translated.push("");
        continue;
      }
      try {
        const res = await fetch("/api/translate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: line.trim(), direction: dirStr }),
        });
        const data = await res.json();
        translated.push(data.translation || line);
      } catch {
        translated.push(line);
      }
    }

    setTranslatedContent(translated.join("\n"));
    setTranslating(false);
  }, [content, direction]);

  const copyTranslation = () => {
    navigator.clipboard.writeText(translatedContent).catch(() => {});
  };

  const useTranslation = () => {
    setContent(translatedContent);
    setShowTranslation(false);
    setTranslatedContent("");
  };

  return (
    <>
      <TopAppBar />
      <main className="mt-16 md:mt-0 flex-1 flex flex-col w-full px-margin-mobile md:px-8 lg:px-12 pt-6 pb-36 md:pb-8 max-w-5xl md:mx-auto">
        <div className="flex items-center gap-3 mb-6">
          <span className="material-symbols-outlined text-primary text-[28px]">
            edit_note
          </span>
          <h1 className="text-headline-md text-on-background">Word Editor</h1>
          <span className="ml-auto text-label-sm text-on-surface-variant bg-surface-container px-3 py-1 rounded-full">
            NLLB Translation
          </span>
        </div>

        {/* Toolbar */}
        <div className="bg-surface-container-lowest rounded-xl premium-shadow border border-outline-variant/30 p-3 mb-4 flex flex-wrap items-center gap-2">
          <div className="flex items-center bg-surface-container rounded-lg p-1 gap-1">
            {[
              {
                icon: "format_bold",
                action: () => setBold(!bold),
                active: bold,
              },
              {
                icon: "format_italic",
                action: () => setItalic(!italic),
                active: italic,
              },
              {
                icon: "format_underlined",
                action: () => {},
                active: false,
              },
            ].map((b) => (
              <button
                key={b.icon}
                onClick={b.action}
                className={`p-2 rounded transition-all cursor-pointer ${
                  b.active
                    ? "bg-primary-fixed text-on-primary-fixed"
                    : "hover:bg-surface-container-highest text-on-surface-variant"
                }`}
              >
                <span className="material-symbols-outlined text-[20px]">
                  {b.icon}
                </span>
              </button>
            ))}
          </div>

          {/* Direction toggle */}
          <div className="flex items-center bg-surface-container rounded-lg p-1 gap-1">
            <button
              onClick={() => setDirection("eng-rny")}
              className={`px-3 py-1.5 rounded text-label-sm font-medium cursor-pointer transition-all ${
                direction === "eng-rny"
                  ? "bg-primary text-on-primary"
                  : "text-on-surface-variant hover:bg-surface-container-highest"
              }`}
            >
              EN → RNY
            </button>
            <button
              onClick={() => setDirection("rny-eng")}
              className={`px-3 py-1.5 rounded text-label-sm font-medium cursor-pointer transition-all ${
                direction === "rny-eng"
                  ? "bg-primary text-on-primary"
                  : "text-on-surface-variant hover:bg-surface-container-highest"
              }`}
            >
              RNY → EN
            </button>
          </div>

          <div className="ml-auto flex gap-2">
            <button
              onClick={translateDocument}
              disabled={!content.trim() || translating}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-primary text-on-primary text-label-md cursor-pointer active:scale-95 disabled:opacity-50 transition-all"
            >
              <span className="material-symbols-outlined text-[18px]">
                {translating ? "hourglass_top" : "translate"}
              </span>
              {translating ? "Translating..." : "Translate"}
            </button>
          </div>
        </div>

        {/* Document area */}
        <div className="flex-1 flex flex-col md:flex-row gap-4 min-h-[480px]">
          {/* Source editor */}
          <div className="flex-1 bg-surface-container-lowest rounded-2xl premium-shadow border border-outline-variant/30 p-6 flex flex-col">
            <div className="flex justify-between items-center mb-3">
              <span className="text-label-sm font-semibold tracking-widest uppercase text-outline">
                {direction === "eng-rny" ? "English" : "Runyoro-Rutooro"}
              </span>
              <span className="text-label-sm text-outline">
                {content.length} chars
              </span>
            </div>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder={`Start typing your ${direction === "eng-rny" ? "English" : "Runyoro"} text here…\n\nWrite paragraphs, sentences, or individual words.\nClick "Translate" to convert to ${direction === "eng-rny" ? "Runyoro-Rutooro" : "English"}.`}
              className={`w-full flex-1 min-h-[300px] bg-transparent border-none outline-none resize-none text-body-lg text-on-surface leading-relaxed placeholder:text-outline ${
                bold ? "font-bold" : ""
              } ${italic ? "italic" : ""}`}
            />
          </div>

          {/* Translation output */}
          {showTranslation && (
            <div className="flex-1 bg-white border-2 border-primary-container rounded-2xl premium-shadow p-6 flex flex-col">
              <div className="flex justify-between items-center mb-3">
                <span className="text-label-sm font-semibold tracking-widest uppercase text-secondary">
                  {direction === "eng-rny" ? "Runyoro-Rutooro" : "English"}
                </span>
                <div className="flex gap-2">
                  <button
                    onClick={copyTranslation}
                    className="text-outline hover:text-primary transition-colors cursor-pointer"
                    aria-label="Copy"
                  >
                    <span className="material-symbols-outlined text-[20px]">
                      content_copy
                    </span>
                  </button>
                  <button
                    onClick={useTranslation}
                    className="text-outline hover:text-primary transition-colors cursor-pointer"
                    aria-label="Use as source"
                    title="Replace source with translation"
                  >
                    <span className="material-symbols-outlined text-[20px]">
                      swap_horiz
                    </span>
                  </button>
                  <button
                    onClick={() => {
                      setShowTranslation(false);
                      setTranslatedContent("");
                    }}
                    className="text-outline hover:text-error transition-colors cursor-pointer"
                    aria-label="Close"
                  >
                    <span className="material-symbols-outlined text-[20px]">
                      close
                    </span>
                  </button>
                </div>
              </div>
              <div className="flex-1 min-h-[300px] text-body-lg text-on-surface leading-relaxed">
                {translating ? (
                  <div className="flex items-center gap-2 text-outline">
                    <span className="material-symbols-outlined animate-spin">
                      progress_activity
                    </span>
                    Translating document...
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{translatedContent}</p>
                )}
              </div>
            </div>
          )}
        </div>
      </main>
      <BottomNavBar />
    </>
  );
}
