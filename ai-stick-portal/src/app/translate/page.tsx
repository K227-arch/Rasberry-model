"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import TopAppBar from "@/components/TopAppBar";
import BottomNavBar from "@/components/BottomNavBar";

const LANGUAGES = ["Runyoro-Rutooro", "English"] as const;
type Lang = (typeof LANGUAGES)[number];

const EXAMPLES: { text: string; src: Lang; tgt: Lang }[] = [
  { text: "Oraire ota?",                          src: "Runyoro-Rutooro", tgt: "English" },
  { text: "Webale muno kunkonyera.",              src: "Runyoro-Rutooro", tgt: "English" },
  { text: "Eizooba nirirasa ha nsozi.",           src: "Runyoro-Rutooro", tgt: "English" },
  { text: "abantu",                               src: "Runyoro-Rutooro", tgt: "English" },
  { text: "How are you?",                         src: "English", tgt: "Runyoro-Rutooro" },
  { text: "Thank you very much.",                 src: "English", tgt: "Runyoro-Rutooro" },
  { text: "We need to plant seeds before rains.", src: "English", tgt: "Runyoro-Rutooro" },
];

function directionStr(src: Lang, tgt: Lang): string {
  return `${src === "Runyoro-Rutooro" ? "Runyoro" : "English"} → ${tgt === "Runyoro-Rutooro" ? "Runyoro" : "English"}`;
}

function saveToHistory(src: Lang, tgt: Lang, source: string, translation: string) {
  try {
    const key = "ai_stick_history";
    const existing = JSON.parse(localStorage.getItem(key) || "[]");
    const entry = {
      id: Date.now(),
      src,
      tgt,
      source,
      translation,
      direction: directionStr(src, tgt),
      timestamp: new Date().toISOString(),
    };
    const updated = [entry, ...existing].slice(0, 100);
    localStorage.setItem(key, JSON.stringify(updated));
  } catch { /* localStorage may be unavailable */ }
}

export default function TranslatePage() {
  const [sourceLang, setSourceLang] = useState<Lang>("English");
  const [targetLang, setTargetLang] = useState<Lang>("Runyoro-Rutooro");
  const [sourceText, setSourceText] = useState("");
  const [translation, setTranslation] = useState("");
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [saved, setSaved] = useState(false);
  const srcRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (srcRef.current) {
      srcRef.current.style.height = "auto";
      srcRef.current.style.height = `${Math.min(srcRef.current.scrollHeight, 240)}px`;
    }
  }, [sourceText]);

  const swapLangs = () => {
    setSourceLang(targetLang);
    setTargetLang(sourceLang);
    setSourceText(translation || "");
    setTranslation("");
  };

  const doTranslate = useCallback(async (text?: string, src?: Lang, tgt?: Lang) => {
    const t = text ?? sourceText;
    const s = src ?? sourceLang;
    const g = tgt ?? targetLang;
    if (!t.trim()) return;
    setLoading(true);
    setTranslation("");
    try {
      const res = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: t, direction: directionStr(s, g) }),
      });
      const data = await res.json();
      const result = data.translation || "No translation returned.";
      setTranslation(result);
      saveToHistory(s, g, t, result);
    } catch {
      setTranslation("Translation service unavailable. Is the model server running?");
    } finally {
      setLoading(false);
    }
  }, [sourceText, sourceLang, targetLang]);

  const handleCopy = () => {
    navigator.clipboard.writeText(translation).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <>
      <TopAppBar />
      <main className="mt-16 md:mt-0 flex-1 flex flex-col w-full px-margin-mobile md:px-8 lg:px-12 pt-6 pb-36 md:pb-8 max-w-5xl md:mx-auto">

        {/* Page title — desktop */}
        <div className="hidden md:flex items-center gap-3 mb-6">
          <span className="material-symbols-outlined text-primary text-[28px]">g_translate</span>
          <h1 className="text-headline-md text-on-background">Translator</h1>
          <span className="ml-auto text-label-sm text-on-surface-variant bg-surface-container px-3 py-1 rounded-full">
            Runyoro-Rutooro ↔ English
          </span>
        </div>

        <div className="w-full flex flex-col gap-4 flex-1">
          {/* Language Selector */}
          <div className="flex items-center justify-between bg-surface-container-lowest rounded-xl p-3 premium-shadow border border-outline-variant/30">
            <div className="flex items-center gap-3">
              <button className="px-4 py-1.5 rounded-full bg-primary-fixed text-on-primary-fixed text-label-md font-semibold uppercase flex items-center gap-1 cursor-pointer">
                {sourceLang === "Runyoro-Rutooro" ? "Runyoro" : "English"}
                <span className="material-symbols-outlined text-[16px]">expand_more</span>
              </button>
              <button
                onClick={swapLangs}
                className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-surface-container transition-all active:scale-90 cursor-pointer"
                aria-label="Swap languages"
              >
                <span className="material-symbols-outlined text-secondary">swap_horiz</span>
              </button>
              <button className="px-4 py-1.5 rounded-full bg-surface-container text-on-surface-variant text-label-md font-semibold uppercase flex items-center gap-1 cursor-pointer">
                {targetLang === "Runyoro-Rutooro" ? "Runyoro" : "English"}
                <span className="material-symbols-outlined text-[16px]">expand_more</span>
              </button>
            </div>
            <span className="text-label-sm text-on-surface-variant hidden sm:block">
              {loading ? "Translating..." : "Neural MT"}
            </span>
          </div>

          {/* Translation Panels */}
          <div className="flex flex-col md:flex-row gap-4 flex-1 min-h-[360px]">
            {/* Source */}
            <div className="flex-1 bg-surface-container-lowest border border-outline-variant rounded-2xl premium-shadow p-5 flex flex-col relative group">
              <div className="absolute top-0 left-0 right-0 h-1 bg-surface-container overflow-hidden rounded-t-2xl opacity-0 group-focus-within:opacity-100 transition-opacity">
                <div className="h-full bg-primary-container w-1/3 animate-pulse" />
              </div>
              <div className="flex justify-between items-center mb-3">
                <span className="text-label-sm font-semibold tracking-widest uppercase text-outline">
                  {sourceLang === "Runyoro-Rutooro" ? "Runyoro-Rutooro" : "English"}
                </span>
                {sourceText && (
                  <button onClick={() => { setSourceText(""); setTranslation(""); }}
                    className="text-outline hover:text-error transition-colors cursor-pointer">
                    <span className="material-symbols-outlined text-[20px]">close</span>
                  </button>
                )}
              </div>
              <textarea
                ref={srcRef}
                value={sourceText}
                onChange={(e) => setSourceText(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && e.ctrlKey) doTranslate(); }}
                className="flex-1 bg-transparent border-none focus:ring-0 text-body-lg text-on-surface resize-none placeholder:text-outline w-full outline-none min-h-[160px]"
                placeholder="Enter text here…  (Ctrl+Enter to translate)"
              />
              <div className="mt-3 flex justify-between items-center border-t border-outline-variant/50 pt-3">
                <span className="text-outline text-label-sm">{sourceText.length} / 5000</span>
                <div className="flex gap-2">
                  <button className="text-outline/40 cursor-not-allowed" aria-label="Voice input (coming soon)" title="Voice input coming soon" disabled>
                    <span className="material-symbols-outlined text-[20px]">mic</span>
                  </button>
                  <button className="text-outline/40 cursor-not-allowed" aria-label="Listen (coming soon)" title="Text-to-speech coming soon" disabled>
                    <span className="material-symbols-outlined text-[20px]">volume_up</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Translate button */}
            <div className="flex flex-row md:flex-col justify-center items-center gap-4">
              <button
                onClick={() => doTranslate()}
                disabled={loading || !sourceText.trim()}
                className="bg-primary text-on-primary w-16 h-16 md:w-20 md:h-20 rounded-full premium-shadow-lg flex items-center justify-center active:scale-90 transition-all hover:bg-primary-container hover:text-on-primary-container disabled:opacity-50 cursor-pointer"
                aria-label="Translate"
              >
                <span className="material-symbols-outlined text-[32px] md:text-[38px]">
                  {loading ? "hourglass_top" : "translate"}
                </span>
              </button>
              <span className="text-label-sm text-on-surface-variant hidden md:block">Translate</span>
            </div>

            {/* Target */}
            <div className="flex-1 bg-white border-2 border-primary-container rounded-2xl premium-shadow p-5 flex flex-col">
              <div className="flex justify-between items-center mb-3">
                <span className="text-label-sm font-semibold tracking-widest uppercase text-secondary">
                  {targetLang === "Runyoro-Rutooro" ? "Runyoro-Rutooro" : "English"}
                </span>
                <div className="flex gap-2">
                  <button onClick={handleCopy} className="text-outline hover:text-primary transition-colors cursor-pointer" aria-label="Copy">
                    <span className="material-symbols-outlined text-[20px]">
                      {copied ? "check" : "content_copy"}
                    </span>
                  </button>
                  <button onClick={handleSave} className="text-outline hover:text-primary transition-colors cursor-pointer" aria-label="Save">
                    <span className="material-symbols-outlined text-[20px]"
                      style={saved ? { fontVariationSettings: "'FILL' 1" } : undefined}>
                      star
                    </span>
                  </button>
                </div>
              </div>
              <div className="flex-1 text-body-lg text-on-surface min-h-[160px]">
                {loading ? (
                  <div className="flex items-center gap-2 text-outline">
                    <span className="material-symbols-outlined animate-spin">progress_activity</span>
                    Translating…
                  </div>
                ) : translation ? (
                  <p className="whitespace-pre-wrap leading-relaxed">{translation}</p>
                ) : (
                  <p className="text-outline-variant italic">Translation will appear here…</p>
                )}
              </div>
              <div className="mt-3 flex justify-between items-center border-t border-outline-variant/50 pt-3">
                <div className="flex gap-2">
                  <button className="text-outline hover:text-primary transition-colors cursor-pointer" aria-label="Share">
                    <span className="material-symbols-outlined text-[20px]">share</span>
                  </button>
                  <button className="text-outline/40 cursor-not-allowed" aria-label="Listen (coming soon)" title="Text-to-speech coming soon" disabled>
                    <span className="material-symbols-outlined text-[20px]">volume_up</span>
                  </button>
                </div>
                <a href="/dictionary" className="text-secondary text-label-md font-semibold uppercase hover:underline cursor-pointer">
                  Dictionary
                </a>
              </div>
            </div>
          </div>

          {/* Example chips */}
          <section>
            <p className="text-label-sm text-on-surface-variant uppercase tracking-widest mb-3">Try these examples</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.text}
                  onClick={() => {
                    setSourceLang(ex.src);
                    setTargetLang(ex.tgt);
                    setSourceText(ex.text);
                    doTranslate(ex.text, ex.src, ex.tgt);
                  }}
                  className="glass-card rounded-full px-4 py-1.5 text-body-sm text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-colors cursor-pointer border-none"
                >
                  {ex.text}
                </button>
              ))}
            </div>
          </section>

          {/* Linguistic tools row */}
          <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { icon: "photo_camera", title: "Camera Lens",    desc: "Point & translate with your camera", href: "/lens"       },
              { icon: "menu_book",    title: "Dictionary",     desc: "Browse Runyoro-English word pairs",  href: "/dictionary" },
              { icon: "history",      title: "History",        desc: "View past translations",             href: "/history"    },
              { icon: "chat_bubble",  title: "AI Chat",        desc: "Conversational translation help",    href: "/chat"       },
            ].map((t) => (
              <a key={t.href} href={t.href}
                className="glass-card rounded-2xl p-4 border border-outline-variant hover:bg-surface-container transition-colors cursor-pointer group flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-primary-fixed/30 flex items-center justify-center flex-shrink-0">
                  <span className="material-symbols-outlined text-primary text-[22px]">{t.icon}</span>
                </div>
                <div>
                  <p className="text-label-md text-on-background font-semibold">{t.title}</p>
                  <p className="text-label-sm text-on-surface-variant">{t.desc}</p>
                </div>
                <span className="material-symbols-outlined text-outline ml-auto">chevron_right</span>
              </a>
            ))}
          </section>
        </div>
      </main>
      <BottomNavBar />
    </>
  );
}
