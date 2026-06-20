"use client";

import { useState } from "react";
import TopAppBar from "@/components/TopAppBar";
import BottomNavBar from "@/components/BottomNavBar";

const LANGUAGES = ["English", "Runyoro-Rutooro"] as const;

export default function TranslatePage() {
  const [sourceLang, setSourceLang] = useState<(typeof LANGUAGES)[number]>("English");
  const [targetLang, setTargetLang] = useState<(typeof LANGUAGES)[number]>("Runyoro-Rutooro");
  const [sourceText, setSourceText] = useState("");
  const [translation, setTranslation] = useState("");
  const [loading, setLoading] = useState(false);

  const swapLangs = () => {
    setSourceLang(targetLang);
    setTargetLang(sourceLang);
    setSourceText("");
    setTranslation("");
  };

  const translate = async () => {
    if (!sourceText.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sourceText, direction: `${sourceLang} → ${targetLang}` }),
      });
      const data = await res.json();
      setTranslation(data.translation || "No translation returned.");
    } catch {
      setTranslation("Translation service unavailable.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <TopAppBar />
      <main className="flex-1 flex flex-col max-w-screen-xl mx-auto w-full px-5 pt-6 pb-32">
        <div className="w-full flex flex-col gap-4 flex-1">
          <div className="flex items-center justify-between bg-surface-container-lowest rounded-xl p-3 shadow-sm">
            <div className="flex items-center gap-3">
              <button className="px-4 py-1.5 rounded-full bg-secondary-container text-on-secondary-container text-[13px] font-semibold tracking-wider uppercase flex items-center gap-1">
                {sourceLang}
                <span className="material-symbols-outlined text-[16px]">expand_more</span>
              </button>
              <button
                onClick={swapLangs}
                className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-surface-container transition-all active:scale-90 cursor-pointer"
              >
                <span className="material-symbols-outlined text-secondary">swap_horiz</span>
              </button>
              <button className="px-4 py-1.5 rounded-full bg-surface-container text-on-surface-variant text-[13px] font-semibold tracking-wider uppercase flex items-center gap-1">
                {targetLang}
                <span className="material-symbols-outlined text-[16px]">expand_more</span>
              </button>
            </div>
            <button className="text-secondary text-[13px] font-semibold tracking-wider uppercase flex items-center gap-1 hover:underline">
              <span className="material-symbols-outlined text-[18px]">history</span>
              Recent
            </button>
          </div>

          <div className="flex-1 flex flex-col md:flex-row gap-4 min-h-[500px]">
            <div className="flex-1 bg-surface-container-lowest border border-outline-variant rounded-xl shadow-lg p-6 flex flex-col relative">
              <div className="flex justify-between items-center mb-4">
                <span className="text-[13px] font-semibold tracking-widest uppercase text-outline">
                  {sourceLang}
                </span>
                <button
                  onClick={() => setSourceText("")}
                  className="text-outline hover:text-error transition-colors active:scale-90 cursor-pointer"
                >
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
              <textarea
                value={sourceText}
                onChange={(e) => setSourceText(e.target.value)}
                className="flex-1 bg-transparent border-none focus:ring-0 text-[18px] text-on-surface resize-none placeholder:text-outline w-full outline-none"
                placeholder="Enter text here to translate..."
              />
              <div className="mt-4 flex justify-between items-center border-t border-outline-variant pt-4">
                <span className="text-outline text-[13px] font-semibold tracking-wider">
                  {sourceText.length} / 5000
                </span>
                <div className="flex gap-3">
                  <button className="text-outline hover:text-primary transition-colors cursor-pointer">
                    <span className="material-symbols-outlined">mic</span>
                  </button>
                  <button className="text-outline hover:text-primary transition-colors cursor-pointer">
                    <span className="material-symbols-outlined">volume_up</span>
                  </button>
                </div>
              </div>
            </div>

            <div className="flex flex-row md:flex-col justify-center items-center gap-4">
              <button
                onClick={translate}
                disabled={loading || !sourceText.trim()}
                className="bg-secondary text-on-secondary w-16 h-16 md:w-20 md:h-20 rounded-full shadow-xl flex items-center justify-center active:scale-90 transition-all hover:bg-on-secondary-container disabled:opacity-50 cursor-pointer"
              >
                <span className="material-symbols-outlined text-[32px] md:text-[40px] group-hover:rotate-12 transition-transform">
                  {loading ? "hourglass_top" : "translate"}
                </span>
              </button>
            </div>

            <div className="flex-1 bg-white border-2 border-secondary-container rounded-xl shadow-lg p-6 flex flex-col">
              <div className="flex justify-between items-center mb-4">
                <span className="text-[13px] font-semibold tracking-widest uppercase text-secondary">
                  {targetLang}
                </span>
                <div className="flex gap-3">
                  <button
                    onClick={() => navigator.clipboard.writeText(translation)}
                    className="text-outline hover:text-primary transition-colors cursor-pointer"
                  >
                    <span className="material-symbols-outlined">content_copy</span>
                  </button>
                  <button className="text-outline hover:text-primary transition-colors cursor-pointer">
                    <span className="material-symbols-outlined">star</span>
                  </button>
                </div>
              </div>
              <div className="flex-1 text-[18px] text-on-surface flex flex-col justify-start">
                {loading ? (
                  <div className="flex items-center gap-2 text-outline">
                    <span className="material-symbols-outlined animate-spin">progress_activity</span>
                    Translating...
                  </div>
                ) : translation ? (
                  <p>{translation}</p>
                ) : (
                  <p className="text-outline-variant italic">Translation will appear here...</p>
                )}
              </div>
              <div className="mt-4 flex justify-between items-center border-t border-outline-variant pt-4">
                <div className="flex gap-3">
                  <button className="text-outline hover:text-primary transition-colors cursor-pointer">
                    <span className="material-symbols-outlined">share</span>
                  </button>
                  <button className="text-outline hover:text-primary transition-colors cursor-pointer">
                    <span className="material-symbols-outlined">volume_up</span>
                  </button>
                </div>
                <button className="text-secondary text-[13px] font-semibold tracking-wider uppercase hover:underline cursor-pointer">
                  Dictionary
                </button>
              </div>
            </div>
          </div>

          <section className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
            {[
              { icon: "menu_book", title: "Dictionary", desc: "Explore complex Runyoro word roots and etymology." },
              { icon: "construction", title: "Linguistic Tools", desc: "Analyze grammar structure and tonal markers." },
              { icon: "star", title: "Saved Phrases", desc: "Quick access to your most used bilingual cards." },
            ].map((item) => (
              <div
                key={item.title}
                className="bg-surface-container-low p-4 rounded-xl border border-outline-variant hover:bg-surface-container transition-colors cursor-pointer group"
              >
                <div className="flex items-center gap-3 mb-1">
                  <span className="material-symbols-outlined text-on-tertiary-container">
                    {item.icon}
                  </span>
                  <h3 className="font-semibold text-primary">{item.title}</h3>
                </div>
                <p className="text-[16px] text-on-surface-variant">{item.desc}</p>
              </div>
            ))}
          </section>
        </div>
      </main>
      <BottomNavBar />
    </>
  );
}
