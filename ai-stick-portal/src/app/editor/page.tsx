"use client";

import { useState } from "react";
import TopAppBar from "@/components/TopAppBar";
import BottomNavBar from "@/components/BottomNavBar";

export default function EditorPage() {
  const [content, setContent] = useState("");
  const [bold, setBold] = useState(false);
  const [italic, setItalic] = useState(false);

  return (
    <>
      <TopAppBar />
      <main className="mt-16 md:mt-0 flex-1 flex flex-col w-full px-margin-mobile pt-6 pb-36 md:pb-8 max-w-4xl md:mx-auto">
        <div className="flex items-center gap-3 mb-6">
          <span className="material-symbols-outlined text-primary text-[28px]">edit_note</span>
          <h1 className="text-headline-md text-on-background">Word Editor</h1>
        </div>

        {/* Toolbar */}
        <div className="bg-surface-container-lowest rounded-xl premium-shadow border border-outline-variant/30 p-3 mb-4 flex flex-wrap items-center gap-2">
          <div className="flex items-center bg-surface-container rounded-lg p-1 gap-1">
            {[
              { icon: "format_bold",        action: () => setBold(!bold),     active: bold   },
              { icon: "format_italic",      action: () => setItalic(!italic), active: italic },
              { icon: "format_underlined",  action: () => {},                 active: false  },
            ].map((b) => (
              <button key={b.icon} onClick={b.action}
                className={`p-2 rounded transition-all cursor-pointer ${b.active ? "bg-primary-fixed text-on-primary-fixed" : "hover:bg-surface-container-highest text-on-surface-variant"}`}>
                <span className="material-symbols-outlined text-[20px]">{b.icon}</span>
              </button>
            ))}
          </div>
          <div className="flex items-center bg-surface-container rounded-lg p-1 gap-1">
            {["format_align_left","format_align_center","format_align_right"].map((icon) => (
              <button key={icon} className="p-2 rounded hover:bg-surface-container-highest text-on-surface-variant transition-all cursor-pointer">
                <span className="material-symbols-outlined text-[20px]">{icon}</span>
              </button>
            ))}
          </div>
          <div className="ml-auto flex gap-2">
            <button className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-outline text-on-surface-variant hover:bg-surface-container text-label-md cursor-pointer">
              <span className="material-symbols-outlined text-[18px]">spellcheck</span>
              Spellcheck
            </button>
            <button className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-primary text-on-primary text-label-md cursor-pointer active:scale-95">
              <span className="material-symbols-outlined text-[18px]">save</span>
              Save
            </button>
          </div>
        </div>

        {/* Document area */}
        <div className="flex-1 bg-surface-container-lowest rounded-2xl premium-shadow border border-outline-variant/30 p-8 min-h-[480px]">
          <div className="flex justify-end mb-4">
            <div className="inline-flex bg-surface-container rounded-full p-1">
              <span className="px-3 py-0.5 rounded-full text-label-sm text-on-surface-variant">English</span>
              <span className="px-3 py-0.5 rounded-full bg-primary text-on-primary text-label-sm">Runyoro</span>
            </div>
          </div>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Start typing your document here…

Mix English and Runyoro-Rutooro text freely. Use the toolbar to format."
            className={`w-full h-full min-h-[380px] bg-transparent border-none outline-none resize-none text-body-lg text-on-surface leading-relaxed ${bold ? "font-bold" : ""} ${italic ? "italic" : ""}`}
          />
        </div>
      </main>
      <BottomNavBar />
    </>
  );
}
