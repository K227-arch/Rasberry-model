"use client";

import { useState } from "react";
import TopAppBar from "@/components/TopAppBar";
import BottomNavBar from "@/components/BottomNavBar";

const TOOLBAR_BUTTONS = [
  { icon: "format_bold", label: "Bold" },
  { icon: "format_italic", label: "Italic" },
  { icon: "format_underlined", label: "Underline" },
];

const ALIGN_BUTTONS = [
  { icon: "format_align_left", label: "Align Left" },
  { icon: "format_align_center", label: "Align Center" },
  { icon: "format_align_right", label: "Align Right" },
];

const LIST_BUTTONS = [
  { icon: "format_list_bulleted", label: "Bullet List" },
  { icon: "format_list_numbered", label: "Numbered List" },
];

export default function EditorPage() {
  const [docLang, setDocLang] = useState<"English" | "Runyoro-Rutooro">("Runyoro-Rutooro");

  return (
    <>
      <TopAppBar />
      <section className="bg-surface-container-lowest border-b border-outline-variant sticky top-[64px] z-40">
        <div className="max-w-screen-xl mx-auto px-5 py-3 flex flex-wrap items-center gap-4">
          <div className="flex items-center bg-surface-container rounded-lg p-1">
            {TOOLBAR_BUTTONS.map((btn) => (
              <button
                key={btn.icon}
                className="p-1.5 hover:bg-surface-container-highest rounded transition-all material-symbols-outlined text-on-surface-variant cursor-pointer"
                title={btn.label}
              >
                {btn.icon}
              </button>
            ))}
            <div className="w-px h-6 bg-outline-variant mx-1" />
            {LIST_BUTTONS.map((btn) => (
              <button
                key={btn.icon}
                className="p-1.5 hover:bg-surface-container-highest rounded transition-all material-symbols-outlined text-on-surface-variant cursor-pointer"
                title={btn.label}
              >
                {btn.icon}
              </button>
            ))}
          </div>
          <div className="flex items-center bg-surface-container rounded-lg p-1">
            {ALIGN_BUTTONS.map((btn) => (
              <button
                key={btn.icon}
                className="p-1.5 hover:bg-surface-container-highest rounded transition-all material-symbols-outlined text-on-surface-variant cursor-pointer"
                title={btn.label}
              >
                {btn.icon}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-3 ml-auto">
            <button className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-outline text-on-surface-variant hover:bg-surface-container transition-all cursor-pointer">
              <span className="material-symbols-outlined text-[20px]">spellcheck</span>
              <span className="text-[13px] font-semibold tracking-wider uppercase">Spellcheck</span>
            </button>
            <button className="flex items-center gap-1 bg-primary text-on-primary px-4 py-1.5 rounded-lg text-[13px] font-semibold tracking-wider uppercase hover:opacity-90 active:scale-95 cursor-pointer">
              <span className="material-symbols-outlined text-[20px]">save</span>
              Save
            </button>
          </div>
        </div>
      </section>

      <main className="flex-1 overflow-y-auto pt-6 pb-6 px-5">
        <div className="max-w-[816px] mx-auto">
          <div className="bg-surface-container-lowest paper-shadow min-h-[1056px] p-8 rounded-lg relative">
            <div className="absolute top-0 left-0 w-full h-1 bg-secondary opacity-20 overflow-hidden">
              <div className="h-full bg-secondary animate-pulse w-1/3 rounded-full" />
            </div>
            <div className="flex justify-end mb-6">
              <div className="inline-flex bg-surface-container rounded-full p-1">
                {(["English", "Runyoro-Rutooro"] as const).map((lang) => (
                  <button
                    key={lang}
                    onClick={() => setDocLang(lang)}
                    className={`px-4 py-1.5 rounded-full text-[13px] font-semibold tracking-wider uppercase transition-all cursor-pointer ${
                      docLang === lang
                        ? "bg-secondary text-on-secondary"
                        : "text-on-surface-variant"
                    }`}
                  >
                    {lang}
                  </button>
                ))}
              </div>
            </div>

            <article className="max-w-none">
              <h2 className="text-[24px] font-semibold text-primary mb-4">
                Okukorra Hamu n&apos;ebitabo
              </h2>
              <p className="text-[18px] text-on-surface mb-6 leading-relaxed">
                Eki kika ky&apos;ebitabo kikatandikibwawo n&apos;ekigendererwa
                ky&apos;okuhwera abantu okusoma n&apos;okwetegereza obuhangwa
                bwaitu. Mukama akatuha obusinge n&apos;obumanzi mukukora gunu
                omulimo.
              </p>
              <p className="text-[18px] text-on-surface mb-6 leading-relaxed">
                Tusemeriire kukuuma ebirabo byaitu kandi{" "}
                <span className="relative inline-block group">
                  <span className="border-b-2 border-error text-on-surface cursor-help">
                    okusomererwa
                  </span>
                  <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 bg-inverse-surface text-inverse-on-surface rounded-lg p-3 shadow-xl z-10 hidden group-hover:block">
                    <p className="text-[12px] font-bold mb-1 opacity-80">
                      Did you mean:
                    </p>
                    <button className="w-full text-left font-bold text-secondary-container hover:bg-white/10 p-1 rounded transition-colors mb-1 cursor-pointer">
                      okusomerwa
                    </button>
                    <div className="flex gap-1 border-t border-white/10 pt-1 mt-1">
                      <button className="text-[11px] hover:underline cursor-pointer">
                        Ignore All
                      </button>
                      <button className="text-[11px] hover:underline ml-auto cursor-pointer">
                        Add to Dictionary
                      </button>
                    </div>
                  </span>
                </span>{" "}
                obugabe bwaitu mukisoro. Abantu boona baine obugabe obuntu.
              </p>
              <p className="text-[18px] text-on-surface leading-relaxed">
                Entekaniza enu n&apos;eya mwebaza muno habw&apos;okutuhwera
                mukuteekaniza ebigambo binu. AI Stick neehwera muno
                mukuboneza ennimi zaitu ez&apos;ekibira.
              </p>
            </article>
          </div>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-surface-container-low p-4 rounded-xl border border-outline-variant">
              <div className="flex items-center gap-3 mb-1">
                <span className="material-symbols-outlined text-secondary">menu_book</span>
                <h3 className="font-bold text-primary">Dictionary Match</h3>
              </div>
              <p className="text-on-surface-variant italic mb-2">
                &ldquo;Okusomerwa&rdquo; (Verb)
              </p>
              <p className="text-on-surface-variant">
                To be read for; to be educated. Root: kusoma (read).
              </p>
            </div>
            <div className="bg-surface-container-low p-4 rounded-xl border border-outline-variant">
              <div className="flex items-center gap-3 mb-1">
                <span className="material-symbols-outlined text-secondary">auto_awesome</span>
                <h3 className="font-bold text-primary">AI Suggestion</h3>
              </div>
              <p className="text-on-surface-variant">
                Consider using <strong>&apos;okuhaburwa&apos;</strong> for a more
                formal tone regarding education context.
              </p>
            </div>
          </div>
        </div>
      </main>
      <BottomNavBar />
    </>
  );
}
