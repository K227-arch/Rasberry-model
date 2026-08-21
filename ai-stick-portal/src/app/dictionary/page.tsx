"use client";

import { useState, useEffect } from "react";
import TopAppBar from "@/components/TopAppBar";
import BottomNavBar from "@/components/BottomNavBar";

type GlossaryEntry = {
  runyoro: string;
  english: string;
  pos?: string;
  domain?: string;
  definition_en?: string;
};

const DOMAIN_COLORS: Record<string, string> = {
  agriculture: "bg-primary-fixed text-on-primary-fixed",
  health:      "bg-secondary-container text-on-secondary-container",
  education:   "bg-tertiary-container/60 text-on-tertiary-container",
  general:     "bg-surface-container-highest text-on-surface-variant",
};

export default function DictionaryPage() {
  const [entries, setEntries] = useState<GlossaryEntry[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [activeDomain, setActiveDomain] = useState("all");

  useEffect(() => {
    fetch("/api/glossary")
      .then((r) => r.json())
      .then((data) => {
        setEntries(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const domains = ["all", ...Array.from(new Set(entries.map((e) => e.domain || "general")))];

  const filtered = entries.filter((e) => {
    const q = query.toLowerCase();
    const matchQ = !q || e.runyoro?.toLowerCase().includes(q) || e.english?.toLowerCase().includes(q);
    const matchD = activeDomain === "all" || (e.domain || "general") === activeDomain;
    return matchQ && matchD;
  });

  return (
    <>
      <TopAppBar />
      <main className="mt-16 md:mt-0 flex-1 flex flex-col w-full px-margin-mobile md:px-8 lg:px-12 pt-6 pb-36 md:pb-8 max-w-4xl md:mx-auto">

        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <span className="material-symbols-outlined text-primary text-[28px]">menu_book</span>
          <div>
            <h1 className="text-headline-md text-on-background">Dictionary</h1>
            <p className="text-body-sm text-on-surface-variant">
              {entries.length} Runyoro-Rutooro / English entries
            </p>
          </div>
        </div>

        {/* Search */}
        <div className="bg-surface-container-lowest rounded-2xl premium-shadow border border-outline-variant/30 p-4 mb-4 flex items-center gap-3">
          <span className="material-symbols-outlined text-outline">search</span>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search in Runyoro or English…"
            className="flex-1 bg-transparent border-none outline-none text-body-md text-on-surface placeholder:text-outline"
          />
          {query && (
            <button onClick={() => setQuery("")} className="text-outline hover:text-error transition-colors cursor-pointer">
              <span className="material-symbols-outlined text-[20px]">close</span>
            </button>
          )}
        </div>

        {/* Domain filter chips */}
        <div className="flex gap-2 flex-wrap mb-4">
          {domains.map((d) => (
            <button
              key={d}
              onClick={() => setActiveDomain(d)}
              className={`px-3 py-1 rounded-full text-label-md font-semibold uppercase cursor-pointer transition-all ${
                activeDomain === d
                  ? "bg-primary text-on-primary"
                  : "bg-surface-container text-on-surface-variant hover:bg-surface-container-high"
              }`}
            >
              {d}
            </button>
          ))}
        </div>

        {/* Results */}
        {loading ? (
          <div className="flex items-center gap-3 text-on-surface-variant py-8 justify-center">
            <span className="material-symbols-outlined animate-spin">progress_activity</span>
            Loading dictionary…
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-16 text-on-surface-variant">
            <span className="material-symbols-outlined text-[48px] text-outline mb-3 block">search_off</span>
            <p className="text-body-md">No results for &ldquo;{query}&rdquo;</p>
          </div>
        ) : (
          <div className="space-y-2">
            <p className="text-label-sm text-on-surface-variant mb-2">
              Showing {filtered.length} of {entries.length} entries
            </p>
            {filtered.map((entry, i) => (
              <div
                key={i}
                className="glass-card rounded-2xl p-4 premium-shadow flex flex-col sm:flex-row sm:items-start gap-3 sm:gap-4 hover:border-primary transition-colors group"
              >
                {/* Runyoro side */}
                <div className="min-w-0 sm:flex-1">
                  <p className="text-body-lg text-on-background font-semibold leading-tight truncate">
                    {entry.runyoro}
                  </p>
                  {entry.pos && (
                    <p className="text-label-sm text-outline italic">{entry.pos}</p>
                  )}
                </div>

                {/* Divider */}
                <div className="hidden sm:block w-px self-stretch bg-outline-variant/50 flex-shrink-0" />
                <div className="sm:hidden h-px w-full bg-outline-variant/50 flex-shrink-0" />

                {/* English side */}
                <div className="min-w-0 sm:flex-1">
                  <p className="text-body-md text-on-surface leading-snug">
                    {entry.english}
                  </p>
                  {entry.definition_en && (
                    <p className="text-label-sm text-on-surface-variant mt-0.5 line-clamp-2">
                      {entry.definition_en}
                    </p>
                  )}
                </div>

                {/* Domain badge */}
                <span className={`text-label-sm px-2 py-0.5 rounded-full flex-shrink-0 self-start ${DOMAIN_COLORS[entry.domain || "general"] || DOMAIN_COLORS.general}`}>
                  {entry.domain || "general"}
                </span>
              </div>
            ))}
          </div>
        )}
      </main>
      <BottomNavBar />
    </>
  );
}
