"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import TopAppBar from "@/components/TopAppBar";
import BottomNavBar from "@/components/BottomNavBar";

type HistoryEntry = {
  id: number;
  src: string;
  tgt: string;
  source: string;
  translation: string;
  direction: string;
  timestamp: string;
};

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export default function HistoryPage() {
  const router = useRouter();
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [query, setQuery] = useState("");

  useEffect(() => {
    try {
      const raw = localStorage.getItem("ai_stick_history") || "[]";
      setEntries(JSON.parse(raw));
    } catch {}
  }, []);

  const clearHistory = () => {
    localStorage.removeItem("ai_stick_history");
    setEntries([]);
  };

  const filtered = entries.filter((e) => {
    const q = query.toLowerCase();
    return !q || e.source.toLowerCase().includes(q) || e.translation.toLowerCase().includes(q);
  });

  return (
    <>
      <TopAppBar />
      <main className="mt-16 md:mt-0 flex-1 flex flex-col w-full px-margin-mobile pt-6 pb-36 md:pb-8 max-w-4xl md:mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <span className="material-symbols-outlined text-primary text-[28px]">history</span>
            <div>
              <h1 className="text-headline-md text-on-background">History</h1>
              <p className="text-body-sm text-on-surface-variant">{entries.length} translations</p>
            </div>
          </div>
          {entries.length > 0 && (
            <button
              onClick={clearHistory}
              className="flex items-center gap-1 px-3 py-1.5 rounded-xl text-label-md text-error hover:bg-error-container transition-colors cursor-pointer"
            >
              <span className="material-symbols-outlined text-[18px]">delete_sweep</span>
              Clear all
            </button>
          )}
        </div>

        {/* Search */}
        {entries.length > 0 && (
          <div className="bg-surface-container-lowest rounded-2xl premium-shadow border border-outline-variant/30 p-4 mb-4 flex items-center gap-3">
            <span className="material-symbols-outlined text-outline">search</span>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search history…"
              className="flex-1 bg-transparent border-none outline-none text-body-md placeholder:text-outline"
            />
          </div>
        )}

        {filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 text-on-surface-variant gap-4">
            <span className="material-symbols-outlined text-[64px] text-outline">history</span>
            <p className="text-body-md">No translation history yet.</p>
            <button
              onClick={() => router.push("/translate")}
              className="bg-primary text-on-primary px-6 py-3 rounded-xl text-label-md font-semibold cursor-pointer active:scale-95 transition-all"
            >
              Start Translating
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {filtered.map((entry) => (
              <div
                key={entry.id}
                className="glass-card rounded-2xl p-4 premium-shadow hover:border-primary transition-colors group cursor-pointer"
                onClick={() => router.push(`/translate`)}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-label-sm bg-primary-fixed text-on-primary-fixed px-2 py-0.5 rounded-full">
                    {entry.direction}
                  </span>
                  <span className="text-label-sm text-on-surface-variant">{timeAgo(entry.timestamp)}</span>
                </div>
                <p className="text-body-md text-on-surface font-medium mb-1 line-clamp-2">{entry.source}</p>
                <div className="flex items-start gap-2">
                  <span className="material-symbols-outlined text-primary text-[16px] mt-0.5 flex-shrink-0">translate</span>
                  <p className="text-body-sm text-on-surface-variant line-clamp-2">{entry.translation}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
      <BottomNavBar />
    </>
  );
}
