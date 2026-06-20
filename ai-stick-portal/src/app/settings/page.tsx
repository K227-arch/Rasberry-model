"use client";

import { useEffect, useState } from "react";
import TopAppBar from "@/components/TopAppBar";
import BottomNavBar from "@/components/BottomNavBar";

type HealthData = {
  status: string;
  model_loaded?: boolean;
  device?: string;
  bleu?: number;
  chrf?: number;
  error?: string;
};

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => { setHealth(d); setChecking(false); })
      .catch(() => {
        setHealth({ status: "offline", model_loaded: false, error: "Unreachable" });
        setChecking(false);
      });
  }, []);

  const isOnline = health?.status === "ok" && health?.model_loaded;

  const sections = [
    {
      title: "Model Server",
      items: [
        { label: "Status",       value: checking ? "Checking…" : isOnline ? "Online ✓" : "Offline ✗", highlight: isOnline },
        { label: "Device",       value: health?.device || "—"   },
        { label: "BLEU Score",   value: health?.bleu ? `${health.bleu}` : "—"   },
        { label: "chrF++ Score", value: health?.chrf ? `${health.chrf}` : "—"   },
        { label: "Endpoint",     value: "http://127.0.0.1:8000" },
      ],
    },
    {
      title: "Start the Model Server",
      items: [
        { label: "Command",  value: "python model_server.py"                    },
        { label: "Location", value: "ai-stick-portal/model_server.py"           },
        { label: "Model",    value: "kathay/runyoro-nmt-v1 (NLLB-200 1.3B)"     },
      ],
    },
    {
      title: "About",
      items: [
        { label: "App",         value: "AI Stick — Offline Language Intelligence" },
        { label: "Model",       value: "runyoro-nmt-v1"                           },
        { label: "Language pair",value: "Runyoro-Rutooro ↔ English"              },
        { label: "HF Hub",      value: "kathay/runyoro-nmt-v1"                   },
        { label: "Dataset",     value: "kathay/runyoro-rutooro-en-parallel"       },
        { label: "Version",     value: "1.0.0"                                   },
      ],
    },
  ];

  return (
    <>
      <TopAppBar />
      <main className="mt-16 md:mt-0 flex-1 flex flex-col w-full px-margin-mobile pt-6 pb-36 md:pb-8 max-w-2xl md:mx-auto">

        <div className="flex items-center gap-3 mb-6">
          <span className="material-symbols-outlined text-primary text-[28px]">settings</span>
          <h1 className="text-headline-md text-on-background">Settings</h1>
        </div>

        {/* Server status banner */}
        <div className={`rounded-2xl p-4 mb-6 flex items-center gap-3 premium-shadow ${
          isOnline ? "bg-primary-fixed" : "bg-error-container"
        }`}>
          <div className={`w-3 h-3 rounded-full flex-shrink-0 ${
            checking ? "bg-outline animate-pulse" : isOnline ? "bg-green-600 animate-pulse" : "bg-error"
          }`} />
          <div>
            <p className={`text-label-md font-semibold ${isOnline ? "text-on-primary-fixed" : "text-on-error-container"}`}>
              {checking ? "Checking model server…" : isOnline ? "Model server is running" : "Model server is offline"}
            </p>
            <p className={`text-label-sm ${isOnline ? "text-on-primary-fixed-variant" : "text-on-error-container"}`}>
              {isOnline
                ? `NLLB-200 1.3B ready on ${health?.device}`
                : "Run: python model_server.py"}
            </p>
          </div>
        </div>

        {sections.map((section) => (
          <div key={section.title} className="mb-6">
            <h2 className="text-label-md text-on-surface-variant uppercase tracking-widest mb-3 px-1">
              {section.title}
            </h2>
            <div className="bg-surface-container-lowest rounded-2xl premium-shadow border border-outline-variant/30 overflow-hidden">
              {section.items.map((item, i) => (
                <div
                  key={item.label}
                  className={`flex items-center justify-between px-4 py-3 ${
                    i < section.items.length - 1 ? "border-b border-outline-variant/30" : ""
                  }`}
                >
                  <span className="text-body-sm text-on-surface-variant">{item.label}</span>
                  <span className={`text-body-sm font-medium text-right max-w-[55%] truncate ${"highlight" in item && item.highlight ? "text-primary" : "text-on-surface"}`}>
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}

        {/* Refresh status */}
        <button
          onClick={() => {
            setChecking(true);
            fetch("/api/health")
              .then((r) => r.json())
              .then((d) => { setHealth(d); setChecking(false); })
              .catch(() => { setHealth({ status: "offline", model_loaded: false }); setChecking(false); });
          }}
          className="w-full bg-surface-container rounded-xl py-3 flex items-center justify-center gap-2 text-label-md text-on-surface-variant hover:bg-surface-container-high transition-colors cursor-pointer active:scale-95"
        >
          <span className="material-symbols-outlined text-[20px]">refresh</span>
          Refresh Server Status
        </button>
      </main>
      <BottomNavBar />
    </>
  );
}
