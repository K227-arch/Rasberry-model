"use client";

import { useRouter } from "next/navigation";
import TopAppBar from "@/components/TopAppBar";
import BottomNavBar from "@/components/BottomNavBar";

const TOOLS = [
  {
    title: "Translator",
    desc: "Instant neural translation across 64 languages without internet.",
    icon: "g_translate",
    iconBg: "bg-secondary-container text-on-secondary-container",
    size: "large",
    href: "/translate",
  },
  {
    title: "Word Editor",
    desc: "Advanced syntax & grammar refining.",
    icon: "edit_note",
    iconBg: "bg-surface-container-highest text-primary",
    size: "small",
    href: "/editor",
  },
  {
    title: "AI Chatbot",
    desc: "Conversational intelligence on-device.",
    icon: "chat_bubble",
    iconBg: "bg-primary-container/20 text-primary",
    size: "small",
    href: "/chat",
  },
  {
    title: "Document & Audio",
    desc: "Batch process large files and voice recordings locally.",
    icon: "description",
    iconBg: "bg-tertiary-container/30 text-tertiary",
    size: "wide",
    href: "#",
  },
  {
    title: "Dictionary",
    desc: "Offline etymology and comprehensive definitions.",
    icon: "menu_book",
    iconBg: "bg-secondary-fixed text-on-secondary-fixed",
    size: "wide",
    href: "#",
  },
] as const;

const SYSTEM_STATUS = [
  { label: "Neural Engine", value: "Ready" },
  { label: "Local Models", value: "64 Installed" },
] as const;

export default function HomePage() {
  const router = useRouter();

  return (
    <>
      <TopAppBar />
      <main className="mt-16 px-margin-mobile pb-32">
        {/* Hero Section */}
        <section className="py-lg">
          <div className="relative overflow-hidden rounded-3xl bg-surface-container-lowest p-lg premium-shadow border border-outline-variant/30">
            <div className="relative z-10">
              <span className="inline-flex items-center gap-2 px-3 py-1 bg-primary-fixed text-on-primary-fixed rounded-full text-label-md mb-md">
                <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'FILL' 1" }}>security</span>
                100% OFFLINE ENCRYPTION
              </span>
              <h1 className="text-headline-lg-mobile text-on-background mb-sm leading-tight">
                Uncompromised Power, <br />
                <span className="text-primary">Fully Offline.</span>
              </h1>
              <p className="text-body-md text-on-surface-variant max-w-[80%] mb-lg">
                Premium AI processing for global professionals. No cloud. No limits. Just performance.
              </p>
              <button className="bg-primary text-on-primary px-lg py-md rounded-xl text-label-md flex items-center gap-2 active:scale-95 transition-all shadow-md cursor-pointer"
                onClick={() => router.push("/translate")}
              >
                START NEW PROJECT
                <span className="material-symbols-outlined">arrow_forward</span>
              </button>
            </div>
            {/* Background decoration */}
            <div className="absolute -right-12 -top-12 w-48 h-48 bg-primary-container/20 rounded-full blur-3xl" />
            <div className="absolute -right-4 bottom-0 opacity-10">
              <span className="material-symbols-outlined text-[120px] text-primary" style={{ fontVariationSettings: "'wght' 200" }}>memory</span>
            </div>
          </div>
        </section>

        {/* Tool Grid (Bento Style) */}
        <section className="mb-xl">
          <h2 className="text-headline-sm text-on-background mb-lg">Primary Tools</h2>
          <div className="grid grid-cols-2 gap-md">
            {TOOLS.map((tool) => {
              const baseClasses = "glass-card rounded-2xl flex flex-col gap-sm hover:border-primary transition-colors cursor-pointer group premium-shadow";
              const sizeClasses = {
                large: "col-span-2 p-lg",
                wide: "col-span-2 p-md flex items-center gap-md",
                small: "p-md",
              };
              return (
                <div
                  key={tool.title}
                  onClick={() => tool.href !== "#" && router.push(tool.href)}
                  className={`${baseClasses} ${sizeClasses[tool.size]}`}
                >
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform flex-shrink-0 ${tool.iconBg}`}>
                    <span className="material-symbols-outlined text-[28px]">{tool.icon}</span>
                  </div>
                  <div>
                    <h3 className="text-headline-sm text-on-background">{tool.title}</h3>
                    <p className="text-body-sm text-on-surface-variant leading-tight">{tool.desc}</p>
                  </div>
                  {tool.size === "wide" && (
                    <span className="material-symbols-outlined text-outline ml-auto">chevron_right</span>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* Status & Recent Activity */}
        <section className="mb-lg">
          <div className="bg-surface-container-low rounded-2xl p-md border border-outline-variant/50">
            <div className="flex justify-between items-center mb-md">
              <h3 className="text-label-md text-on-surface">SYSTEM STATUS</h3>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-label-sm text-on-surface-variant">Optimized</span>
              </div>
            </div>
            <div className="space-y-sm">
              {SYSTEM_STATUS.map((item, i) => (
                <div key={i} className="flex items-center justify-between text-on-surface-variant">
                  <span className="text-body-sm">{item.label}</span>
                  <span className="text-label-md">{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
      <BottomNavBar />
    </>
  );
}