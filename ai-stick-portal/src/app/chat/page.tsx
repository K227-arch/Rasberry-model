"use client";

import { useState, useRef, useEffect } from "react";
import TopAppBar from "@/components/TopAppBar";
import BottomNavBar from "@/components/BottomNavBar";

type Message = {
  role: "user" | "assistant";
  content: string;
  lang: "EN" | "RU";
  isAudio?: boolean;
  duration?: string;
  original?: string;
};

const LANGUAGES = ["English", "Runyoro-Rutooro"] as const;

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Welcome! Type a message in English or Runyoro-Rutooro and I'll translate it for you.",
      lang: "EN",
    },
  ]);
  const [input, setInput] = useState("");
  const [activeLang, setActiveLang] = useState<(typeof LANGUAGES)[number]>("English");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg: Message = {
      role: "user",
      content: input,
      lang: activeLang === "English" ? "EN" : "RU",
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    const placeholder: Message = {
      role: "assistant",
      content: "Thinking...",
      lang: activeLang === "English" ? "RU" : "EN",
    };
    setMessages((prev) => [...prev, placeholder]);

    try {
      const res = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: input,
          direction:
            activeLang === "English"
              ? "English → Runyoro"
              : "Runyoro → English",
        }),
      });
      const data = await res.json();
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          role: "assistant",
          content: data.translation || "No translation returned.",
          lang: activeLang === "English" ? "RU" : "EN",
        };
        return next;
      });
    } catch {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          role: "assistant",
          content: "Service unavailable. Please try again.",
          lang: activeLang === "English" ? "RU" : "EN",
        };
        return next;
      });
    }
  };

  return (
    <>
      <TopAppBar />
      <main className="mt-16 md:mt-0 flex flex-col h-[calc(100dvh-4rem)] md:h-screen w-full max-w-3xl mx-auto">
        {/* Language Toggle */}
        <div className="flex justify-center gap-3 py-3 px-4 border-b border-outline-variant/30 bg-background/80 backdrop-blur-sm sticky top-16 md:top-0 z-10">
          <div className="glass-card p-1 rounded-full flex gap-1">
            {LANGUAGES.map((lang) => (
              <button
                key={lang}
                onClick={() => setActiveLang(lang)}
                className={`px-4 py-1.5 rounded-full text-label-md font-semibold tracking-wider uppercase transition-all cursor-pointer ${
                  activeLang === lang
                    ? "bg-primary text-on-primary"
                    : "hover:bg-surface-container-high text-on-surface-variant"
                }`}
              >
                {lang === "Runyoro-Rutooro" ? "Runyoro" : lang}
              </button>
            ))}
          </div>
          <span className="hidden sm:flex items-center text-label-sm text-on-surface-variant">
            Input: {activeLang}
          </span>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              {msg.role === "user" ? (
                <div className="flex flex-col items-end gap-1 max-w-[80%] sm:max-w-[70%]">
                  <div className="bg-primary text-on-primary px-4 py-3 rounded-2xl rounded-tr-sm text-body-md">
                    {msg.content}
                  </div>
                  <span className="text-on-surface-variant text-label-sm mr-1">
                    {msg.lang}
                  </span>
                </div>
              ) : (
                <div className="flex flex-col items-start gap-1 max-w-[80%] sm:max-w-[70%]">
                  <div className="bg-surface-container-lowest border border-outline-variant/50 px-4 py-3 rounded-2xl rounded-tl-sm text-body-md">
                    {msg.content === "Thinking..." ? (
                      <div className="flex items-center gap-2 text-outline">
                        <span className="material-symbols-outlined animate-spin text-[18px]">
                          progress_activity
                        </span>
                        Translating...
                      </div>
                    ) : (
                      <p className="text-on-surface whitespace-pre-line">{msg.content}</p>
                    )}
                  </div>
                  <div className="flex items-center gap-1 ml-1">
                    <span className="text-on-surface-variant text-label-sm">{msg.lang}</span>
                    {msg.content !== "Thinking..." && (
                      <button
                        onClick={() => navigator.clipboard.writeText(msg.content)}
                        className="material-symbols-outlined text-on-surface-variant/60 text-[16px] hover:text-primary cursor-pointer"
                        aria-label="Copy"
                      >
                        content_copy
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar — positioned above bottom nav */}
        <div className="border-t border-outline-variant/30 bg-background px-3 py-3 mb-16 md:mb-0">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              className="flex-1 bg-surface-container-low rounded-xl py-3 px-4 text-on-surface text-body-md focus:ring-2 focus:ring-primary resize-none outline-none border-none min-h-[44px] max-h-[120px]"
              placeholder={`Type in ${activeLang}...`}
              rows={1}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim()}
              className="p-3 bg-primary text-on-primary rounded-xl hover:opacity-90 transition-all active:scale-90 disabled:opacity-40 cursor-pointer flex-shrink-0"
              aria-label="Send"
            >
              <span className="material-symbols-outlined text-[22px]">send</span>
            </button>
          </div>
        </div>
      </main>
      <BottomNavBar />
    </>
  );
}
