"use client";

import { useState } from "react";
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
      content:
        'In Runyoro-Rutooro, you would say:\n\n**"Eizooba nirirasa ha nsozi."**',
      lang: "RU",
    },
    {
      role: "user",
      content: 'Audio message • 0:04\n"Webale muno kunkonyera"',
      lang: "RU",
      isAudio: true,
      duration: "0:04",
      original: "Webale muno kunkonyera",
    },
    {
      role: "assistant",
      content:
        'You\'re very welcome! (You said: "Thank you very much for helping me"). Is there anything else you\'d like to translate?',
      lang: "EN",
    },
  ]);
  const [input, setInput] = useState("");
  const [activeLang, setActiveLang] = useState<(typeof LANGUAGES)[number]>("English");

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
              ? "English → Runyoro-Rutooro"
              : "Runyoro-Rutooro → English",
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

  const getGreeting = () => {
    const h = new Date().getHours();
    if (h < 12) return "Good Morning";
    if (h < 18) return "Good Afternoon";
    return "Good Evening";
  };

  return (
    <>
      <TopAppBar />
      <div className="w-full h-1 bg-surface-container overflow-hidden">
        <div className="h-full bg-secondary-container w-1/3 animate-pulse" />
      </div>
      <main className="flex-1 flex flex-col max-w-screen-xl mx-auto w-full px-5 py-6 gap-6 pb-36">
        <div className="flex justify-center gap-3 sticky top-20 z-40">
          <div className="bg-surface-container-highest/80 backdrop-blur-md p-1 rounded-full flex gap-1 shadow-sm">
            {LANGUAGES.map((lang) => (
              <button
                key={lang}
                onClick={() => setActiveLang(lang)}
                className={`px-4 py-1.5 rounded-full text-[13px] font-semibold tracking-wider uppercase transition-all cursor-pointer ${
                  activeLang === lang
                    ? "bg-secondary text-on-secondary"
                    : "hover:bg-surface-container-high text-on-surface-variant"
                }`}
              >
                {lang}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-4">
          {messages.map((msg, i) => (
            <div key={i}>
              {msg.role === "user" ? (
                <div className="flex flex-col items-end gap-1 max-w-[85%] self-end ml-auto">
                  {msg.isAudio ? (
                    <div className="bg-primary text-on-primary p-4 rounded-xl rounded-tr-none shadow-sm flex items-center gap-3">
                      <span
                        className="material-symbols-outlined"
                        style={{ fontVariationSettings: "'FILL' 1" }}
                      >
                        mic
                      </span>
                      <div className="flex flex-col">
                        <span className="text-[16px]">Audio message • {msg.duration}</span>
                        <span className="text-xs text-primary-fixed-dim italic">
                          &ldquo;{msg.original}&rdquo;
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-primary text-on-primary p-4 rounded-xl rounded-tr-none shadow-sm text-[16px]">
                      {msg.content}
                    </div>
                  )}
                  <span className="text-on-surface-variant text-[13px] font-semibold tracking-wider mr-1">
                    {msg.lang}
                  </span>
                </div>
              ) : (
                <div className="flex flex-col items-start gap-1 max-w-[85%] self-start">
                  <div className="bg-surface-container-lowest border border-outline-variant p-4 rounded-xl rounded-tl-none shadow-lg text-[16px] flex flex-col gap-3">
                    {msg.content === "Thinking..." ? (
                      <div className="flex items-center gap-2 text-outline">
                        <span className="material-symbols-outlined animate-spin">
                          progress_activity
                        </span>
                        Thinking...
                      </div>
                    ) : (
                      <>
                        <p className="text-on-surface whitespace-pre-line">{msg.content}</p>
                        <div className="flex items-center gap-2 mt-1">
                          <button className="material-symbols-outlined text-secondary text-[20px] hover:bg-surface-container transition-all p-1 rounded-full cursor-pointer">
                            volume_up
                          </button>
                          <button
                            onClick={() => navigator.clipboard.writeText(msg.content)}
                            className="material-symbols-outlined text-on-surface-variant text-[20px] hover:bg-surface-container transition-all p-1 rounded-full cursor-pointer"
                          >
                            content_copy
                          </button>
                          <button className="material-symbols-outlined text-on-surface-variant text-[20px] hover:bg-surface-container transition-all p-1 rounded-full cursor-pointer">
                            share
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                  <span className="text-on-surface-variant text-[13px] font-semibold tracking-wider ml-1">
                    {msg.lang}
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      </main>

      <div className="fixed bottom-0 left-0 w-full z-50">
        <div className="max-w-screen-xl mx-auto px-5 pb-4">
          <div className="bg-surface-container-lowest rounded-xl shadow-[0_-8px_24px_rgba(7,2,53,0.08)] border border-outline-variant p-3 flex items-end gap-3">
            <button className="text-on-surface-variant hover:bg-surface-container transition-all rounded-lg active:scale-90 p-3 cursor-pointer">
              <span className="material-symbols-outlined">add_circle</span>
            </button>
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                className="w-full bg-surface-container-low border-none rounded-lg py-3 px-4 text-on-surface text-[16px] focus:ring-2 focus:ring-secondary resize-none outline-none"
                placeholder="Type or speak a message..."
                rows={1}
              />
            </div>
            <div className="flex gap-1">
              <button className="p-3 bg-secondary-container text-on-secondary-container rounded-lg hover:bg-secondary-fixed transition-all active:scale-90 cursor-pointer">
                <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>
                  mic
                </span>
              </button>
              <button
                onClick={sendMessage}
                className="p-3 bg-primary text-on-primary rounded-lg hover:opacity-90 transition-all active:scale-90 cursor-pointer"
              >
                <span className="material-symbols-outlined">send</span>
              </button>
            </div>
          </div>
        </div>
        <BottomNavBar />
      </div>
    </>
  );
}
