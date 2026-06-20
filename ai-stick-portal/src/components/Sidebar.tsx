"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { label: "Home",        icon: "home",        href: "/"           },
  { label: "Translate",   icon: "g_translate",  href: "/translate"  },
  { label: "Chat",        icon: "chat_bubble",  href: "/chat"       },
  { label: "Editor",      icon: "edit_note",    href: "/editor"     },
  { label: "Dictionary",  icon: "menu_book",    href: "/dictionary" },
  { label: "History",     icon: "history",      href: "/history"    },
  { label: "Settings",    icon: "settings",     href: "/settings"   },
] as const;

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden md:flex flex-col w-64 bg-surface-container-lowest border-r border-outline-variant/30 fixed top-0 left-0 h-full z-50 premium-shadow">
      {/* Logo */}
      <div className="flex items-center gap-3 px-6 h-16 border-b border-outline-variant/30">
        <span className="material-symbols-outlined text-primary text-[32px]">stylus</span>
        <div>
          <h1 className="text-display-lg text-primary leading-tight">AI Stick</h1>
          <p className="text-label-sm text-on-surface-variant -mt-0.5">Offline Intelligence</p>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all group ${
                isActive
                  ? "bg-primary-fixed text-on-primary-fixed font-semibold"
                  : "text-on-surface-variant hover:bg-surface-container hover:text-on-surface"
              }`}
            >
              <span
                className="material-symbols-outlined text-[22px]"
                style={isActive ? { fontVariationSettings: "'FILL' 1" } : undefined}
              >
                {item.icon}
              </span>
              <span className="text-body-md">{item.label}</span>
              {isActive && (
                <span className="ml-auto w-1.5 h-1.5 rounded-full bg-primary" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-outline-variant/30">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-label-sm text-on-surface-variant">Model Ready</span>
        </div>
        <p className="text-label-sm text-on-surface-variant mt-1">
          BLEU 18.77 · chrF++ 22.53
        </p>
      </div>
    </aside>
  );
}
