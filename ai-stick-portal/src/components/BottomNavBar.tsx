"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { label: "Home",       icon: "home",         href: "/",           fillIcon: "home"         },
  { label: "Translate",  icon: "g_translate",   href: "/translate",  fillIcon: "g_translate"  },
  { label: "Lens",       icon: "photo_camera",  href: "/lens",       fillIcon: "photo_camera" },
  { label: "Chat",       icon: "chat_bubble",   href: "/chat",       fillIcon: "chat_bubble"  },
  { label: "Dictionary", icon: "menu_book",     href: "/dictionary", fillIcon: "menu_book"    },
] as const;

export default function BottomNavBar() {
  const pathname = usePathname();

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-[9999] md:hidden" style={{ position: 'fixed' }}>
      <div className="bg-white shadow-[0_-4px_12px_rgba(0,0,0,0.1)] border-t border-gray-200">
        <div className="flex justify-around items-center px-2 py-2 pb-2">
          {NAV_ITEMS.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex flex-col items-center gap-0.5 transition-all active:scale-90 px-2 py-1 ${
                  isActive
                    ? "bg-primary-fixed text-on-primary-fixed rounded-full px-4"
                    : "text-on-surface-variant opacity-70 hover:bg-surface-container-high rounded-xl"
                }`}
              >
                <span
                  className="material-symbols-outlined text-[22px]"
                  style={isActive ? { fontVariationSettings: "'FILL' 1" } : undefined}
                >
                  {isActive ? item.fillIcon : item.icon}
                </span>
                <span className="text-label-sm">{item.label}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}