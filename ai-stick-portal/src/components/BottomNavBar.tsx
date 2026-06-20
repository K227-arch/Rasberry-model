"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS: {
  label: string;
  icon: string;
  href: string;
  fillIcon?: string;
}[] = [
  {
    label: "Home",
    icon: "home",
    href: "/",
    fillIcon: "home",
  },
  {
    label: "Translate",
    icon: "translate",
    href: "/translate",
  },
  {
    label: "Chat",
    icon: "chat_bubble",
    href: "/chat",
  },
  {
    label: "Editor",
    icon: "description",
    href: "/editor",
  },
];

export default function BottomNavBar() {
  const pathname = usePathname();

  return (
    <nav className="fixed bottom-0 left-0 w-full z-50 bg-surface-container-lowest shadow-[0_-8px_24px_rgba(7,2,53,0.08)] rounded-t-xl md:hidden">
      <div className="flex justify-around items-center px-4 py-3">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex flex-col items-center gap-1 transition-all active:scale-90 ${
                isActive
                  ? "bg-secondary-container text-on-secondary-container rounded-full px-5 py-1"
                  : "text-on-surface-variant opacity-70 hover:bg-surface-container-high rounded-xl p-1"
              }`}
            >
              <span
                className="material-symbols-outlined"
                style={
                  isActive && item.fillIcon
                    ? { fontVariationSettings: "'FILL' 1" }
                    : undefined
                }
              >
                {item.icon}
              </span>
              <span className="text-[11px] font-semibold tracking-wider uppercase">
                {item.label}
              </span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
