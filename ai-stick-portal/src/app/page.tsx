"use client";

import { useRouter } from "next/navigation";
import TopAppBar from "@/components/TopAppBar";
import BottomNavBar from "@/components/BottomNavBar";

const CARDS = [
  {
    title: "Translate",
    desc: "Quick English to Runyoro text conversion.",
    icon: "translate",
    bg: "bg-secondary-container text-on-secondary-container",
    arrowColor: "text-secondary",
    href: "/translate",
  },
  {
    title: "AI Chat",
    desc: "Conversational assistance & insights.",
    icon: "chat_bubble",
    bg: "bg-primary-container text-on-primary-container",
    arrowColor: "text-primary",
    href: "/chat",
  },
  {
    title: "Editor",
    desc: "Full document localization & layout.",
    icon: "description",
    bg: "bg-secondary-fixed text-on-secondary-fixed-variant",
    arrowColor: "text-secondary",
    href: "/editor",
  },
  {
    title: "Voice Mode",
    desc: "Speak and listen in real-time.",
    icon: "mic",
    bg: "bg-primary-fixed text-on-primary-fixed-variant",
    arrowColor: "text-primary",
    href: "#",
  },
];

const ACTIVITIES = [
  {
    icon: "history",
    iconBg: "bg-surface-container-high text-primary-container",
    title: "Regional Agriculture Report",
    subtitle: "Last Edited • 2h ago",
  },
  {
    icon: "translate",
    iconBg: "bg-surface-container-high text-secondary",
    title: "Greeting Protocol (En → Run)",
    subtitle: "Recent Translation • 5h ago",
  },
  {
    icon: "chat_bubble",
    iconBg: "bg-surface-container-high text-primary",
    title: "Idiomatic Expressions Inquiry",
    subtitle: "Chat Archive • Yesterday",
  },
];

export default function HomePage() {
  const router = useRouter();

  return (
    <>
      <TopAppBar />
      <main className="max-w-screen-xl mx-auto px-5 py-6 pb-32">
        <section className="mb-10">
          <div className="flex flex-col gap-1">
            <h2 className="text-[24px] font-semibold tracking-tight text-primary">
              Good Morning, Linguist
            </h2>
            <p className="text-[16px] text-on-surface-variant">
              What would you like to build today?
            </p>
          </div>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
          {CARDS.map((card) => (
            <div
              key={card.title}
              onClick={() => card.href !== "#" && router.push(card.href)}
              className="bg-surface-container-lowest rounded-xl p-6 shadow-sm hover:shadow-md transition-all active:scale-95 cursor-pointer flex flex-col gap-4 border border-outline-variant/10"
            >
              <div
                className={`h-12 w-12 rounded-lg flex items-center justify-center ${card.bg}`}
              >
                <span className="material-symbols-outlined text-[24px]">
                  {card.icon}
                </span>
              </div>
              <div>
                <h3 className="text-[20px] font-semibold text-primary">
                  {card.title}
                </h3>
                <p className="text-[14px] text-on-surface-variant">
                  {card.desc}
                </p>
              </div>
              <div className="mt-auto flex justify-end">
                <span className={`material-symbols-outlined ${card.arrowColor}`}>
                  arrow_forward
                </span>
              </div>
            </div>
          ))}
        </section>

        <div className="w-full h-1 bg-surface-container rounded-full overflow-hidden mb-10">
          <div className="h-full bg-secondary w-1/3 animate-pulse" />
        </div>

        <section>
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-[24px] font-semibold text-primary">
              Recent Activity
            </h3>
            <button className="text-secondary text-[13px] font-semibold tracking-wider uppercase flex items-center gap-1">
              View All
              <span className="material-symbols-outlined text-[16px]">
                chevron_right
              </span>
            </button>
          </div>
          <div className="bg-surface-container-lowest rounded-xl shadow-sm overflow-hidden border border-outline-variant/10">
            {ACTIVITIES.map((item, i) => (
              <div
                key={i}
                className="p-4 flex items-center justify-between border-b border-outline-variant/10 hover:bg-surface-container-low transition-colors cursor-pointer group last:border-b-0"
              >
                <div className="flex items-center gap-4">
                  <div
                    className={`h-10 w-10 rounded-lg flex items-center justify-center ${item.iconBg}`}
                  >
                    <span className="material-symbols-outlined">{item.icon}</span>
                  </div>
                  <div>
                    <p className="font-semibold text-primary">{item.title}</p>
                    <p className="text-[13px] text-on-surface-variant">
                      {item.subtitle}
                    </p>
                  </div>
                </div>
                <span className="material-symbols-outlined text-outline group-hover:text-primary transition-colors">
                  chevron_right
                </span>
              </div>
            ))}
          </div>
        </section>
      </main>
      <BottomNavBar />
    </>
  );
}
