"use client";

import Link from "next/link";

export default function TopAppBar() {
  return (
    // Mobile only — desktop uses the Sidebar
    <header className="bg-surface-bright shadow-sm fixed top-0 w-full z-40 md:hidden">
      <div className="flex justify-between items-center px-margin-mobile h-16">
        <Link href="/" className="flex items-center gap-2">
          <img src="/logo.png" alt="AI Stick Logo" className="h-9 w-auto" />
          <h1 className="text-display-lg text-primary leading-tight">AI Stick</h1>
        </Link>
        <div className="flex items-center gap-2">
          <Link
            href="/settings"
            className="p-2 rounded-full hover:bg-surface-container transition-colors"
            aria-label="Settings"
          >
            <span className="material-symbols-outlined text-primary text-[22px]">settings</span>
          </Link>
          <div className="w-9 h-9 rounded-full bg-primary-container flex items-center justify-center ring-2 ring-primary-container">
            <span className="material-symbols-outlined text-on-primary-container text-[20px]">person</span>
          </div>
        </div>
      </div>
    </header>
  );
}