"use client";

export default function TopAppBar() {
  return (
    <header className="bg-surface sticky top-0 z-40 shadow-sm">
      <div className="flex justify-between items-center w-full px-5 py-2 max-w-screen-xl mx-auto">
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-primary text-[32px]">stylus</span>
          <h1 className="text-[32px] font-bold text-primary tracking-tight leading-tight">
            AI Stick
          </h1>
        </div>
        <div className="flex items-center gap-4">
          <button className="active:scale-95 duration-200 hover:bg-surface-container transition-colors rounded-full p-1 cursor-pointer">
            <span className="material-symbols-outlined text-primary">settings</span>
          </button>
          <div className="h-10 w-10 rounded-full bg-primary-fixed overflow-hidden ring-2 ring-primary-container flex items-center justify-center text-primary font-bold">
            <span className="material-symbols-outlined text-on-primary-fixed">person</span>
          </div>
        </div>
      </div>
    </header>
  );
}
