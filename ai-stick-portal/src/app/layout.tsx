import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "AI Stick — Runyoro-Rutooro Language Portal",
  description:
    "Bidirectional Runyoro-Rutooro ↔ English Neural Machine Translation. " +
    "Offline AI tools: translator, dictionary, chat, editor. BLEU 18.77.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,0..1&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-background text-on-background min-h-screen font-sans antialiased flex overflow-x-hidden">
        {/* Desktop sidebar */}
        <Sidebar />
        {/* Main content shifts right on md+ */}
        <div className="flex-1 md:ml-64 flex flex-col min-h-screen">
          {children}
        </div>
      </body>
    </html>
  );
}
