import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

// Inter and Material Symbols are both served from /public/fonts/
// Zero Google CDN requests — fully offline-capable

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
    <html lang="en" className="h-full" suppressHydrationWarning>
      <head>
        <link rel="icon" type="image/png" href="/favicon.png" />
        <link rel="apple-touch-icon" href="/logo.png" />
        {/* Preload fonts — eliminates FOUT and icon flash */}
        <link rel="preload" href="/fonts/Inter.woff2" as="font" type="font/woff2" crossOrigin="anonymous" />
        <link rel="preload" href="/fonts/MaterialSymbolsOutlined.woff2" as="font" type="font/woff2" crossOrigin="anonymous" />
      </head>
      <body className="bg-background text-on-background min-h-screen font-sans antialiased flex overflow-x-hidden">
        <Sidebar />
        <div className="flex-1 md:ml-64 flex flex-col min-h-screen">
          {children}
        </div>
      </body>
    </html>
  );
}
