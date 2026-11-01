import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const robotoMono = localFont({
  src: [
    { path: "../public/fonts/roboto-mono-regular.ttf", weight: "400", style: "normal" },
    { path: "../public/fonts/roboto-mono-bold.ttf", weight: "700", style: "normal" },
  ],
  variable: "--font-roboto-mono",
  display: "swap",
});

const redaction35 = localFont({
  src: "../public/fonts/redaction35-regular.ttf",
  variable: "--font-redaction35",
  weight: "400",
  style: "normal",
  display: "swap",
});

export const metadata: Metadata = {
  title: "25worlds — Turn video into a world",
  description: "Reconstruct a real environment in 3D from one video walkthrough. Upload, reconstruct, and explore your space.",
  applicationName: "25worlds",
  icons: { icon: "data:," },
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${robotoMono.variable} ${redaction35.variable}`}>{children}</body>
    </html>
  );
}
