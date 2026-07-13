import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Seer — KYC document verification",
  description:
    "Document localization, OCR, face verification and tamper forensics on a CPU latency budget.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
