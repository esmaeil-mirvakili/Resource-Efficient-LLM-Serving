export const metadata = {
  title: "LLM Chat",
  description: "Chat with your local LLM server"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}