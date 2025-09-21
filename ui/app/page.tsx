"use client";

import { useEffect, useRef, useState } from "react";
import type { ChatMessage, ChatRequestBody, ChatCompletionChunk } from "@/lib/types";

const DEFAULT_MODEL = "distilgpt2"; // or your actual model id (e.g., "sshleifer/tiny-gpt2")

type Status = "idle" | "sending" | "streaming";

export default function Page() {
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "system", content: "You are a concise assistant." }
  ]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, status]);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!input.trim() || status !== "idle") return;

    const userMsg: ChatMessage = { role: "user", content: input.trim() };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setInput("");
    setStatus("sending");

    // Prepare payload
    const payload: ChatRequestBody = {
        messages: nextMessages,
        max_tokens: 256,
        temperature: 0.7,
        top_p: 1.0,
        stream: true
    };

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const text = await resp.text();
        setError(text || `HTTP ${resp.status}`);
        setStatus("idle");
        return;
      }

      // Prepare assistant message placeholder for streaming
      let acc = "";
      const assistantIndex = nextMessages.length;
      setMessages((curr) => [...curr, { role: "assistant", content: "" }]);
      setStatus("streaming");

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        // Parse SSE lines
        const lines = buf.split("\n");
        // keep the last partial for next iteration
        buf = lines.pop() || "";
        for (const line of lines) {
          const s = line.trim();
          if (!s.startsWith("data:")) continue;
          const datum = s.slice(5).trim();
          if (datum === "[DONE]") {
            // finalize
            setStatus("idle");
            setMessages((curr) => {
              const copy = [...curr];
              copy[assistantIndex] = { role: "assistant", content: acc };
              return copy;
            });
            return;
          }
          // Parse chunk
          try {
            const obj: ChatCompletionChunk = JSON.parse(datum);
            const delta = obj.choices?.[0]?.delta || {};
            const piece = delta.content || "";
            if (piece) {
              acc += piece;
              setMessages((curr) => {
                const copy = [...curr];
                copy[assistantIndex] = { role: "assistant", content: acc };
                return copy;
              });
            }
          } catch {
            // ignore malformed
          }
        }
      }

      // If stream ended without [DONE], finalize anyway
      setStatus("idle");
      setMessages((curr) => {
        const copy = [...curr];
        copy[assistantIndex] = { role: "assistant", content: acc };
        return copy;
      });
    } catch (err: any) {
      setError(err?.message || String(err));
      setStatus("idle");
    }
  }

  function clearChat() {
    setMessages([{ role: "system", content: "You are a concise assistant." }]);
    setError(null);
  }

  return (
    <main style={{ maxWidth: 920, margin: "0 auto", padding: 20 }}>
      <h1 style={{ fontSize: 20, fontWeight: 600, marginBottom: 8 }}>LLM Chat</h1>
      <p style={{ color: "#666", marginBottom: 16 }}>
        Talking to: <code>{model}</code>
      </p>

      <div style={{
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        padding: 12,
        minHeight: 360,
        background: "#fff"
      }}>
        {messages
          .filter(m => m.role !== "system")
          .map((m, i) => (
            <div key={i} style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 8,
              marginBottom: 10
            }}>
              <div style={{
                fontSize: 12,
                fontWeight: 700,
                color: m.role === "user" ? "#2563eb" : "#16a34a",
                minWidth: 80,
                textTransform: "capitalize"
              }}>
                {m.role}
              </div>
              <div style={{
                whiteSpace: "pre-wrap",
                lineHeight: 1.5,
                background: m.role === "user" ? "#f8fafc" : "#f9fafb",
                border: "1px solid #eef2f7",
                borderRadius: 6,
                padding: 10,
                flex: 1
              }}>
                {m.content}
              </div>
            </div>
          ))}
        {status === "streaming" && (
          <div style={{ color: "#888", fontSize: 12, marginTop: 10 }}>
            streaming…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && (
        <div style={{ marginTop: 10, color: "#b91c1c" }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      <form onSubmit={send} style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <input
          type="text"
          value={model}
          readOnly
          disabled
          placeholder="model id (server-controlled)"
          style={{
            flex: "0 0 240px",
            border: "1px solid #e5e7eb",
            borderRadius: 6,
            padding: "8px 10px"
          }}
        />
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something…"
          rows={2}
          style={{
            flex: 1,
            border: "1px solid #e5e7eb",
            borderRadius: 6,
            padding: "8px 10px",
            resize: "vertical"
          }}
        />
        <button
          type="submit"
          disabled={status !== "idle"}
          style={{
            background: status === "idle" ? "#111827" : "#9ca3af",
            color: "white",
            border: 0,
            borderRadius: 8,
            padding: "0 16px",
            fontWeight: 600
          }}
        >
          {status === "idle" ? "Send" : "…"}
        </button>
        <button
          type="button"
          onClick={clearChat}
          style={{
            background: "#f3f4f6",
            color: "#374151",
            border: "1px solid #e5e7eb",
            borderRadius: 8,
            padding: "0 12px",
            fontWeight: 600
          }}
        >
          Clear
        </button>
      </form>

      <div style={{ marginTop: 12, fontSize: 12, color: "#6b7280" }}>
        Tip: Set <code>LLM_API_BASE</code> and (if needed) <code>LLM_API_KEY</code> in <code>.env.local</code>.
      </div>
    </main>
  );
}