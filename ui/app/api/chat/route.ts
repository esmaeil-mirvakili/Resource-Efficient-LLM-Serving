import { NextRequest } from "next/server";

const API_BASE = process.env.LLM_API_BASE!;
const API_KEY = process.env.LLM_API_KEY || "";

export const dynamic = "force-dynamic"; // no caching

export async function POST(req: NextRequest) {
  if (!API_BASE) {
    return new Response(
      JSON.stringify({ error: { message: "LLM_API_BASE not configured" } }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }

  const body = await req.json();

  const headers: Record<string, string> = {
    "content-type": "application/json",
  };
  if (API_KEY) {
    headers["authorization"] = `Bearer ${API_KEY}`;
  }

  // Forward to FastAPI
  const upstream = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  // If not streaming, just pipe JSON
  const contentType = upstream.headers.get("content-type") || "";
  const isSSE = contentType.includes("text/event-stream");

  if (!isSSE) {
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "content-type": contentType || "application/json" },
    });
  }

  // Streaming SSE: re-emit as-is to the client
  const stream = new ReadableStream({
    async start(controller) {
      const reader = upstream.body!.getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          controller.enqueue(value);
        }
      } catch (e) {
        controller.error(e);
      } finally {
        reader.releaseLock();
        controller.close();
      }
    },
  });

  return new Response(stream, {
    status: upstream.status,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
      // Allow buffering proxies like Nginx to flush
      "x-accel-buffering": "no",
    },
  });
}