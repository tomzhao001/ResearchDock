import { apiFetch, apiJson, readApiErrorMessage, websocketUrl } from "@/lib/api";

export type ChatTopic = {
  id: number;
  title: string;
  message_count: number;
  last_message_at: string | null;
  created_at: string;
  updated_at: string;
};

export type ChatCitation = {
  evidence_id?: string | null;
  chunk_id: number;
  paper_id: number;
  paper_title: string | null;
  source_url: string | null;
  snippet: string;
  score: number | null;
  support_score?: number | null;
  page_from: number | null;
  page_to: number | null;
  section_path: string | null;
  selection_reason?: string | null;
  claim_texts?: string[] | null;
};

export type ChatSufficiencyDecision = {
  is_sufficient: boolean;
  llm_sufficient: boolean | null;
  evidence_count: number;
  top_support_score: number;
  total_support_score: number;
  overall_support_score: number;
  min_support_score_threshold: number;
  min_total_support_score_threshold: number;
  policy_name: string | null;
  reason_codes: string[];
};

export type ChatMessage = {
  id: number;
  topic_id: number;
  role: string;
  content: string;
  model: string | null;
  answer_mode: string | null;
  used_knowledge_base: boolean;
  citations: ChatCitation[];
  sufficiency_decision?: ChatSufficiencyDecision | null;
  missing_information?: string | null;
  response_kind?: string | null;
  attribution_status?: string | null;
  status_message?: string | null;
  status_detail?: string | null;
  created_at: string;
};

export type ChatProgressEvent = {
  type?: "chat-progress";
  topic_id?: number;
  phase?: string;
  status?: string;
  message: string;
  detail?: string | null;
  created_at: string;
};

type AssistantStartPayload = {
  answer_mode: string | null;
  used_knowledge_base: boolean;
  sufficiency_decision?: ChatSufficiencyDecision | null;
  missing_information?: string | null;
  response_kind?: string | null;
  attribution_status?: string | null;
  status_message?: string | null;
  status_detail?: string | null;
};

type StreamTopicMessageOptions = {
  signal?: AbortSignal;
  onUserMessage?: (message: ChatMessage) => void;
  onAssistantStart?: (payload: AssistantStartPayload) => void;
  onAssistantDelta?: (delta: string) => void;
  onAssistantComplete?: (message: ChatMessage) => void;
};

async function apiList<T>(input: string): Promise<T[]> {
  const payload = await apiJson<{ items: T[] }>(input);
  return payload.items ?? [];
}

export function fetchChatTopics(): Promise<ChatTopic[]> {
  return apiList<ChatTopic>("/api/chat/topics");
}

export function createChatTopic(title?: string | null): Promise<ChatTopic> {
  return apiJson<ChatTopic>("/api/chat/topics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title ?? null }),
  });
}

export function fetchTopicMessages(topicId: number): Promise<ChatMessage[]> {
  return apiList<ChatMessage>(`/api/chat/topics/${topicId}/messages`);
}

export function subscribeChatProgressEvents(options: {
  topicId: number;
  onEvent: (event: ChatProgressEvent) => void;
}): () => void {
  const socket = new WebSocket(websocketUrl(`/api/ws/chat-progress/${options.topicId}`));
  socket.onmessage = (message) => {
    try {
      options.onEvent(JSON.parse(message.data) as ChatProgressEvent);
    } catch {
      return;
    }
  };
  return () => socket.close();
}

export async function streamTopicMessage(
  topicId: number,
  message: string,
  options?: StreamTopicMessageOptions
): Promise<void> {
  const response = await apiFetch(`/api/chat/topics/${topicId}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
    signal: options?.signal,
  });
  if (!response.ok) {
    throw new Error(await readApiErrorMessage(response));
  }
  if (!response.body) {
    throw new Error("聊天流式输出失败");
  }
  await readSseStream(response.body, options);
}

async function readSseStream(
  stream: ReadableStream<Uint8Array>,
  options?: StreamTopicMessageOptions
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        dispatchSseBlocks(buffer, options, true);
        return;
      }
      buffer += decoder.decode(value, { stream: true });
      buffer = dispatchSseBlocks(buffer, options, false);
    }
  } finally {
    reader.releaseLock();
  }
}

function dispatchSseBlocks(
  buffer: string,
  options: StreamTopicMessageOptions | undefined,
  flush: boolean
): string {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");
  const rest = flush ? "" : (parts.pop() ?? "");
  for (const block of parts) {
    if (!block.trim()) {
      continue;
    }
    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) {
        eventName = line.slice("event:".length).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice("data:".length).trimStart());
      }
    }
    if (dataLines.length === 0) {
      continue;
    }
    const payload = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
    switch (eventName) {
      case "user_message":
        options?.onUserMessage?.(payload.user_message as ChatMessage);
        break;
      case "assistant_start":
        options?.onAssistantStart?.(payload as AssistantStartPayload);
        break;
      case "assistant_delta":
        options?.onAssistantDelta?.(String(payload.delta ?? ""));
        break;
      case "assistant_complete":
        options?.onAssistantComplete?.(payload.assistant_message as ChatMessage);
        break;
      case "error":
        throw new Error(typeof payload.detail === "string" ? payload.detail : "聊天流式输出失败");
      case "done":
        break;
      default:
        break;
    }
  }
  return rest;
}
