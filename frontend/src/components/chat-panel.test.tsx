import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ChatPanel } from "@/components/chat-panel";

const createChatTopic = vi.fn();
const fetchChatTopics = vi.fn();
const fetchTopicMessages = vi.fn();
const streamTopicMessage = vi.fn();
const subscribeChatProgressEvents = vi.fn(() => vi.fn());

vi.mock("@/lib/chat", () => ({
  createChatTopic: (...args: unknown[]) => createChatTopic(...args),
  fetchChatTopics: (...args: unknown[]) => fetchChatTopics(...args),
  fetchTopicMessages: (...args: unknown[]) => fetchTopicMessages(...args),
  streamTopicMessage: (...args: unknown[]) => streamTopicMessage(...args),
  subscribeChatProgressEvents: (...args: unknown[]) => subscribeChatProgressEvents(...args),
}));

describe("ChatPanel", () => {
  it("展示 citation 的章节和页码信息", async () => {
    fetchChatTopics.mockResolvedValue([
      {
        id: 1,
        title: "测试话题",
        message_count: 1,
        last_message_at: "2026-05-19T08:00:00Z",
        created_at: "2026-05-19T08:00:00Z",
        updated_at: "2026-05-19T08:00:00Z",
      },
    ]);
    fetchTopicMessages.mockResolvedValue([
      {
        id: 11,
        topic_id: 1,
        role: "assistant",
        content: "这是回答。",
        model: "test-model",
        answer_mode: "knowledge_base",
        used_knowledge_base: true,
        created_at: "2026-05-19T08:00:00Z",
        citations: [
          {
            chunk_id: 101,
            paper_id: 7,
            paper_title: "Demo Paper",
            source_url: "https://example.com/paper",
            snippet: "Results section evidence snippet.",
            score: 0.92,
            page_from: 3,
            page_to: 4,
            section_path: "Results > Table 1",
          },
        ],
      },
    ]);

    render(<ChatPanel />);

    await waitFor(() => expect(fetchChatTopics).toHaveBeenCalled());
    await waitFor(() => expect(fetchTopicMessages).toHaveBeenCalledWith(1));

    expect(screen.getByText("章节：Results > Table 1")).toBeInTheDocument();
    expect(screen.getByText("页码：3-4")).toBeInTheDocument();
    expect(screen.getByText("Results section evidence snippet.")).toBeInTheDocument();
  });
});
