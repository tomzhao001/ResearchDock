import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
  beforeEach(() => {
    vi.clearAllMocks();
  });

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

  it("展示 sufficiency 判定与不足原因", async () => {
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
        id: 12,
        topic_id: 1,
        role: "assistant",
        content: "知识库中未找到确切依据。以下为通用补充。",
        model: "fallback-model",
        answer_mode: "kb_insufficient_evidence",
        used_knowledge_base: false,
        missing_information: "证据能解释术语，但不足以覆盖背景问题。",
        sufficiency_decision: {
          is_sufficient: false,
          llm_sufficient: false,
          evidence_count: 1,
          top_support_score: 0.82,
          total_support_score: 0.82,
          overall_support_score: 0.82,
          min_support_score_threshold: 0.45,
          min_total_support_score_threshold: 0.75,
          policy_name: "relaxed_chat",
          reason_codes: ["llm_marked_insufficient"],
        },
        created_at: "2026-05-19T08:00:00Z",
        citations: [],
      },
    ]);

    render(<ChatPanel />);

    await waitFor(() => expect(fetchChatTopics).toHaveBeenCalled());
    await waitFor(() => expect(fetchTopicMessages).toHaveBeenCalledWith(1));
    await screen.findByText("知识库中未找到确切依据。以下为通用补充。");

    expect(screen.getByText("保守回答（含通用补充）")).toBeInTheDocument();
    expect(screen.getByText("系统判定：证据不足，请自行判断。")).toBeInTheDocument();
    expect(screen.getByText("证据能解释术语，但不足以覆盖背景问题。")).toBeInTheDocument();
  });
});
