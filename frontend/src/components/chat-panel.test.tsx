import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

  it("以引用标签展示 citation，并支持 tooltip 和详情弹窗", async () => {
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
            support_score: 0.88,
            page_from: 3,
            page_to: 4,
            section_path: "Results > Table 1",
            selection_reason: "表 1 直接提供了结果摘要。",
            claim_texts: ["结果表明实验组更优。"],
          },
        ],
      },
    ]);

    const user = userEvent.setup();

    render(<ChatPanel />);

    await waitFor(() => expect(fetchChatTopics).toHaveBeenCalled());
    await waitFor(() => expect(fetchTopicMessages).toHaveBeenCalledWith(1));

    const citationTag = screen.getByRole("button", { name: "[1]" });
    expect(citationTag).toBeInTheDocument();

    await user.hover(citationTag);
    expect(await screen.findByRole("tooltip")).toHaveTextContent("Demo Paper");
    expect(screen.getByRole("tooltip")).toHaveTextContent("Results section evidence snippet.");

    await user.click(citationTag);
    const dialog = await screen.findByRole("dialog", { name: "引用 [1]" });
    expect(within(dialog).getByText("章节：Results > Table 1")).toBeInTheDocument();
    expect(within(dialog).getByText("页码：3-4")).toBeInTheDocument();
    expect(within(dialog).getByText("Results section evidence snippet.")).toBeInTheDocument();
    expect(within(dialog).getByText("表 1 直接提供了结果摘要。")).toBeInTheDocument();
    expect(within(dialog).getByText("结果表明实验组更优。")).toBeInTheDocument();
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
        content: "现有材料可以说明该术语含义，但更广泛的背景问题在当前材料中无法确认。",
        model: "fallback-model",
        answer_mode: "kb_insufficient_evidence",
        used_knowledge_base: false,
        response_kind: "evidence_backed_fallback",
        attribution_status: "partial_evidence",
        status_message: "已找到部分相关材料，以下为基于现有材料的归纳总结。",
        status_detail: "证据能解释术语，但不足以覆盖背景问题。",
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
    await screen.findByText("现有材料可以说明该术语含义，但更广泛的背景问题在当前材料中无法确认。");

    expect(screen.getByText("基于材料的保守总结")).toBeInTheDocument();
    expect(screen.getByText("系统判定：仅找到部分相关材料，以下为保守总结。")).toBeInTheDocument();
    expect(screen.getByText("证据能解释术语，但不足以覆盖背景问题。")).toBeInTheDocument();
  });

  it("渲染助手回答中的 Markdown", async () => {
    fetchChatTopics.mockResolvedValue([
      {
        id: 1,
        title: "Markdown 话题",
        message_count: 1,
        last_message_at: "2026-05-19T08:00:00Z",
        created_at: "2026-05-19T08:00:00Z",
        updated_at: "2026-05-19T08:00:00Z",
      },
    ]);
    fetchTopicMessages.mockResolvedValue([
      {
        id: 21,
        topic_id: 1,
        role: "assistant",
        content: "# 结论\n\n- **核心发现**\n- `code sample`\n\n[查看原文](https://example.com/report)",
        model: "markdown-model",
        answer_mode: "knowledge_base",
        used_knowledge_base: true,
        created_at: "2026-05-19T08:00:00Z",
        citations: [],
      },
    ]);

    render(<ChatPanel />);

    await waitFor(() => expect(fetchChatTopics).toHaveBeenCalled());
    await waitFor(() => expect(fetchTopicMessages).toHaveBeenCalledWith(1));

    expect(screen.getByRole("heading", { name: "结论" })).toBeInTheDocument();
    expect(screen.getByText("核心发现").tagName).toBe("STRONG");
    expect(screen.getByText("code sample").tagName).toBe("CODE");
    expect(screen.getByRole("link", { name: "查看原文" })).toHaveAttribute("href", "https://example.com/report");
  });

  it("按 topicId 隔离流式状态，切换话题后不串流", async () => {
    const user = userEvent.setup();
    const topicMessagesStore: Record<number, Array<Record<string, unknown>>> = {
      1: [],
      2: [],
    };

    fetchChatTopics.mockResolvedValue([
      {
        id: 1,
        title: "话题 A",
        message_count: 0,
        last_message_at: "2026-05-19T08:00:00Z",
        created_at: "2026-05-19T08:00:00Z",
        updated_at: "2026-05-19T08:00:00Z",
      },
      {
        id: 2,
        title: "话题 B",
        message_count: 0,
        last_message_at: "2026-05-19T08:00:00Z",
        created_at: "2026-05-19T08:00:00Z",
        updated_at: "2026-05-19T08:00:00Z",
      },
    ]);
    fetchTopicMessages.mockImplementation(async (topicId: number) => (topicMessagesStore[topicId] ?? []) as never[]);

    let finishStream: (() => void) | null = null;
    streamTopicMessage.mockImplementation(
      async (
        topicId: number,
        message: string,
        options?: {
          onUserMessage?: (payload: Record<string, unknown>) => void;
          onAssistantStart?: (payload: Record<string, unknown>) => void;
          onAssistantDelta?: (payload: string) => void;
          onAssistantComplete?: (payload: Record<string, unknown>) => void;
        }
      ) =>
        await new Promise<void>((resolve) => {
          const userMessage = {
            id: 100 + topicId,
            topic_id: topicId,
            role: "user",
            content: message,
            model: null,
            answer_mode: null,
            used_knowledge_base: false,
            created_at: "2026-05-19T08:00:00Z",
            citations: [],
          };
          topicMessagesStore[topicId] = [...(topicMessagesStore[topicId] ?? []), userMessage];
          options?.onUserMessage?.(userMessage);
          options?.onAssistantStart?.({
            answer_mode: "knowledge_base",
            used_knowledge_base: true,
            sufficiency_decision: null,
            missing_information: null,
            response_kind: "grounded_rag",
            attribution_status: "grounded",
            status_message: null,
            status_detail: null,
          });
          options?.onAssistantDelta?.("话题 A 的流式草稿");
          finishStream = () => {
            const assistantMessage = {
              id: 200 + topicId,
              topic_id: topicId,
              role: "assistant",
              content: "话题 A 的最终答案",
              model: "stream-model",
              answer_mode: "knowledge_base",
              used_knowledge_base: true,
              created_at: "2026-05-19T08:01:00Z",
              citations: [],
            };
            topicMessagesStore[topicId] = [...(topicMessagesStore[topicId] ?? []), assistantMessage];
            options?.onAssistantComplete?.(assistantMessage);
            resolve();
          };
        })
    );

    render(<ChatPanel />);

    await waitFor(() => expect(fetchChatTopics).toHaveBeenCalled());
    await waitFor(() => expect(fetchTopicMessages).toHaveBeenCalledWith(1));

    await user.type(screen.getByRole("textbox"), "请总结话题 A");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText("话题 A 的流式草稿")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "发送中..." })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /话题 B/ }));
    await waitFor(() => expect(fetchTopicMessages).toHaveBeenCalledWith(2));

    expect(screen.queryByText("话题 A 的流式草稿")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "发送中..." })).not.toBeInTheDocument();
    expect(screen.getByText("当前话题还没有消息，试着问一个和已归档论文相关的问题。")).toBeInTheDocument();

    finishStream?.();

    await waitFor(() => expect(fetchChatTopics).toHaveBeenCalledTimes(2));
    expect(screen.queryByText("话题 A 的最终答案")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /话题 A/ }));
    await waitFor(() => expect(fetchTopicMessages).toHaveBeenCalledTimes(3));
    expect(await screen.findByText("话题 A 的最终答案")).toBeInTheDocument();
  });
});
