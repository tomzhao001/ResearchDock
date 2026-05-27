"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bot, Database, ExternalLink, LoaderCircle, MessageSquarePlus, MessagesSquare, Send, User2 } from "lucide-react";

import { ChatMarkdown } from "@/components/chat-markdown";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog } from "@/components/ui/dialog";
import { Tooltip } from "@/components/ui/tooltip";
import {
  createChatTopic,
  fetchChatTopics,
  fetchTopicMessages,
  streamTopicMessage,
  subscribeChatProgressEvents,
  type ChatCitation,
  type ChatProgressEvent,
  type ChatMessage,
  type ChatSufficiencyDecision,
  type ChatTopic,
} from "@/lib/chat";
import { cn } from "@/lib/utils";

type StreamingAssistantDraft = {
  content: string;
  answerMode: string | null;
  usedKnowledgeBase: boolean;
  sufficiencyDecision: ChatSufficiencyDecision | null;
  missingInformation: string | null;
  responseKind: string | null;
  attributionStatus: string | null;
  statusMessage: string | null;
  statusDetail: string | null;
};

type TopicStreamState = {
  submitting: boolean;
  progressEvents: ChatProgressEvent[];
  streamingAssistantDraft: StreamingAssistantDraft | null;
};

type CitationDialogState = {
  citation: ChatCitation;
  index: number;
} | null;

const EMPTY_STREAM_STATE: TopicStreamState = {
  submitting: false,
  progressEvents: [],
  streamingAssistantDraft: null,
};

function formatTime(value: string | null): string {
  if (!value) return "-";
  return new Date(value).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatPageRange(citation: ChatCitation): string | null {
  if (citation.page_from === null) {
    return null;
  }
  if (citation.page_to !== null && citation.page_to !== citation.page_from) {
    return `${citation.page_from}-${citation.page_to}`;
  }
  return String(citation.page_from);
}

function summarizeCitation(citation: ChatCitation, maxLength = 120): string {
  const text = citation.snippet.replace(/\s+/g, " ").trim();
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 1)}...`;
}

function getTopicStreamState(
  topicStreamStates: Record<number, TopicStreamState>,
  topicId: number | null
): TopicStreamState {
  if (!topicId) {
    return EMPTY_STREAM_STATE;
  }
  return topicStreamStates[topicId] ?? EMPTY_STREAM_STATE;
}

function getAnswerBadge(responseKind?: string | null, usedKnowledgeBase?: boolean): string {
  switch (responseKind) {
    case "metadata_answer":
      return "元数据回答";
    case "evidence_backed_fallback":
      return "基于材料的保守总结";
    case "general_fallback":
      return "通用知识补充";
    case "grounded_rag":
      return "知识库回答";
    default:
      return usedKnowledgeBase ? "知识库回答" : "保守回答";
  }
}

function getBadgeTone(responseKind?: string | null, usedKnowledgeBase?: boolean): string {
  if (responseKind === "metadata_answer") {
    return "bg-sky-500/10 text-sky-700 ring-sky-500/20";
  }
  if (responseKind === "evidence_backed_fallback") {
    return "bg-amber-500/10 text-amber-700 ring-amber-500/20";
  }
  if (responseKind === "general_fallback") {
    return "bg-rose-500/10 text-rose-700 ring-rose-500/20";
  }
  return usedKnowledgeBase
    ? "bg-emerald-500/10 text-emerald-700 ring-emerald-500/20"
    : "bg-amber-500/10 text-amber-700 ring-amber-500/20";
}

function summarizeReasonCodes(decision: ChatSufficiencyDecision | null | undefined): string | null {
  if (!decision?.reason_codes?.length) {
    return null;
  }
  const mapping: Record<string, string> = {
    sufficient: "证据较充足",
    no_evidence_selected: "未选中可直接支撑回答的证据",
    top_support_below_threshold: "最高证据支持度偏低",
    total_support_below_threshold: "总体证据支持度偏低",
    llm_marked_insufficient: "模型判定证据仍不足",
    llm_marked_insufficient_advisory: "模型曾判定不足，已按对话模式放宽",
    relaxed_chat_policy: "当前使用对话放宽模式",
  };
  const labels = decision.reason_codes
    .map((code) => mapping[code] ?? code)
    .filter((value, index, list) => value && list.indexOf(value) === index);
  return labels.join("；") || null;
}

function renderSufficiencyHint(
  responseKind: string | null | undefined,
  attributionStatus: string | null | undefined,
  statusMessage: string | null | undefined,
  statusDetail: string | null | undefined,
  decision: ChatSufficiencyDecision | null | undefined,
  missingInformation?: string | null
): { title: string; detail: string | null } | null {
  if (!decision && !attributionStatus && !statusMessage) {
    return null;
  }
  const titleMapping: Record<string, string> = {
    grounded: "系统判定：材料可完整支撑，请结合引用核验。",
    partial_evidence: "系统判定：仅找到部分相关材料，以下为保守总结。",
    verification_failed: "系统判定：已找到相关材料，但完整答案未通过最终校验。",
    scope_empty: "系统判定：按当前筛选条件未匹配到论文。",
    no_usable_evidence: "系统判定：未找到可直接支撑问题的材料。",
    metadata_only: "系统判定：这是元数据/范围回答，不属于正文证据归因回答。",
  };
  const title =
    (attributionStatus ? titleMapping[attributionStatus] : null) ??
    statusMessage ??
    (decision?.is_sufficient ? "系统判定：证据较充足，请结合引用自行核验。" : "系统判定：证据不足，请自行判断。");
  const reasonSummary = summarizeReasonCodes(decision);
  const detail =
    statusDetail ||
    (attributionStatus === "partial_evidence" || responseKind === "evidence_backed_fallback"
      ? missingInformation || reasonSummary
      : decision?.is_sufficient
        ? reasonSummary
        : missingInformation || reasonSummary);
  return { title, detail: detail || null };
}

export function ChatPanel() {
  const [topics, setTopics] = useState<ChatTopic[]>([]);
  const [topicMessages, setTopicMessages] = useState<Record<number, ChatMessage[]>>({});
  const [selectedTopicId, setSelectedTopicId] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [loadingTopics, setLoadingTopics] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [creatingTopic, setCreatingTopic] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [topicStreamStates, setTopicStreamStates] = useState<Record<number, TopicStreamState>>({});
  const [citationDialogState, setCitationDialogState] = useState<CitationDialogState>(null);
  const messageViewportRef = useRef<HTMLDivElement | null>(null);
  const activeTopicIdRef = useRef<number | null>(null);
  const streamAbortControllersRef = useRef<Map<number, AbortController>>(new Map());
  const progressUnsubscribersRef = useRef<Map<number, () => void>>(new Map());

  const selectedTopic = useMemo(
    () => topics.find((topic) => topic.id === selectedTopicId) ?? null,
    [selectedTopicId, topics]
  );
  const messages = useMemo(
    () => (selectedTopicId ? topicMessages[selectedTopicId] ?? [] : []),
    [selectedTopicId, topicMessages]
  );
  const currentStreamState = useMemo(
    () => getTopicStreamState(topicStreamStates, selectedTopicId),
    [selectedTopicId, topicStreamStates]
  );

  const upsertTopic = useCallback((nextTopic: ChatTopic) => {
    setTopics((current) => {
      const deduped = current.filter((topic) => topic.id !== nextTopic.id);
      return [nextTopic, ...deduped].sort((left, right) => {
        const leftTime = new Date(left.updated_at).getTime();
        const rightTime = new Date(right.updated_at).getTime();
        return rightTime - leftTime;
      });
    });
  }, []);

  const appendMessageToTopic = useCallback((topicId: number, nextMessage: ChatMessage) => {
    setTopicMessages((current) => {
      const items = current[topicId] ?? [];
      const existingIndex = items.findIndex((item) => item.id === nextMessage.id);
      if (existingIndex >= 0) {
        const updated = items.slice();
        updated[existingIndex] = nextMessage;
        return { ...current, [topicId]: updated };
      }
      return { ...current, [topicId]: [...items, nextMessage] };
    });
  }, []);

  const replaceTopicMessages = useCallback((topicId: number, items: ChatMessage[]) => {
    setTopicMessages((current) => ({ ...current, [topicId]: items }));
  }, []);

  const updateTopicStreamState = useCallback(
    (topicId: number, updater: (current: TopicStreamState) => TopicStreamState) => {
      setTopicStreamStates((current) => ({
        ...current,
        [topicId]: updater(current[topicId] ?? EMPTY_STREAM_STATE),
      }));
    },
    []
  );

  const stopProgressSubscription = useCallback((topicId: number) => {
    const unsubscribe = progressUnsubscribersRef.current.get(topicId);
    if (unsubscribe) {
      unsubscribe();
      progressUnsubscribersRef.current.delete(topicId);
    }
  }, []);

  const startProgressSubscription = useCallback(
    (topicId: number) => {
      stopProgressSubscription(topicId);
      const unsubscribe = subscribeChatProgressEvents({
        topicId,
        onEvent: (event) => {
          updateTopicStreamState(topicId, (current) => ({
            ...current,
            progressEvents: [...current.progressEvents, event].slice(-8),
          }));
        },
      });
      progressUnsubscribersRef.current.set(topicId, unsubscribe);
    },
    [stopProgressSubscription, updateTopicStreamState]
  );

  const loadTopics = useCallback(async () => {
    setLoadingTopics(true);
    try {
      const items = await fetchChatTopics();
      setTopics(items);
      setError(null);
      setSelectedTopicId((current) => {
        if (current && items.some((item) => item.id === current)) {
          return current;
        }
        return items[0]?.id ?? null;
      });
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : "获取话题列表失败";
      setError(message);
    } finally {
      setLoadingTopics(false);
    }
  }, []);

  const loadMessages = useCallback(async (topicId: number) => {
    setLoadingMessages(true);
    try {
      const items = await fetchTopicMessages(topicId);
      if (activeTopicIdRef.current !== topicId) {
        return;
      }
      replaceTopicMessages(topicId, items);
      setError(null);
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : "获取会话消息失败";
      setError(message);
    } finally {
      if (activeTopicIdRef.current === topicId) {
        setLoadingMessages(false);
      }
    }
  }, [replaceTopicMessages]);

  useEffect(() => {
    void loadTopics();
  }, [loadTopics]);

  useEffect(() => {
    activeTopicIdRef.current = selectedTopicId;
  }, [selectedTopicId]);

  useEffect(() => {
    if (!selectedTopicId) {
      setLoadingMessages(false);
      return;
    }
    void loadMessages(selectedTopicId);
  }, [loadMessages, selectedTopicId]);

  useEffect(() => {
    const viewport = messageViewportRef.current;
    if (!viewport) return;
    viewport.scrollTop = viewport.scrollHeight;
  }, [messages, currentStreamState.progressEvents, currentStreamState.streamingAssistantDraft, currentStreamState.submitting]);

  useEffect(() => {
    const streamAbortControllers = streamAbortControllersRef.current;
    const progressUnsubscribers = progressUnsubscribersRef.current;
    return () => {
      streamAbortControllers.forEach((controller) => controller.abort());
      progressUnsubscribers.forEach((unsubscribe) => unsubscribe());
      streamAbortControllers.clear();
      progressUnsubscribers.clear();
    };
  }, []);

  async function handleCreateTopic() {
    setCreatingTopic(true);
    setError(null);
    try {
      const topic = await createChatTopic();
      upsertTopic(topic);
      setSelectedTopicId(topic.id);
      replaceTopicMessages(topic.id, []);
    } catch (createError) {
      const message = createError instanceof Error ? createError.message : "创建话题失败";
      setError(message);
    } finally {
      setCreatingTopic(false);
    }
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const prompt = input.trim();
    if (!prompt) {
      setError("请输入问题");
      return;
    }
    if (!selectedTopicId) {
      setError("请先新建一个话题");
      return;
    }

    const requestTopicId = selectedTopicId;
    setError(null);
    setInput("");
    updateTopicStreamState(requestTopicId, () => ({
      submitting: true,
      progressEvents: [],
      streamingAssistantDraft: null,
    }));
    startProgressSubscription(requestTopicId);

    try {
      const controller = new AbortController();
      streamAbortControllersRef.current.set(requestTopicId, controller);
      await streamTopicMessage(requestTopicId, prompt, {
        signal: controller.signal,
        onUserMessage: (userMessage) => {
          appendMessageToTopic(requestTopicId, userMessage);
        },
        onAssistantStart: ({
          answer_mode,
          used_knowledge_base,
          sufficiency_decision,
          missing_information,
          response_kind,
          attribution_status,
          status_message,
          status_detail,
        }) => {
          updateTopicStreamState(requestTopicId, (current) => ({
            ...current,
            submitting: true,
            streamingAssistantDraft: {
              content: "",
              answerMode: answer_mode,
              usedKnowledgeBase: used_knowledge_base,
              sufficiencyDecision: sufficiency_decision ?? null,
              missingInformation: missing_information ?? null,
              responseKind: response_kind ?? null,
              attributionStatus: attribution_status ?? null,
              statusMessage: status_message ?? null,
              statusDetail: status_detail ?? null,
            },
          }));
        },
        onAssistantDelta: (delta) => {
          updateTopicStreamState(requestTopicId, (current) => ({
            ...current,
            submitting: true,
            streamingAssistantDraft: current.streamingAssistantDraft
              ? { ...current.streamingAssistantDraft, content: current.streamingAssistantDraft.content + delta }
              : {
                  content: delta,
                  answerMode: null,
                  usedKnowledgeBase: false,
                  sufficiencyDecision: null,
                  missingInformation: null,
                  responseKind: null,
                  attributionStatus: null,
                  statusMessage: null,
                  statusDetail: null,
                },
          }));
        },
        onAssistantComplete: (assistantMessage) => {
          appendMessageToTopic(requestTopicId, assistantMessage);
          updateTopicStreamState(requestTopicId, (current) => ({
            ...current,
            submitting: false,
            streamingAssistantDraft: null,
          }));
        },
      });
      await loadTopics();
    } catch (submitError) {
      const message =
        submitError instanceof Error && submitError.name === "AbortError"
          ? "已取消本次生成"
          : submitError instanceof Error
            ? submitError.message
            : "发送失败";
      setError(message);
      updateTopicStreamState(requestTopicId, (current) => ({
        ...current,
        submitting: false,
        streamingAssistantDraft: null,
      }));
    } finally {
      stopProgressSubscription(requestTopicId);
      streamAbortControllersRef.current.delete(requestTopicId);
      updateTopicStreamState(requestTopicId, (current) => ({
        ...current,
        submitting: false,
      }));
    }
  }

  const currentDraft = currentStreamState.streamingAssistantDraft;
  const currentIsSubmitting = currentStreamState.submitting;

  return (
    <>
      <div className="grid h-full min-h-0 gap-6 xl:grid-cols-[280px_minmax(0,1fr)]">
        <Card className="min-h-0 border-none bg-white/85 shadow-sm ring-1 ring-slate-200 backdrop-blur">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <MessagesSquare className="size-4" />
              话题
            </CardTitle>
            <CardDescription>新建话题后会自动保存上下文，刷新页面后仍可继续。</CardDescription>
          </CardHeader>
          <CardContent className="grid min-h-0 gap-4">
            <Button type="button" onClick={() => void handleCreateTopic()} disabled={creatingTopic} className="gap-2">
              {creatingTopic ? <LoaderCircle className="size-4 animate-spin" /> : <MessageSquarePlus className="size-4" />}
              {creatingTopic ? "创建中..." : "新建话题"}
            </Button>
            <div className="grid min-h-0 gap-3 overflow-y-auto">
              {loadingTopics ? (
                <div className="rounded-2xl border border-dashed border-slate-200 px-4 py-6 text-sm text-slate-500">
                  正在加载话题...
                </div>
              ) : null}
              {!loadingTopics && topics.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-8 text-sm text-slate-500">
                  还没有话题，先新建一个开始知识库对话。
                </div>
              ) : null}
              {topics.map((topic) => (
                <button
                  key={topic.id}
                  type="button"
                  onClick={() => setSelectedTopicId(topic.id)}
                  className={cn(
                    "grid gap-2 rounded-2xl border px-4 py-4 text-left transition",
                    selectedTopicId === topic.id
                      ? "border-slate-400 bg-slate-200 text-slate-900 shadow-sm"
                      : "border-slate-200 bg-slate-100 text-slate-800 hover:border-slate-300 hover:bg-slate-50"
                  )}
                >
                  <div className="flex items-center justify-between gap-3">
                    <p className="font-medium leading-6">{topic.title}</p>
                    <span className="text-xs text-slate-500">{topic.message_count} 条</span>
                  </div>
                  <p className="text-xs text-slate-500">更新于 {formatTime(topic.last_message_at ?? topic.updated_at)}</p>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="min-h-0 border-none bg-white/85 shadow-sm ring-1 ring-slate-200 backdrop-blur">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Database className="size-4" />
              知识库对话
            </CardTitle>
            <CardDescription>回答优先基于已归档论文；仅在完全没有可用材料时才显示通用知识补充。</CardDescription>
          </CardHeader>
          <CardContent className="flex min-h-0 flex-1 flex-col gap-4">
            <div ref={messageViewportRef} className="grid min-h-0 flex-1 gap-4 overflow-y-auto rounded-3xl bg-slate-50 p-4 ring-1 ring-slate-200">
              {!selectedTopic && !loadingTopics ? (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-white px-6 py-10 text-center text-sm text-slate-500">
                  先在左侧创建一个话题，再围绕知识库继续提问。
                </div>
              ) : null}
              {selectedTopic && loadingMessages ? (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-white px-4 py-6 text-sm text-slate-500">
                  正在加载会话内容...
                </div>
              ) : null}
              {selectedTopic && !loadingMessages && messages.length === 0 && !currentIsSubmitting ? (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-white px-6 py-10 text-center text-sm text-slate-500">
                  当前话题还没有消息，试着问一个和已归档论文相关的问题。
                </div>
              ) : null}
              {messages.map((message) => {
                const sufficiencyHint =
                  message.role === "assistant"
                    ? renderSufficiencyHint(
                        message.response_kind,
                        message.attribution_status,
                        message.status_message,
                        message.status_detail,
                        message.sufficiency_decision,
                        message.missing_information
                      )
                    : null;
                return (
                  <div
                    key={message.id}
                    className={cn(
                      "max-w-[92%] rounded-3xl px-4 py-4 shadow-sm",
                      message.role === "assistant"
                        ? "bg-white text-slate-800 ring-1 ring-slate-200"
                        : "ml-auto bg-slate-950 text-slate-50"
                    )}
                  >
                    <div className="mb-2 flex flex-wrap items-center gap-2 text-xs font-medium">
                      {message.role === "assistant" ? <Bot className="size-3.5" /> : <User2 className="size-3.5" />}
                      <span>{message.role === "assistant" ? "助手" : "你"}</span>
                      {message.model ? <span className="text-slate-400">· {message.model}</span> : null}
                      <span className={cn(message.role === "assistant" ? "text-slate-400" : "text-slate-300")}>
                        · {formatTime(message.created_at)}
                      </span>
                    </div>
                    {message.role === "assistant" ? (
                      <div className="mb-3 flex flex-wrap gap-2">
                        <span
                          className={cn(
                            "rounded-full px-2.5 py-1 text-xs ring-1",
                            getBadgeTone(message.response_kind, message.used_knowledge_base)
                          )}
                        >
                          {getAnswerBadge(message.response_kind, message.used_knowledge_base)}
                        </span>
                      </div>
                    ) : null}
                    {message.role === "assistant" && sufficiencyHint ? (
                      <div className="mb-3 rounded-2xl bg-slate-50 px-3 py-2 text-xs text-slate-600 ring-1 ring-slate-200">
                        <div className="font-medium text-slate-700">{sufficiencyHint.title}</div>
                        {sufficiencyHint.detail ? <div className="mt-1 text-slate-500">{sufficiencyHint.detail}</div> : null}
                      </div>
                    ) : null}
                    {message.role === "assistant" ? (
                      <ChatMarkdown content={message.content} />
                    ) : (
                      <p className="whitespace-pre-wrap text-sm leading-7">{message.content}</p>
                    )}
                    {message.role === "assistant" && message.citations.length > 0 ? (
                      <div className="mt-4 flex flex-wrap items-center gap-2">
                        <span className="text-xs font-medium text-slate-500">引用</span>
                        {message.citations.map((citation, index) => {
                          const pageRange = formatPageRange(citation);
                          return (
                            <Tooltip
                              key={`${citation.chunk_id}-${index}`}
                              content={
                                <div className="grid gap-1 text-left">
                                  <span className="font-medium text-white">
                                    {citation.paper_title || `论文 #${citation.paper_id}`}
                                  </span>
                                  {(citation.section_path || pageRange) ? (
                                    <span className="text-slate-200">
                                      {[citation.section_path ? `章节：${citation.section_path}` : null, pageRange ? `页码：${pageRange}` : null]
                                        .filter(Boolean)
                                        .join(" · ")}
                                    </span>
                                  ) : null}
                                  <span className="text-slate-100">{summarizeCitation(citation)}</span>
                                </div>
                              }
                            >
                              <button
                                type="button"
                                className="rounded-full border border-slate-200 bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:bg-slate-200"
                                onClick={() => setCitationDialogState({ citation, index: index + 1 })}
                              >
                                [{index + 1}]
                              </button>
                            </Tooltip>
                          );
                        })}
                      </div>
                    ) : null}
                  </div>
                );
              })}
              {currentIsSubmitting ? (
                <div className="max-w-[92%] rounded-3xl bg-white px-4 py-4 text-sm text-slate-500 shadow-sm ring-1 ring-slate-200">
                  <div className="flex items-center gap-2">
                    <LoaderCircle className="size-4 animate-spin" />
                    {currentDraft ? "正在输出最终答案..." : "正在处理你的问题..."}
                  </div>
                  {currentStreamState.progressEvents.length > 0 ? (
                    <div className="mt-3 grid gap-2">
                      {currentStreamState.progressEvents.map((event, index) => (
                        <div key={`${event.created_at}-${index}`} className="rounded-2xl bg-slate-50 px-3 py-2 text-xs text-slate-600 ring-1 ring-slate-200">
                          <div className="font-medium text-slate-700">{event.message}</div>
                          {event.detail ? <div className="mt-1 text-slate-500">{event.detail}</div> : null}
                        </div>
                      ))}
                    </div>
                  ) : null}
                  {currentDraft ? (
                    <div className="mt-3">
                      {(() => {
                        const sufficiencyHint = renderSufficiencyHint(
                          currentDraft.responseKind,
                          currentDraft.attributionStatus,
                          currentDraft.statusMessage,
                          currentDraft.statusDetail,
                          currentDraft.sufficiencyDecision,
                          currentDraft.missingInformation
                        );
                        return (
                          <>
                            <div className="mb-3 flex flex-wrap gap-2">
                              <span
                                className={cn(
                                  "rounded-full px-2.5 py-1 text-xs ring-1",
                                  getBadgeTone(currentDraft.responseKind, currentDraft.usedKnowledgeBase)
                                )}
                              >
                                {getAnswerBadge(currentDraft.responseKind, currentDraft.usedKnowledgeBase)}
                              </span>
                            </div>
                            {sufficiencyHint ? (
                              <div className="mb-3 rounded-2xl bg-slate-50 px-3 py-2 text-xs text-slate-600 ring-1 ring-slate-200">
                                <div className="font-medium text-slate-700">{sufficiencyHint.title}</div>
                                {sufficiencyHint.detail ? <div className="mt-1 text-slate-500">{sufficiencyHint.detail}</div> : null}
                              </div>
                            ) : null}
                            <ChatMarkdown
                              content={currentDraft.content || "正在准备输出..."}
                              className="text-slate-700"
                            />
                          </>
                        );
                      })()}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>

            <form className="grid gap-3" onSubmit={handleSubmit}>
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder={selectedTopic ? "例如：这些论文里，Transformer 的核心改进点是什么？" : "请先在左侧创建话题"}
                disabled={!selectedTopic || currentIsSubmitting}
                className="min-h-28 rounded-3xl border border-slate-200 bg-white px-4 py-4 text-sm leading-7 text-slate-800 outline-none transition focus:border-slate-400 focus:ring-4 focus:ring-slate-200 disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-slate-400"
              />
              {error ? <p className="text-sm text-destructive">{error}</p> : null}
              <div className="flex items-center justify-between gap-3">
                <p className="text-xs text-slate-500">话题会自动保存上下文；新建话题后会与当前会话隔离。</p>
                <Button type="submit" disabled={!selectedTopic || currentIsSubmitting} className="gap-2">
                  <Send className="size-4" />
                  {currentIsSubmitting ? "发送中..." : "发送"}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>

      <Dialog
        open={Boolean(citationDialogState)}
        title={citationDialogState ? `引用 [${citationDialogState.index}]` : undefined}
        description={citationDialogState ? citationDialogState.citation.paper_title || `论文 #${citationDialogState.citation.paper_id}` : undefined}
        onClose={() => setCitationDialogState(null)}
      >
        {citationDialogState ? (
          <div className="grid gap-5 px-6 py-6 text-sm text-slate-700">
            <div className="flex flex-wrap gap-3 text-xs text-slate-500">
              {citationDialogState.citation.section_path ? <span>章节：{citationDialogState.citation.section_path}</span> : null}
              {formatPageRange(citationDialogState.citation) ? (
                <span>页码：{formatPageRange(citationDialogState.citation)}</span>
              ) : null}
              {citationDialogState.citation.score !== null ? (
                <span>相关度 {citationDialogState.citation.score.toFixed(2)}</span>
              ) : null}
              {citationDialogState.citation.support_score !== null && citationDialogState.citation.support_score !== undefined ? (
                <span>支撑度 {citationDialogState.citation.support_score.toFixed(2)}</span>
              ) : null}
            </div>

            <div className="grid gap-2">
              <h3 className="text-sm font-semibold text-slate-900">证据摘录</h3>
              <div className="rounded-2xl bg-slate-50 px-4 py-3 leading-7 text-slate-700 ring-1 ring-slate-200">
                {citationDialogState.citation.snippet}
              </div>
            </div>

            {citationDialogState.citation.selection_reason ? (
              <div className="grid gap-2">
                <h3 className="text-sm font-semibold text-slate-900">选择理由</h3>
                <div className="rounded-2xl bg-slate-50 px-4 py-3 leading-7 text-slate-700 ring-1 ring-slate-200">
                  {citationDialogState.citation.selection_reason}
                </div>
              </div>
            ) : null}

            {citationDialogState.citation.claim_texts?.length ? (
              <div className="grid gap-2">
                <h3 className="text-sm font-semibold text-slate-900">支撑的回答片段</h3>
                <ul className="list-disc space-y-2 pl-5 text-slate-700">
                  {citationDialogState.citation.claim_texts.map((claimText, index) => (
                    <li key={`${citationDialogState.citation.chunk_id}-claim-${index}`}>{claimText}</li>
                  ))}
                </ul>
              </div>
            ) : null}

            {citationDialogState.citation.source_url ? (
              <div>
                <a
                  href={citationDialogState.citation.source_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 font-medium text-sky-700 underline underline-offset-4"
                >
                  查看来源
                  <ExternalLink className="size-3.5" />
                </a>
              </div>
            ) : null}
          </div>
        ) : null}
      </Dialog>
    </>
  );
}
