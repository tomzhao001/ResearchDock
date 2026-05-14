"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bot, Database, LoaderCircle, MessageSquarePlus, MessagesSquare, Send, User2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  createChatTopic,
  fetchChatTopics,
  fetchTopicMessages,
  streamTopicMessage,
  subscribeChatProgressEvents,
  type ChatProgressEvent,
  type ChatMessage,
  type ChatTopic,
} from "@/lib/chat";
import { cn } from "@/lib/utils";

function formatTime(value: string | null): string {
  if (!value) return "-";
  return new Date(value).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function ChatPanel() {
  const [topics, setTopics] = useState<ChatTopic[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [selectedTopicId, setSelectedTopicId] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [loadingTopics, setLoadingTopics] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [creatingTopic, setCreatingTopic] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progressEvents, setProgressEvents] = useState<ChatProgressEvent[]>([]);
  const [streamingAssistantDraft, setStreamingAssistantDraft] = useState<{
    content: string;
    answerMode: string | null;
    usedKnowledgeBase: boolean;
  } | null>(null);
  const messageViewportRef = useRef<HTMLDivElement | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);

  const selectedTopic = useMemo(
    () => topics.find((topic) => topic.id === selectedTopicId) ?? null,
    [selectedTopicId, topics]
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
      setMessages(items);
      setError(null);
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : "获取会话消息失败";
      setError(message);
    } finally {
      setLoadingMessages(false);
    }
  }, []);

  useEffect(() => {
    void loadTopics();
  }, [loadTopics]);

  useEffect(() => {
    if (!selectedTopicId) {
      setMessages([]);
      return;
    }
    void loadMessages(selectedTopicId);
  }, [loadMessages, selectedTopicId]);

  useEffect(() => {
    const viewport = messageViewportRef.current;
    if (!viewport) return;
    viewport.scrollTop = viewport.scrollHeight;
  }, [messages, progressEvents, streamingAssistantDraft, submitting]);

  useEffect(() => {
    if (!selectedTopicId) {
      setProgressEvents([]);
      return;
    }
    const unsubscribe = subscribeChatProgressEvents({
      topicId: selectedTopicId,
      onEvent: (event) => {
        setProgressEvents((current) => [...current, event].slice(-8));
      },
    });
    return unsubscribe;
  }, [selectedTopicId]);

  useEffect(() => {
    setProgressEvents([]);
    setStreamingAssistantDraft(null);
  }, [selectedTopicId]);

  useEffect(() => {
    return () => {
      streamAbortRef.current?.abort();
    };
  }, []);

  async function handleCreateTopic() {
    setCreatingTopic(true);
    setError(null);
    try {
      const topic = await createChatTopic();
      upsertTopic(topic);
      setSelectedTopicId(topic.id);
      setMessages([]);
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

    setSubmitting(true);
    setError(null);
    setProgressEvents([]);
    setStreamingAssistantDraft(null);

    try {
      const controller = new AbortController();
      streamAbortRef.current = controller;
      await streamTopicMessage(selectedTopicId, prompt, {
        signal: controller.signal,
        onUserMessage: (userMessage) => {
          setInput("");
          setMessages((current) => [...current, userMessage]);
        },
        onAssistantStart: ({ answer_mode, used_knowledge_base }) => {
          setStreamingAssistantDraft({
            content: "",
            answerMode: answer_mode,
            usedKnowledgeBase: used_knowledge_base,
          });
        },
        onAssistantDelta: (delta) => {
          setStreamingAssistantDraft((current) =>
            current
              ? { ...current, content: current.content + delta }
              : {
                  content: delta,
                  answerMode: null,
                  usedKnowledgeBase: false,
                }
          );
        },
        onAssistantComplete: (assistantMessage) => {
          setStreamingAssistantDraft(null);
          setMessages((current) => [...current, assistantMessage]);
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
    } finally {
      streamAbortRef.current = null;
      setSubmitting(false);
    }
  }

  return (
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
          <CardDescription>回答优先基于已归档论文；若知识库没有确切依据，会明确标出通用补充。</CardDescription>
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
            {selectedTopic && !loadingMessages && messages.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-white px-6 py-10 text-center text-sm text-slate-500">
                当前话题还没有消息，试着问一个和已归档论文相关的问题。
              </div>
            ) : null}
            {messages.map((message) => (
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
                    {message.used_knowledge_base ? (
                      <span className="rounded-full bg-emerald-500/10 px-2.5 py-1 text-xs text-emerald-700 ring-1 ring-emerald-500/20">
                        知识库回答
                      </span>
                    ) : (
                      <span className="rounded-full bg-amber-500/10 px-2.5 py-1 text-xs text-amber-700 ring-1 ring-amber-500/20">
                        知识库未命中，以下为通用补充
                      </span>
                    )}
                  </div>
                ) : null}
                <p className="whitespace-pre-wrap text-sm leading-7">{message.content}</p>
                {message.role === "assistant" && message.citations.length > 0 ? (
                  <div className="mt-4 grid gap-2">
                    {message.citations.map((citation) => (
                      <div key={citation.chunk_id} className="rounded-2xl bg-slate-50 px-4 py-3 text-sm text-slate-600 ring-1 ring-slate-200">
                        <p className="font-medium text-slate-700">{citation.paper_title || `论文 #${citation.paper_id}`}</p>
                        <p className="mt-1 whitespace-pre-wrap leading-6">{citation.snippet}</p>
                        <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-500">
                          {citation.source_url ? <span>{citation.source_url}</span> : null}
                          {citation.score !== null ? <span>相关度 {citation.score.toFixed(2)}</span> : null}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>
            ))}
            {submitting ? (
              <div className="max-w-[92%] rounded-3xl bg-white px-4 py-4 text-sm text-slate-500 shadow-sm ring-1 ring-slate-200">
                <div className="flex items-center gap-2">
                  <LoaderCircle className="size-4 animate-spin" />
                  {streamingAssistantDraft ? "正在输出最终答案..." : "正在处理你的问题..."}
                </div>
                {progressEvents.length > 0 ? (
                  <div className="mt-3 grid gap-2">
                    {progressEvents.map((event, index) => (
                      <div key={`${event.created_at}-${index}`} className="rounded-2xl bg-slate-50 px-3 py-2 text-xs text-slate-600 ring-1 ring-slate-200">
                        <div className="font-medium text-slate-700">{event.message}</div>
                        {event.detail ? <div className="mt-1 text-slate-500">{event.detail}</div> : null}
                      </div>
                    ))}
                  </div>
                ) : null}
                {streamingAssistantDraft ? (
                  <div className="mt-3">
                    <div className="mb-3 flex flex-wrap gap-2">
                      {streamingAssistantDraft.usedKnowledgeBase ? (
                        <span className="rounded-full bg-emerald-500/10 px-2.5 py-1 text-xs text-emerald-700 ring-1 ring-emerald-500/20">
                          知识库回答
                        </span>
                      ) : (
                        <span className="rounded-full bg-amber-500/10 px-2.5 py-1 text-xs text-amber-700 ring-1 ring-amber-500/20">
                          保守回答
                        </span>
                      )}
                    </div>
                    <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">
                      {streamingAssistantDraft.content || "正在准备输出..."}
                    </p>
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
              disabled={!selectedTopic || submitting}
              className="min-h-28 rounded-3xl border border-slate-200 bg-white px-4 py-4 text-sm leading-7 text-slate-800 outline-none transition focus:border-slate-400 focus:ring-4 focus:ring-slate-200 disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-slate-400"
            />
            {error ? <p className="text-sm text-destructive">{error}</p> : null}
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs text-slate-500">话题会自动保存上下文；新建话题后会与当前会话隔离。</p>
              <Button type="submit" disabled={!selectedTopic || submitting} className="gap-2">
                <Send className="size-4" />
                {submitting ? "发送中..." : "发送"}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
