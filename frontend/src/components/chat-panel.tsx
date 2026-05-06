"use client";

import { useState } from "react";
import { Bot, LoaderCircle, Send, Sparkles, User2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { sendChatMessage } from "@/lib/chat";
import { cn } from "@/lib/utils";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  model?: string | null;
};

const INITIAL_MESSAGES: ChatMessage[] = [
  {
    id: "assistant-welcome",
    role: "assistant",
    content: "这里先提供一个通用对话入口，当前不绑定真实知识库；只要配置好 OpenAI 兼容接口，就可以先验证模型链路是否打通。",
  },
];

export function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>(INITIAL_MESSAGES);
  const [input, setInput] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const prompt = input.trim();
    if (!prompt) {
      setError("请输入问题");
      return;
    }

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: prompt,
    };

    setMessages((current) => [...current, userMessage]);
    setInput("");
    setError(null);
    setSubmitting(true);

    try {
      const response = await sendChatMessage(prompt);
      setMessages((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: response.answer,
          model: response.model,
        },
      ]);
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : "发送失败";
      setError(message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="grid h-full min-h-0 gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
      <Card className="min-h-0 border-none bg-white/85 shadow-sm ring-1 ring-slate-200 backdrop-blur">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Bot className="size-4" />
            通用对话
          </CardTitle>
          <CardDescription>当前阶段仅验证大模型链路，不接入知识库，也不会返回论文引用。</CardDescription>
        </CardHeader>
        <CardContent className="flex min-h-0 flex-1 flex-col gap-4">
          <div className="grid min-h-0 flex-1 gap-4 overflow-y-auto rounded-3xl bg-slate-50 p-4 ring-1 ring-slate-200">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "max-w-[88%] rounded-3xl px-4 py-4 shadow-sm",
                  message.role === "assistant"
                    ? "bg-white text-slate-800 ring-1 ring-slate-200"
                    : "ml-auto bg-slate-950 text-slate-50"
                )}
              >
                <div className="mb-2 flex items-center gap-2 text-xs font-medium">
                  {message.role === "assistant" ? <Bot className="size-3.5" /> : <User2 className="size-3.5" />}
                  <span>{message.role === "assistant" ? "助手" : "你"}</span>
                  {message.model ? <span className="text-slate-400">· {message.model}</span> : null}
                </div>
                <p className="whitespace-pre-wrap text-sm leading-7">{message.content}</p>
              </div>
            ))}
            {submitting ? (
              <div className="max-w-[88%] rounded-3xl bg-white px-4 py-4 text-sm text-slate-500 shadow-sm ring-1 ring-slate-200">
                <div className="flex items-center gap-2">
                  <LoaderCircle className="size-4 animate-spin" />
                  正在等待模型回复...
                </div>
              </div>
            ) : null}
          </div>

          <form className="grid gap-3" onSubmit={handleSubmit}>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="例如：请简单解释什么是 OCR fallback？"
              className="min-h-28 rounded-3xl border border-slate-200 bg-white px-4 py-4 text-sm leading-7 text-slate-800 outline-none transition focus:border-slate-400 focus:ring-4 focus:ring-slate-200"
            />
            {error ? <p className="text-sm text-destructive">{error}</p> : null}
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs text-slate-500">这里不保存会话历史，刷新页面后将重新开始。</p>
              <Button type="submit" disabled={submitting} className="gap-2">
                <Send className="size-4" />
                {submitting ? "发送中..." : "发送"}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      <Card className="min-h-0 border-none bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-slate-50 shadow-sm ring-1 ring-slate-800/80">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Sparkles className="size-4" />
            当前范围
          </CardTitle>
          <CardDescription className="text-slate-300">
            本阶段只做通用问答与接口打通，避免提前引入 RAG 和知识库复杂度。
          </CardDescription>
        </CardHeader>
        <CardContent className="grid min-h-0 gap-3 overflow-y-auto text-sm leading-7 text-slate-200">
          <p>已支持：</p>
          <p>- 通过环境变量配置 OpenAI 兼容 Base URL、API Key 和模型名。</p>
          <p>- 发送问题并获得真实模型回复。</p>
          <p>- 为后续论文级问答预留统一模型服务。</p>
          <p className="pt-2 text-slate-400">暂未支持引用、论文检索、多轮会话持久化。</p>
        </CardContent>
      </Card>
    </div>
  );
}
