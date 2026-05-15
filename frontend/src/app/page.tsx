"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { FileQuestion, FileText, MessagesSquare } from "lucide-react";

import { ChatPanel } from "@/components/chat-panel";
import { OrgQuestionSetPanel } from "@/components/org-question-set-panel";
import { PaperWorkbench } from "@/components/paper-workbench";
import { TaskListPopover } from "@/components/task-list-popover";
import { Button } from "@/components/ui/button";
import { apiFetch } from "@/lib/api";
import { SessionProvider, type SessionUser } from "@/lib/session";
import { cn } from "@/lib/utils";

export default function Home() {
  const router = useRouter();
  const [me, setMe] = useState<SessionUser | null>(null);
  const [phase, setPhase] = useState<"loading" | "ready" | "redirect">("loading");
  const [activeTab, setActiveTab] = useState<"papers" | "chat" | "question-set">("papers");
  const [selectedPaperId, setSelectedPaperId] = useState<number | null>(null);

  useEffect(() => {
    void apiFetch("/api/auth/me").then((r) => {
      if (!r.ok) {
        setPhase("redirect");
        router.replace("/login");
        return;
      }
      void r.json().then((data: SessionUser) => {
        setMe(data);
        setPhase("ready");
      });
    });
  }, [router]);

  async function logout() {
    await apiFetch("/api/auth/logout", { method: "POST" });
    router.replace("/login");
    router.refresh();
  }

  if (phase !== "ready" || !me) {
    return (
      <div className="flex min-h-screen items-center justify-center p-6">
        <p className="text-muted-foreground text-sm">{phase === "redirect" ? "跳转登录…" : "加载中…"}</p>
      </div>
    );
  }

  const canReadPapers = me.permissions.includes("papers:read");
  const canReadJobs = me.permissions.includes("jobs:read");
  const canReadOrgSettings = me.permissions.includes("org_settings:read");
  const resolvedTab =
    activeTab === "papers" && !canReadPapers
      ? canReadOrgSettings
        ? "question-set"
        : "chat"
      : activeTab === "question-set" && !canReadOrgSettings
        ? canReadPapers
          ? "papers"
          : "chat"
        : activeTab;

  return (
    <SessionProvider value={me}>
      <div className="h-dvh overflow-hidden bg-[radial-gradient(circle_at_top,rgba(148,163,184,0.18),transparent_34%),linear-gradient(180deg,#f8fafc_0%,#eef2ff_45%,#f8fafc_100%)]">
        <div className="mx-auto flex h-full min-h-0 max-w-7xl flex-col gap-6 px-6 py-6 lg:px-10">
          <header className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
            <div className="grid gap-3">
              <div className="inline-flex w-fit items-center rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-xs font-medium text-slate-600 shadow-sm backdrop-blur">
                Milestone 4 Knowledge Base
              </div>
              <div>
                <h1 className="text-3xl font-semibold tracking-tight text-slate-950">ResearchDock</h1>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
                  首页直接进入论文归档工作台与知识库对话页，支持围绕论文归档进行带引用问答、话题上下文保存，以及右上角任务追踪。
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              {canReadJobs ? (
                <TaskListPopover
                  onOpenPaper={(paperId) => {
                    setActiveTab("papers");
                    setSelectedPaperId(paperId);
                  }}
                />
              ) : null}
              <div className="rounded-full border border-slate-200 bg-white/80 px-3 py-1.5 text-sm text-slate-600 shadow-sm backdrop-blur">
                已登录：{me.username} / {me.organization.name}
              </div>
              <Button variant="outline" size="sm" onClick={() => void logout()}>
                退出
              </Button>
            </div>
          </header>

          <div className="flex flex-wrap items-center gap-3">
            {canReadPapers ? (
              <TabButton active={resolvedTab === "papers"} onClick={() => setActiveTab("papers")} icon={<FileText className="size-4" />}>
                论文
              </TabButton>
            ) : null}
            {canReadOrgSettings ? (
              <TabButton active={resolvedTab === "question-set"} onClick={() => setActiveTab("question-set")} icon={<FileQuestion className="size-4" />}>
                问题集
              </TabButton>
            ) : null}
            <TabButton active={resolvedTab === "chat"} onClick={() => setActiveTab("chat")} icon={<MessagesSquare className="size-4" />}>
              对话
            </TabButton>
          </div>

          <div className="min-h-0 flex-1 overflow-hidden">
            {resolvedTab === "papers" ? (
              <PaperWorkbench selectedPaperId={selectedPaperId} onSelectedPaperChange={setSelectedPaperId} />
            ) : resolvedTab === "question-set" ? (
              <OrgQuestionSetPanel />
            ) : (
              <ChatPanel />
            )}
          </div>
        </div>
      </div>
    </SessionProvider>
  );
}

function TabButton({
  active,
  children,
  icon,
  onClick,
}: {
  active: boolean;
  children: ReactNode;
  icon: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition",
        active
          ? "border-slate-900 bg-slate-950 text-white shadow-sm"
          : "border-slate-200 bg-white/85 text-slate-600 shadow-sm backdrop-blur hover:border-slate-300 hover:text-slate-900"
      )}
    >
      {icon}
      {children}
    </button>
  );
}
