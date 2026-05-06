"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { FileText, MessagesSquare } from "lucide-react";

import { ChatPanel } from "@/components/chat-panel";
import { PaperWorkbench } from "@/components/paper-workbench";
import { TaskListPopover } from "@/components/task-list-popover";
import { Button } from "@/components/ui/button";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";

type Me = { id: number; username: string };

export default function Home() {
  const router = useRouter();
  const [me, setMe] = useState<Me | null>(null);
  const [phase, setPhase] = useState<"loading" | "ready" | "redirect">("loading");
  const [activeTab, setActiveTab] = useState<"papers" | "chat">("papers");
  const [selectedPaperId, setSelectedPaperId] = useState<number | null>(null);

  useEffect(() => {
    void apiFetch("/api/auth/me").then((r) => {
      if (!r.ok) {
        setPhase("redirect");
        router.replace("/login");
        return;
      }
      void r.json().then((data: Me) => {
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

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(148,163,184,0.18),transparent_34%),linear-gradient(180deg,#f8fafc_0%,#eef2ff_45%,#f8fafc_100%)]">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-8 px-6 py-8 lg:px-10">
        <header className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
          <div className="grid gap-3">
            <div className="inline-flex w-fit items-center rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-xs font-medium text-slate-600 shadow-sm backdrop-blur">
              Milestone 3 Workspace
            </div>
            <div>
              <h1 className="text-3xl font-semibold tracking-tight text-slate-950">ResearchDock</h1>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
                首页直接进入论文归档工作台，支持查看论文列表、OCR 预览、通用对话，以及右上角任务追踪。
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <TaskListPopover
              onOpenPaper={(paperId) => {
                setActiveTab("papers");
                setSelectedPaperId(paperId);
              }}
            />
            <div className="rounded-full border border-slate-200 bg-white/80 px-3 py-1.5 text-sm text-slate-600 shadow-sm backdrop-blur">
              已登录：{me.username}
            </div>
            <Button variant="outline" size="sm" onClick={() => void logout()}>
              退出
            </Button>
          </div>
        </header>

        <div className="flex flex-wrap items-center gap-3">
          <TabButton active={activeTab === "papers"} onClick={() => setActiveTab("papers")} icon={<FileText className="size-4" />}>
            论文
          </TabButton>
          <TabButton active={activeTab === "chat"} onClick={() => setActiveTab("chat")} icon={<MessagesSquare className="size-4" />}>
            对话
          </TabButton>
        </div>

        {activeTab === "papers" ? (
          <PaperWorkbench selectedPaperId={selectedPaperId} onSelectedPaperChange={setSelectedPaperId} />
        ) : (
          <ChatPanel />
        )}
      </div>
    </div>
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
