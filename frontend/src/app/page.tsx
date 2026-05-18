"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { ChevronDown, FileQuestion, FileText, MessagesSquare } from "lucide-react";

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
            <div>
              <h1 className="text-3xl font-semibold tracking-tight text-slate-950">ResearchDock</h1>
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
              <AccountMenu username={me.username} onLogout={() => void logout()} />
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

function AccountMenu({
  username,
  onLogout,
}: {
  username: string;
  onLogout: () => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;

    function handlePointerDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, [open]);

  return (
    <div ref={rootRef} className="relative">
      <Button type="button" variant="outline" size="sm" onClick={() => setOpen((value) => !value)} className="gap-2">
        {username}
        <ChevronDown className={cn("size-4 transition-transform", open ? "rotate-180" : "")} />
      </Button>
      {open ? (
        <div className="absolute right-0 top-11 z-20 min-w-[160px] rounded-2xl border border-slate-200 bg-white/95 p-2 shadow-xl backdrop-blur">
          <button
            type="button"
            onClick={() => {
              setOpen(false);
              onLogout();
            }}
            className="flex w-full items-center rounded-xl px-3 py-2 text-left text-sm text-slate-700 transition hover:bg-slate-100 hover:text-slate-950"
          >
            退出登录
          </button>
        </div>
      ) : null}
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
