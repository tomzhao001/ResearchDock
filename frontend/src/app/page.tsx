"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";

import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";

type Me = { id: number; username: string };

export default function Home() {
  const router = useRouter();
  const [me, setMe] = useState<Me | null>(null);
  const [phase, setPhase] = useState<"loading" | "ready" | "redirect">("loading");

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

  const n8nUrl = process.env.NEXT_PUBLIC_N8N_URL ?? "http://localhost:5678";

  if (phase !== "ready" || !me) {
    return (
      <div className="flex min-h-screen items-center justify-center p-6">
        <p className="text-muted-foreground text-sm">{phase === "redirect" ? "跳转登录…" : "加载中…"}</p>
      </div>
    );
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-3xl flex-col gap-6 p-8">
      <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">ResearchDock</h1>
          <p className="text-muted-foreground text-sm">论文归档与问答系统 — Milestone 1 工程骨架</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground text-sm">已登录：{me.username}</span>
          <Button variant="outline" size="sm" onClick={() => void logout()}>
            退出
          </Button>
        </div>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>状态</CardTitle>
          <CardDescription>后端、数据库与 n8n 已就绪后，将在后续里程碑中接入业务页面。</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 text-sm">
          <Row label="当前用户" value={me.username} />
          <Row label="前端" value="Next.js + Tailwind + shadcn/ui" />
          <Row label="后端 API" value={process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"} />
        </CardContent>
        <CardFooter className="flex flex-col items-start gap-3 border-t pt-6 sm:flex-row sm:flex-wrap">
          <PlaceholderButton disabled>设置（即将推出）</PlaceholderButton>
          <PlaceholderButton disabled>抓取源（即将推出）</PlaceholderButton>
          <PlaceholderButton disabled>论文库（即将推出）</PlaceholderButton>
          <PlaceholderButton disabled>问答（即将推出）</PlaceholderButton>
          <PlaceholderButton disabled>任务（即将推出）</PlaceholderButton>
          <a
            href={n8nUrl}
            target="_blank"
            rel="noopener noreferrer"
            className={cn(buttonVariants({ variant: "secondary", size: "sm" }))}
          >
            打开 n8n
          </a>
        </CardFooter>
      </Card>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5 sm:flex-row sm:gap-3">
      <span className="text-muted-foreground w-28 shrink-0">{label}</span>
      <span className="font-medium break-all">{value}</span>
    </div>
  );
}

function PlaceholderButton({ children, disabled }: { children: ReactNode; disabled?: boolean }) {
  return (
    <Button variant="outline" size="sm" disabled={disabled}>
      {children}
    </Button>
  );
}
