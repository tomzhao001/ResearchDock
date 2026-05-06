"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Clock3, LoaderCircle, TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { fetchJobs, type JobPublic } from "@/lib/papers";

const ACTIVE_STATUSES = new Set(["queued", "processing"]);

function statusLabel(status: string | null): string {
  if (status === "queued") return "排队中";
  if (status === "processing") return "处理中";
  if (status === "completed") return "已完成";
  if (status === "failed") return "失败";
  return "未知";
}

function formatTime(value: string | null): string {
  if (!value) return "-";
  return new Date(value).toLocaleString("zh-CN");
}

function getStatusClassName(status: string | null): string {
  if (status === "completed") return "bg-emerald-500/10 text-emerald-700 ring-emerald-500/20";
  if (status === "failed") return "bg-rose-500/10 text-rose-700 ring-rose-500/20";
  if (status === "processing") return "bg-amber-500/10 text-amber-700 ring-amber-500/20";
  if (status === "queued") return "bg-sky-500/10 text-sky-700 ring-sky-500/20";
  return "bg-muted text-muted-foreground ring-border";
}

type TaskListPopoverProps = {
  onOpenPaper: (paperId: number) => void;
};

export function TaskListPopover({ onOpenPaper }: TaskListPopoverProps) {
  const [open, setOpen] = useState(false);
  const [jobs, setJobs] = useState<JobPublic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  async function loadJobs() {
    try {
      const nextJobs = await fetchJobs();
      setJobs(nextJobs);
      setError(null);
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : "获取任务列表失败";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadJobs();
  }, []);

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

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadJobs();
    }, open ? 3000 : 5000);

    return () => window.clearTimeout(timer);
  }, [jobs, open]);

  const activeJobs = useMemo(() => jobs.filter((job) => ACTIVE_STATUSES.has(job.status ?? "")), [jobs]);
  const failedJobs = useMemo(() => jobs.filter((job) => job.status === "failed"), [jobs]);

  return (
    <div ref={rootRef} className="relative">
      <Button type="button" variant="outline" size="sm" onClick={() => setOpen((value) => !value)} className="gap-2">
        任务
        {activeJobs.length > 0 ? (
          <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[11px] font-medium text-amber-700">
            {activeJobs.length}
          </span>
        ) : failedJobs.length > 0 ? (
          <span className="rounded-full bg-rose-500/15 px-2 py-0.5 text-[11px] font-medium text-rose-700">
            {failedJobs.length}
          </span>
        ) : null}
      </Button>

      {open ? (
        <Card className="absolute right-0 top-11 z-20 w-[360px] border-none bg-white/95 shadow-xl ring-1 ring-slate-200 backdrop-blur">
          <CardHeader className="border-b border-slate-200/80">
            <CardTitle className="flex items-center justify-between text-sm">
              <span>最近任务</span>
              <span className="text-xs font-normal text-slate-500">
                进行中 {activeJobs.length} / 失败 {failedJobs.length}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="grid max-h-[420px] gap-3 overflow-y-auto py-4">
            {loading ? (
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <LoaderCircle className="size-4 animate-spin" />
                正在加载任务...
              </div>
            ) : null}
            {error ? <p className="text-sm text-destructive">{error}</p> : null}
            {!loading && jobs.length === 0 ? (
              <p className="text-sm text-slate-500">还没有任务记录。</p>
            ) : null}
            {jobs.map((job) => (
              <button
                key={job.id}
                type="button"
                className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 text-left transition hover:border-slate-300 hover:bg-white"
                onClick={() => {
                  if (job.paper_id) {
                    onOpenPaper(job.paper_id);
                    setOpen(false);
                  }
                }}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-slate-900">
                    <StatusIcon status={job.status} />
                    <span>任务 #{job.id}</span>
                  </div>
                  <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium ring-1", getStatusClassName(job.status))}>
                    {statusLabel(job.status)}
                  </span>
                </div>
                <div className="mt-3 grid gap-1 text-xs text-slate-500">
                  <p>类型：{job.job_type || "-"}</p>
                  <p>论文：{job.paper_id ? `#${job.paper_id}` : "-"}</p>
                  <p>创建：{formatTime(job.created_at)}</p>
                  {job.error_message ? <p className="text-rose-600">错误：{job.error_message}</p> : null}
                </div>
              </button>
            ))}
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}

function StatusIcon({ status }: { status: string | null }) {
  if (status === "completed") return <CheckCircle2 className="size-4 text-emerald-600" />;
  if (status === "failed") return <TriangleAlert className="size-4 text-rose-600" />;
  if (status === "processing") return <LoaderCircle className="size-4 animate-spin text-amber-600" />;
  return <Clock3 className="size-4 text-sky-600" />;
}
