"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Clock3, LoaderCircle, Square, Trash2, TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  getJobStatusClassName,
  getJobStatusLabel,
  isActiveJobStatus,
  isDeletableJobStatus,
  isHiddenTaskListJobStatus,
  isStoppableJobStatus,
} from "@/lib/job-status";
import { useHasPermission } from "@/lib/session";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { cancelJob, deleteJob, fetchJobs, subscribeTaskStatusEvents, type JobPublic } from "@/lib/papers";

function formatTime(value: string | null): string {
  if (!value) return "-";
  return new Date(value).toLocaleString("zh-CN");
}

type TaskListPopoverProps = {
  onOpenPaper: (paperId: number) => void;
};

function mergeJobs(currentJobs: JobPublic[], nextJob: JobPublic): JobPublic[] {
  const remaining = currentJobs.filter((job) => job.id !== nextJob.id);
  return [nextJob, ...remaining].sort((left, right) => right.id - left.id).slice(0, 20);
}

export function TaskListPopover({ onOpenPaper }: TaskListPopoverProps) {
  const canManageJobs = useHasPermission("jobs:manage");
  const [open, setOpen] = useState(false);
  const [jobs, setJobs] = useState<JobPublic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingJobId, setDeletingJobId] = useState<number | null>(null);
  const [cancellingJobId, setCancellingJobId] = useState<number | null>(null);
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
    return subscribeTaskStatusEvents({
      onEvent: (event) => {
        if (!event.job) {
          return;
        }
        setJobs((current) => mergeJobs(current, event.job!));
      },
    });
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
    if (open) {
      void loadJobs();
    }
  }, [open]);

  const activeJobs = useMemo(() => jobs.filter((job) => isActiveJobStatus(job.status)), [jobs]);
  const failedJobs = useMemo(() => jobs.filter((job) => job.status === "failed"), [jobs]);
  const displayJobs = useMemo(() => jobs.filter((job) => !isHiddenTaskListJobStatus(job.status)), [jobs]);

  async function handleDeleteJob(job: JobPublic) {
    if (!isDeletableJobStatus(job.status)) {
      setError("仅已结束的任务支持删除");
      return;
    }

    setDeletingJobId(job.id);
    setError(null);
    try {
      await deleteJob(job.id);
      setJobs((current) => current.filter((item) => item.id !== job.id));
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : "删除任务失败";
      setError(message);
    } finally {
      setDeletingJobId(null);
    }
  }

  async function handleCancelJob(job: JobPublic) {
    if (!isStoppableJobStatus(job.status)) {
      return;
    }

    setCancellingJobId(job.id);
    setError(null);
    try {
      const updatedJob = await cancelJob(job.id);
      setJobs((current) => mergeJobs(current, updatedJob));
    } catch (cancelError) {
      const message = cancelError instanceof Error ? cancelError.message : "停止任务失败";
      setError(message);
    } finally {
      setCancellingJobId(null);
    }
  }

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
        <Card className="absolute right-0 top-11 z-20 w-[420px] border-none bg-white/95 shadow-xl ring-1 ring-slate-200 backdrop-blur">
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
            {!loading && displayJobs.length === 0 ? (
              <p className="text-sm text-slate-500">还没有任务记录。</p>
            ) : null}
            {displayJobs.map((job) => {
              const isActive = isActiveJobStatus(job.status);
              const isDeletable = isDeletableJobStatus(job.status);
              const isStoppable = isStoppableJobStatus(job.status);
              const isCancelling = cancellingJobId === job.id || job.status === "cancel_requested";
              const isDeleting = deletingJobId === job.id;

              return (
                <div key={job.id} className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 transition hover:border-slate-300 hover:bg-white">
                  <div className="flex items-start justify-between gap-3">
                    <button
                      type="button"
                      className="min-w-0 flex-1 text-left"
                      onClick={() => {
                        if (job.paper_id) {
                          onOpenPaper(job.paper_id);
                          setOpen(false);
                        }
                      }}
                    >
                      <div className="flex items-center gap-2 text-sm font-medium text-slate-900">
                        <StatusIcon status={job.status} />
                        <span>任务 #{job.id}</span>
                      </div>
                    </button>
                    <div className="flex items-center gap-2">
                      <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium ring-1", getJobStatusClassName(job.status))}>
                        {getJobStatusLabel(job.status)}
                      </span>
                      {canManageJobs ? (
                        <>
                          {(isStoppable || job.status === "cancel_requested") && (
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="gap-1.5"
                              aria-label={`停止任务 ${job.id}`}
                              onClick={() => void handleCancelJob(job)}
                              disabled={!isStoppable || isCancelling}
                              title={job.status === "cancel_requested" ? "任务正在取消中" : "停止任务"}
                            >
                              {isCancelling ? <LoaderCircle className="size-4 animate-spin" /> : <Square className="size-4" />}
                              停止
                            </Button>
                          )}
                          {isDeletable ? (
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon-sm"
                              aria-label={`删除任务 ${job.id}`}
                              onClick={() => void handleDeleteJob(job)}
                              disabled={isDeleting}
                              title="删除任务"
                            >
                              {isDeleting ? <LoaderCircle className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
                            </Button>
                          ) : null}
                        </>
                      ) : null}
                    </div>
                  </div>
                  <div className="mt-3 grid gap-1 text-xs text-slate-500">
                    <p>类型：{job.job_type || "-"}</p>
                    <p>论文：{job.paper_id ? `#${job.paper_id}` : "-"}</p>
                    <p>创建：{formatTime(job.created_at)}</p>
                    <p>开始：{formatTime(job.started_at)}</p>
                    <p>结束：{formatTime(job.finished_at)}</p>
                    {job.error_message ? <p className="text-rose-600">错误：{job.error_message}</p> : null}
                    {isActive ? <p>{job.status === "cancel_requested" ? "任务正在等待工作线程优雅取消。" : "任务运行中，可点击停止请求取消。"}</p> : null}
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}

function StatusIcon({ status }: { status: string | null }) {
  if (status === "completed") return <CheckCircle2 className="size-4 text-emerald-600" />;
  if (status === "failed") return <TriangleAlert className="size-4 text-rose-600" />;
  if (status === "cancel_requested") return <LoaderCircle className="size-4 animate-spin text-amber-600" />;
  if (status === "cancelled") return <Clock3 className="size-4 text-slate-500" />;
  if (status === "processing") return <LoaderCircle className="size-4 animate-spin text-amber-600" />;
  return <Clock3 className="size-4 text-sky-600" />;
}
