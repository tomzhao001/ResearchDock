"use client";

import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { fetchJob, uploadPaper, type JobPublic } from "@/lib/papers";

const POLLABLE_STATUSES = new Set(["queued", "processing"]);

function statusLabel(status: string | null): string {
  if (status === "queued") return "已入队";
  if (status === "processing") return "处理中";
  if (status === "completed") return "已完成";
  if (status === "failed") return "失败";
  return "未开始";
}

function formatTime(value: string | null): string {
  if (!value) return "-";
  return new Date(value).toLocaleString("zh-CN");
}

export function UploadPanel({ pollDelayMs = 2000 }: { pollDelayMs?: number }) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [job, setJob] = useState<JobPublic | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [activeFilename, setActiveFilename] = useState<string>("-");

  useEffect(() => {
    if (!job || !POLLABLE_STATUSES.has(job.status ?? "")) {
      return;
    }

    const timer = window.setTimeout(() => {
      void fetchJob(job.id)
        .then((nextJob) => setJob(nextJob))
        .catch((nextError: Error) => setError(nextError.message));
    }, pollDelayMs);

    return () => window.clearTimeout(timer);
  }, [job, pollDelayMs]);

  const jobStatus = useMemo(() => statusLabel(job?.status ?? null), [job?.status]);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedFile) {
      setError("请先选择 PDF 文件");
      return;
    }

    setUploading(true);
    setError(null);
    setJob(null);
    setActiveFilename(selectedFile.name);

    try {
      const accepted = await uploadPaper(selectedFile);
      const nextJob = await fetchJob(accepted.job_id);
      setJob(nextJob);
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : "上传失败";
      setError(message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>PDF 上传与 OCR</CardTitle>
        <CardDescription>上传论文 PDF 后，系统会先提取文本层，再按需对低质量页面触发 OCR。</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6">
        <form className="grid gap-4" onSubmit={handleSubmit}>
          <div className="grid gap-2">
            <Label htmlFor="paper-upload">选择 PDF</Label>
            <Input
              id="paper-upload"
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
            />
            <p className="text-muted-foreground text-xs">当前仅支持单文件上传，任务会在后台异步完成文本提取与 OCR fallback。</p>
          </div>
          {error ? <p className="text-sm text-destructive">{error}</p> : null}
          <div className="flex gap-3">
            <Button type="submit" disabled={uploading}>
              {uploading ? "上传中…" : "上传 PDF"}
            </Button>
            <span className="text-muted-foreground self-center text-sm">
              {selectedFile ? `已选择：${selectedFile.name}` : "尚未选择文件"}
            </span>
          </div>
        </form>

        <div className="grid gap-3 text-sm">
          <Row label="文件名" value={activeFilename} />
          <Row label="任务状态" value={jobStatus} />
          <Row label="任务编号" value={job ? String(job.id) : "-"} />
          <Row label="开始时间" value={formatTime(job?.started_at ?? null)} />
          <Row label="结束时间" value={formatTime(job?.finished_at ?? null)} />
          <Row label="失败原因" value={job?.error_message ?? "-"} />
        </div>
      </CardContent>
    </Card>
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
