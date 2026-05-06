"use client";

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  UploadConflictError,
  fetchJob,
  uploadPaper,
  type JobPublic,
  type UploadAcceptedResponse,
} from "@/lib/papers";

const POLLABLE_STATUSES = new Set(["queued", "processing"]);

function statusLabel(status: string | null): string {
  if (status === "queued") return "已入队";
  if (status === "processing") return "处理中";
  if (status === "completed") return "已完成";
  if (status === "failed") return "失败";
  return "未开始";
}

export type UploadPanelProps = {
  pollDelayMs?: number;
  onUploadAccepted?: (accepted: UploadAcceptedResponse) => void;
  onJobUpdate?: (job: JobPublic) => void;
  onCloseAfterUpload?: () => void;
};

export function UploadPanel({ pollDelayMs = 2000, onUploadAccepted, onJobUpdate, onCloseAfterUpload }: UploadPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [job, setJob] = useState<JobPublic | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [conflict, setConflict] = useState<{ filename: string; existingPaperId: number } | null>(null);

  useEffect(() => {
    if (!job || !POLLABLE_STATUSES.has(job.status ?? "")) {
      return;
    }

    const timer = window.setTimeout(() => {
      void fetchJob(job.id)
        .then((nextJob) => {
          setJob(nextJob);
          onJobUpdate?.(nextJob);
        })
        .catch((nextError: Error) => setError(nextError.message));
    }, pollDelayMs);

    return () => window.clearTimeout(timer);
  }, [job, onJobUpdate, pollDelayMs]);

  async function doUpload(overwrite: boolean) {
    if (!selectedFile) {
      setError("请先选择 PDF 文件");
      return;
    }

    setUploading(true);
    setError(null);
    setConflict(null);

    try {
      const accepted = await uploadPaper(selectedFile, overwrite);
      onUploadAccepted?.(accepted);
      const nextJob = await fetchJob(accepted.job_id);
      setJob(nextJob);
      onJobUpdate?.(nextJob);
      onCloseAfterUpload?.();
    } catch (submitError) {
      if (submitError instanceof UploadConflictError) {
        setConflict({
          filename: submitError.detail.filename,
          existingPaperId: submitError.detail.existing_paper_id,
        });
        return;
      }

      const message = submitError instanceof Error ? submitError.message : "上传失败";
      setError(message);
    } finally {
      setUploading(false);
    }
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await doUpload(false);
  }

  return (
    <div className="grid gap-5 p-6">
      <form className="grid gap-4" onSubmit={handleSubmit}>
        <div className="grid gap-2">
          <Label htmlFor="paper-upload">选择 PDF</Label>
          <Input
            id="paper-upload"
            type="file"
            accept="application/pdf,.pdf"
            onChange={(event) => {
              setSelectedFile(event.target.files?.[0] ?? null);
              setConflict(null);
              setError(null);
            }}
          />
          <p className="text-muted-foreground text-xs">当前仍只支持 PDF。上传后会自动继续做文本提取与 OCR fallback。</p>
        </div>
        {error ? <p className="text-sm text-destructive">{error}</p> : null}
        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" disabled={uploading}>
            {uploading ? "上传中..." : "上传 PDF"}
          </Button>
          <span className="text-muted-foreground text-sm">{selectedFile ? `已选择：${selectedFile.name}` : "尚未选择文件"}</span>
        </div>
      </form>

      {conflict ? (
        <div className="grid gap-3 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-4 text-sm text-amber-900">
          <p>已有相同原始文件名的文档：{conflict.filename}。</p>
          <p className="text-amber-800">如果继续覆盖，系统会软删除旧文档并创建新的 OCR 任务。</p>
          <div className="flex flex-wrap gap-3">
            <Button type="button" onClick={() => void doUpload(true)} disabled={uploading}>
              {uploading ? "覆盖中..." : "确认覆盖上传"}
            </Button>
            <Button type="button" variant="outline" onClick={() => setConflict(null)} disabled={uploading}>
              取消
            </Button>
          </div>
        </div>
      ) : null}

      {job ? (
        <div className="rounded-2xl border border-sky-200 bg-sky-50 px-4 py-4 text-sm text-sky-900">
          当前上传已创建任务 #{job.id}，状态为 {statusLabel(job.status ?? null)}。详细的开始时间、结束时间和错误原因可在右上角任务列表中查看。
        </div>
      ) : null}
    </div>
  );
}
