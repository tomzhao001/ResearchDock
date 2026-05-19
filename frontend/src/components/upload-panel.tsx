"use client";

import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { getJobStatusLabel } from "@/lib/job-status";
import {
  UploadConflictError,
  fetchJob,
  subscribeTaskStatusEvents,
  uploadPaper,
  type JobPublic,
  type UploadAcceptedResponse,
} from "@/lib/papers";

function statusLabel(status: string | null): string {
  return getJobStatusLabel(status, { variant: "upload", emptyLabel: "未开始" });
}

export type UploadPanelProps = {
  onUploadAccepted?: (accepted: UploadAcceptedResponse) => void;
  onJobUpdate?: (job: JobPublic) => void;
  onCloseAfterUpload?: () => void;
};

type UploadConflictState = {
  existingPaperId: number;
  filename: string;
  file: File;
  nextIndex: number;
};

function isPdfFile(file: File): boolean {
  if (file.type === "application/pdf" || file.type === "application/x-pdf") {
    return true;
  }
  return file.name.toLowerCase().endsWith(".pdf");
}

export function UploadPanel({ onUploadAccepted, onJobUpdate, onCloseAfterUpload }: UploadPanelProps) {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [job, setJob] = useState<JobPublic | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [conflict, setConflict] = useState<UploadConflictState | null>(null);
  const [submittedCount, setSubmittedCount] = useState(0);
  const [currentFilename, setCurrentFilename] = useState<string | null>(null);

  const selectionLabel = useMemo(() => {
    if (selectedFiles.length === 0) {
      return "尚未选择文件";
    }
    if (selectedFiles.length === 1) {
      return `已选择：${selectedFiles[0].name}`;
    }
    return `已选择 ${selectedFiles.length} 个 PDF`;
  }, [selectedFiles]);

  useEffect(() => {
    if (!job?.id) {
      return;
    }

    return subscribeTaskStatusEvents({
      onEvent: (event) => {
        const nextJob = event.job;
        if (!nextJob || nextJob.id !== job.id) {
          return;
        }
        setJob(nextJob);
        onJobUpdate?.(nextJob);
      },
    });
  }, [job?.id, onJobUpdate]);

  async function processUploads(files: File[], startIndex = 0, overwriteFile?: File) {
    if (files.length === 0) {
      setError("请先选择 PDF 文件");
      return;
    }

    setUploading(true);

    try {
      for (let index = startIndex; index < files.length; index += 1) {
        const file = files[index];
        const overwrite = overwriteFile !== undefined && overwriteFile === file;

        setCurrentFilename(file.name);

        try {
          const accepted = await uploadPaper(file, overwrite);
          setSubmittedCount(index + 1);
          onUploadAccepted?.(accepted);
          const nextJob = await fetchJob(accepted.job_id);
          setJob(nextJob);
          onJobUpdate?.(nextJob);
        } catch (submitError) {
          if (submitError instanceof UploadConflictError) {
            setConflict({
              filename: submitError.detail.filename,
              existingPaperId: submitError.detail.existing_paper_id,
              file,
              nextIndex: index,
            });
            return;
          }

          const message = submitError instanceof Error ? submitError.message : "上传失败";
          setError(message);
          return;
        }
      }

      onCloseAfterUpload?.();
    } finally {
      setCurrentFilename(null);
      setUploading(false);
    }
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (selectedFiles.length === 0) {
      setError("请先选择 PDF 文件");
      return;
    }

    setError(null);
    setConflict(null);
    setSubmittedCount(0);
    setJob(null);
    await processUploads(selectedFiles);
  }

  return (
    <div className="grid gap-5 p-6">
      <form className="grid gap-4" onSubmit={handleSubmit}>
        <div className="grid gap-2">
          <Label htmlFor="paper-upload">选择 PDF</Label>
          <Input
            id="paper-upload"
            type="file"
            multiple
            accept="application/pdf,.pdf"
            onChange={(event) => {
              const files = Array.from(event.target.files ?? []);
              const invalidFiles = files.filter((file) => !isPdfFile(file));
              setSelectedFiles(invalidFiles.length === 0 ? files : []);
              setConflict(null);
              setSubmittedCount(0);
              setCurrentFilename(null);
              setJob(null);
              setError(invalidFiles.length > 0 ? "仅支持上传 PDF 文件，请重新选择。" : null);
            }}
          />
          <p className="text-muted-foreground text-xs">支持一次选择多个 PDF，系统会按选择顺序依次处理。</p>
        </div>
        {error ? <p className="text-sm text-destructive">{error}</p> : null}
        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" disabled={uploading}>
            {uploading ? "上传中..." : "批量上传 PDF"}
          </Button>
          <span className="text-muted-foreground text-sm">{selectionLabel}</span>
        </div>
        {selectedFiles.length > 1 ? (
          <p className="text-muted-foreground text-xs">当前批次共 {selectedFiles.length} 个文件，将按你选择的顺序依次提交。</p>
        ) : null}
        {selectedFiles.length > 0 ? (
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-700">
            <p>当前文件顺序：{selectedFiles.map((file) => file.name).join(" -> ")}</p>
          </div>
        ) : null}
      </form>

      {conflict ? (
        <div className="grid gap-3 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-4 text-sm text-amber-900">
          <p>已有相同原始文件名的文档：{conflict.filename}。</p>
          <p className="text-amber-800">如果继续覆盖，系统会软删除旧文档并创建新的解析任务。确认后将从当前文件继续，后续文件仍按原顺序处理。</p>
          <div className="flex flex-wrap gap-3">
            <Button
              type="button"
              onClick={() => {
                setError(null);
                setConflict(null);
                void processUploads(selectedFiles, conflict.nextIndex, conflict.file);
              }}
              disabled={uploading}
            >
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
          当前批次已提交 {submittedCount} / {selectedFiles.length} 个文件
          {currentFilename ? `，正在处理：${currentFilename}` : ""}。最近创建的任务 #{job.id} 状态为 {statusLabel(job.status ?? null)}。详细的开始时间、结束时间和错误原因可在右上角任务列表中查看。
        </div>
      ) : null}
    </div>
  );
}
