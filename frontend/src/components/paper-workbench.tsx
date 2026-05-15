"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowUpDown, CalendarDays, FilePenLine, LoaderCircle, RefreshCw, ScanText, TextCursorInput, Trash2, Upload } from "lucide-react";

import { PaperMetadataDialog } from "@/components/paper-metadata-dialog";
import { PaperUploadDialog } from "@/components/paper-upload-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog } from "@/components/ui/dialog";
import { TabPanel, Tabs } from "@/components/ui/tabs";
import { useHasPermission } from "@/lib/session";
import { Tooltip } from "./ui/tooltip";
import { cn } from "@/lib/utils";
import {
  deletePaper,
  fetchPaper,
  fetchPapers,
  regeneratePaperSummary,
  rerunPaperOcr,
  subscribeTaskStatusEvents,
  type JobPublic,
  type PaperDetail,
  type PaperListItem,
  type UploadAcceptedResponse,
} from "@/lib/papers";

const ACTIVE_STATUSES = new Set(["queued", "processing"]);
type PaperSortMode = "alphabet" | "publishedAt";
type PaperSortDirection = "asc" | "desc";

function statusLabel(status: string | null): string {
  if (status === "queued") return "排队中";
  if (status === "processing") return "处理中";
  if (status === "completed") return "已完成";
  if (status === "failed") return "失败";
  if (status === "uploaded") return "已上传";
  return "未知";
}

function phaseStatusLabel(status: string | null): string {
  if (!status) return "未开始";
  return statusLabel(status);
}

function formatTime(value: string | null): string {
  if (!value) return "-";
  return new Date(value).toLocaleString("zh-CN");
}

type MetadataSourceLabel = "摘要自动提取" | "手动编辑" | null;

function normalizeMetadataValue(value: string | null | undefined): string {
  return (value || "").trim();
}

function normalizeMetadataDate(value: string | null | undefined): string {
  const normalized = normalizeMetadataValue(value);
  if (!normalized) return "";
  const timestamp = Date.parse(normalized);
  return Number.isNaN(timestamp) ? normalized : new Date(timestamp).toISOString();
}

function getMetadataSourceLabel({
  currentValue,
  extractedValue,
  isDate = false,
}: {
  currentValue: string | null | undefined;
  extractedValue: string | null | undefined;
  isDate?: boolean;
}): MetadataSourceLabel {
  const normalize = isDate ? normalizeMetadataDate : normalizeMetadataValue;
  const current = normalize(currentValue);
  if (!current) return null;
  const extracted = normalize(extractedValue);
  if (!extracted) return "手动编辑";
  return current === extracted ? "摘要自动提取" : "手动编辑";
}

function getStatusClassName(status: string | null): string {
  if (status === "completed") return "bg-emerald-500/10 text-emerald-700 ring-emerald-500/20";
  if (status === "failed") return "bg-rose-500/10 text-rose-700 ring-rose-500/20";
  if (status === "processing") return "bg-amber-500/10 text-amber-700 ring-amber-500/20";
  if (status === "queued") return "bg-sky-500/10 text-sky-700 ring-sky-500/20";
  return "bg-muted text-muted-foreground ring-border";
}

function getPhaseStatusClassName(status: string | null): string {
  if (!status) return "bg-slate-200/80 text-slate-600 ring-slate-300";
  return getStatusClassName(status);
}

function excerpt(value: string | null, maxLength = 96): string {
  const text = (value || "").trim();
  if (!text) return "尚未生成摘要";
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function PhaseBadge({
  label,
  status,
  className,
  action,
}: {
  label: string;
  status: string | null;
  className?: string;
  action?: {
    tooltip: string;
    ariaLabel: string;
    disabled?: boolean;
    loading?: boolean;
    onClick: () => void;
  };
}) {
  return (
    <span
      className={cn(
        "inline-flex min-w-[88px] items-center justify-center gap-1 whitespace-nowrap rounded-full px-3 py-1 text-center text-xs font-medium ring-1",
        getPhaseStatusClassName(status),
        className
      )}
    >
      <span>{label}</span>
      <span>{phaseStatusLabel(status)}</span>
      {action ? (
        <Tooltip content={action.tooltip}>
          <button
            type="button"
            aria-label={action.ariaLabel}
            className="inline-flex size-5 items-center justify-center rounded-full text-current transition hover:bg-black/5 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={(event) => {
              event.stopPropagation();
              action.onClick();
            }}
            disabled={action.disabled}
          >
            <RefreshCw className={cn("size-3", action.loading ? "animate-spin" : "")} />
          </button>
        </Tooltip>
      ) : null}
    </span>
  );
}

function getPaperAlphabetKey(paper: PaperListItem): string {
  return (paper.title || paper.original_filename || "").trim().toLocaleLowerCase("zh-CN");
}

function comparePaperAlphabet(left: PaperListItem, right: PaperListItem): number {
  const leftKey = getPaperAlphabetKey(left);
  const rightKey = getPaperAlphabetKey(right);
  const nameComparison = leftKey.localeCompare(rightKey, "zh-CN");
  if (nameComparison !== 0) {
    return nameComparison;
  }
  return left.id - right.id;
}

function comparePaperPublishedAt(left: PaperListItem, right: PaperListItem): number {
  const timeDifference = new Date(left.published_at!).getTime() - new Date(right.published_at!).getTime();
  if (timeDifference !== 0) {
    return timeDifference;
  }
  return comparePaperAlphabet(left, right);
}

function sortPapersForDisplay(
  items: PaperListItem[],
  sortMode: PaperSortMode,
  sortDirection: PaperSortDirection
): PaperListItem[] {
  return [...items].sort((left, right) => {
    if (sortMode === "alphabet") {
      const comparison = comparePaperAlphabet(left, right);
      return sortDirection === "asc" ? comparison : -comparison;
    }

    if (!left.published_at && !right.published_at) {
      const comparison = comparePaperAlphabet(left, right);
      return sortDirection === "asc" ? comparison : -comparison;
    }
    if (!left.published_at) {
      return 1;
    }
    if (!right.published_at) {
      return -1;
    }

    const comparison = comparePaperPublishedAt(left, right);
    return sortDirection === "asc" ? comparison : -comparison;
  });
}

function upsertPaper(currentPapers: PaperListItem[], nextPaper: PaperListItem): PaperListItem[] {
  const existingPaper = currentPapers.find((paper) => paper.id === nextPaper.id);
  if (!existingPaper) {
    return [nextPaper, ...currentPapers];
  }
  return currentPapers.map((paper) => (paper.id === nextPaper.id ? nextPaper : paper));
}

type PaperWorkbenchProps = {
  selectedPaperId: number | null;
  onSelectedPaperChange: (paperId: number | null) => void;
};

export function PaperWorkbench({ selectedPaperId, onSelectedPaperChange }: PaperWorkbenchProps) {
  const canWritePapers = useHasPermission("papers:write");
  const canDeletePapers = useHasPermission("papers:delete");
  const [papers, setPapers] = useState<PaperListItem[]>([]);
  const [paperDetail, setPaperDetail] = useState<PaperDetail | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [manualRefreshing, setManualRefreshing] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [metadataDialogOpen, setMetadataDialogOpen] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [confirmAction, setConfirmAction] = useState<"rerun-ocr" | "regenerate-summary" | null>(null);
  const [previewTab, setPreviewTab] = useState<"summary" | "ocr">("summary");
  const [actionLoading, setActionLoading] = useState<"rerun-ocr" | "regenerate-summary" | "delete" | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [sortMode, setSortMode] = useState<PaperSortMode>("publishedAt");
  const [sortDirection, setSortDirection] = useState<PaperSortDirection>("desc");

  const loadPapers = useCallback(
    async (preferredPaperId?: number | null, options?: { silent?: boolean }) => {
      if (!options?.silent) {
        setLoadingList(true);
      }
      try {
        const items = await fetchPapers();
        setPapers(items);
        setListError(null);

        if (items.length === 0) {
          onSelectedPaperChange(null);
          return;
        }

        const nextId =
          preferredPaperId && items.some((item) => item.id === preferredPaperId)
            ? preferredPaperId
            : selectedPaperId && items.some((item) => item.id === selectedPaperId)
              ? selectedPaperId
              : items[0].id;
        if (nextId !== selectedPaperId) {
          onSelectedPaperChange(nextId);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "获取论文列表失败";
        setListError(message);
      } finally {
        if (!options?.silent) {
          setLoadingList(false);
        }
      }
    },
    [onSelectedPaperChange, selectedPaperId]
  );

  const loadDetail = useCallback(async (paperId?: number | null, options?: { silent?: boolean }) => {
    const targetPaperId = paperId ?? selectedPaperId;
    if (!targetPaperId) {
      setPaperDetail(null);
      return;
    }

    if (!options?.silent) {
      setLoadingDetail(true);
    }
    try {
      const detail = await fetchPaper(targetPaperId);
      setPaperDetail(detail);
      setDetailError(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "获取论文详情失败";
      setDetailError(message);
    } finally {
      if (!options?.silent) {
        setLoadingDetail(false);
      }
    }
  }, [selectedPaperId]);

  useEffect(() => {
    void loadPapers();
  }, [loadPapers]);

  useEffect(() => {
    void loadDetail();
  }, [loadDetail]);

  useEffect(() => {
    return subscribeTaskStatusEvents({
      onEvent: (event) => {
        setPapers((current) => upsertPaper(current, event.paper_list_item));
        setPaperDetail((current) => {
          if (current?.id === event.paper_id || selectedPaperId === event.paper_id) {
            return event.paper_detail;
          }
          return current;
        });
      },
    });
  }, [selectedPaperId]);

  async function handleUploadAccepted(accepted: UploadAcceptedResponse) {
    onSelectedPaperChange(accepted.paper_id);
    await loadPapers(accepted.paper_id);
    await loadDetail(accepted.paper_id);
  }

  async function handleJobUpdate(job: JobPublic) {
    if (job.paper_id && job.paper_id !== selectedPaperId) {
      await loadPapers(job.paper_id);
      return;
    }
    await loadPapers(job.paper_id ?? selectedPaperId);
    await loadDetail(job.paper_id ?? selectedPaperId);
  }

  const hasActiveJob = useMemo(() => {
    if (!paperDetail) {
      return false;
    }
    return ACTIVE_STATUSES.has(paperDetail.ocr_status ?? "") || ACTIVE_STATUSES.has(paperDetail.summary_status ?? "");
  }, [paperDetail]);

  const sortedPapers = useMemo(
    () => sortPapersForDisplay(papers, sortMode, sortDirection),
    [papers, sortDirection, sortMode]
  );

  const isBusy = loadingList || loadingDetail || actionLoading !== null;

  async function handleDeletePaper() {
    if (!paperDetail) {
      return;
    }

    setActionLoading("delete");
    setActionError(null);
    try {
      const deletedPaperId = paperDetail.id;
      const remaining = papers.filter((paper) => paper.id !== deletedPaperId);
      const nextPaperId = sortPapersForDisplay(remaining, sortMode, sortDirection)[0]?.id ?? null;

      await deletePaper(deletedPaperId);
      setDeleteConfirmOpen(false);
      setMetadataDialogOpen(false);
      setPaperDetail(null);
      onSelectedPaperChange(nextPaperId);
      await loadPapers(nextPaperId);
      if (nextPaperId) {
        await loadDetail(nextPaperId);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "删除文档失败";
      setActionError(message);
    } finally {
      setActionLoading(null);
    }
  }

  async function handleRerunOcr() {
    if (!paperDetail) {
      return;
    }

    setConfirmAction(null);
    setActionLoading("rerun-ocr");
    setActionError(null);
    try {
      await rerunPaperOcr(paperDetail.id);
      await loadPapers(paperDetail.id);
      await loadDetail(paperDetail.id);
    } catch (error) {
      const message = error instanceof Error ? error.message : "重新 OCR 失败";
      setActionError(message);
    } finally {
      setActionLoading(null);
    }
  }

  async function handleRegenerateSummary() {
    if (!paperDetail) {
      return;
    }

    setConfirmAction(null);
    setActionLoading("regenerate-summary");
    setActionError(null);
    try {
      await regeneratePaperSummary(paperDetail.id);
      await loadPapers(paperDetail.id);
      await loadDetail(paperDetail.id);
    } catch (error) {
      const message = error instanceof Error ? error.message : "重新生成摘要失败";
      setActionError(message);
    } finally {
      setActionLoading(null);
    }
  }

  async function handleManualRefresh() {
    setManualRefreshing(true);
    try {
      await loadPapers(selectedPaperId, { silent: true });
      await loadDetail(selectedPaperId, { silent: true });
    } finally {
      setManualRefreshing(false);
    }
  }

  const confirmActionText =
    confirmAction === "rerun-ocr"
      ? {
          title: "确认重新执行 OCR",
          description: "系统会重新提取当前 PDF 的文本，并在完成后继续同步最新 OCR 结果。",
          buttonLabel: "确认重新 OCR",
        }
      : confirmAction === "regenerate-summary"
        ? {
            title: "确认重新生成摘要",
            description: "系统会基于当前 OCR 文本重新生成摘要和结构化信息。",
            buttonLabel: "确认重新生成摘要",
          }
        : null;

  return (
    <>
      <PaperUploadDialog
        open={uploadDialogOpen}
        onClose={() => setUploadDialogOpen(false)}
        onUploadAccepted={handleUploadAccepted}
        onJobUpdate={handleJobUpdate}
      />
      <PaperMetadataDialog
        open={metadataDialogOpen}
        paper={paperDetail}
        onClose={() => setMetadataDialogOpen(false)}
        onSaved={async (detail) => {
          setPaperDetail(detail);
          setActionError(null);
          await loadPapers(detail.id);
          await loadDetail(detail.id);
        }}
      />
      <Dialog
        open={deleteConfirmOpen}
        onClose={() => {
          if (actionLoading === "delete") {
            return;
          }
          setDeleteConfirmOpen(false);
        }}
        title="确认删除文档"
        description="删除后文档会从列表中移除，关联的任务记录也会一起删除。该操作不可撤销。"
        className="max-w-lg"
      >
        <div className="grid gap-4 p-6">
          <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-4 text-sm text-rose-800">
            {paperDetail ? `确定删除“${paperDetail.title || `未命名论文 #${paperDetail.id}`}”吗？` : "确定删除当前文档吗？"}
          </div>
          <div className="flex justify-end gap-3">
            <Button
              type="button"
              variant="outline"
              onClick={() => setDeleteConfirmOpen(false)}
              disabled={actionLoading === "delete"}
            >
              取消
            </Button>
            <Button type="button" variant="destructive" onClick={() => void handleDeletePaper()} disabled={actionLoading === "delete"}>
              {actionLoading === "delete" ? (
                <>
                  <LoaderCircle className="size-4 animate-spin" />
                  删除中...
                </>
              ) : (
                <>
                  <Trash2 className="size-4" />
                  确认删除
                </>
              )}
            </Button>
          </div>
        </div>
      </Dialog>
      <Dialog
        open={confirmAction !== null}
        onClose={() => {
          if (actionLoading === confirmAction && confirmAction !== null) {
            return;
          }
          setConfirmAction(null);
        }}
        title={confirmActionText?.title}
        description={confirmActionText?.description}
        className="max-w-lg"
      >
        <div className="grid gap-4 p-6">
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-700">
            {paperDetail
              ? `即将对“${paperDetail.title || `未命名论文 #${paperDetail.id}`}”发起新的任务。`
              : "即将发起新的任务。"}
          </div>
          <div className="flex justify-end gap-3">
            <Button
              type="button"
              variant="outline"
              onClick={() => setConfirmAction(null)}
              disabled={confirmAction !== null && actionLoading === confirmAction}
            >
              取消
            </Button>
            <Button
              type="button"
              onClick={() => {
                if (confirmAction === "rerun-ocr") {
                  void handleRerunOcr();
                  return;
                }
                if (confirmAction === "regenerate-summary") {
                  void handleRegenerateSummary();
                }
              }}
              disabled={confirmAction !== null && actionLoading === confirmAction}
            >
              {confirmAction !== null && actionLoading === confirmAction ? (
                <>
                  <LoaderCircle className="size-4 animate-spin" />
                  提交中...
                </>
              ) : (
                <>
                  <RefreshCw className="size-4" />
                  {confirmActionText?.buttonLabel}
                </>
              )}
            </Button>
          </div>
        </div>
      </Dialog>

      <div className="grid h-full min-h-0 gap-6 xl:grid-cols-[minmax(320px,420px)_minmax(0,1fr)]">
        <Card className="min-h-0 border-none bg-white/80 shadow-sm ring-1 ring-slate-200 backdrop-blur">
          <CardHeader className="gap-3">
            <div className="grid gap-3">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <CardTitle>论文列表</CardTitle>
                </div>
                <div className="flex items-center gap-1.5">
                  <Tooltip
                    content={sortMode === "alphabet" ? "当前按字母排序，点击切换到论文时间" : "当前按论文时间排序，点击切换到字母"}
                    side="bottom"
                  >
                    <Button
                      type="button"
                      variant="outline"
                      size="icon-sm"
                      aria-label={sortMode === "alphabet" ? "当前按字母排序，点击切换到论文时间" : "当前按论文时间排序，点击切换到字母"}
                      onClick={() => setSortMode((current) => (current === "alphabet" ? "publishedAt" : "alphabet"))}
                      disabled={isBusy}
                    >
                      {sortMode === "alphabet" ? <TextCursorInput className="size-3.5" /> : <CalendarDays className="size-3.5" />}
                    </Button>
                  </Tooltip>
                  <Tooltip
                    content={sortDirection === "asc" ? "当前正序，点击切换到倒序" : "当前倒序，点击切换到正序"}
                    side="bottom"
                  >
                    <Button
                      type="button"
                      variant="outline"
                      size="icon-sm"
                      aria-label={sortDirection === "asc" ? "当前正序，点击切换到倒序" : "当前倒序，点击切换到正序"}
                      onClick={() => setSortDirection((current) => (current === "asc" ? "desc" : "asc"))}
                      disabled={isBusy}
                    >
                      <ArrowUpDown className={cn("size-3.5 transition-transform", sortDirection === "desc" ? "rotate-180" : "")} />
                    </Button>
                  </Tooltip>
                </div>
              </div>
              <div className="grid gap-3">
                <CardDescription>
                  支持按字母或论文时间在前端排序，点击左侧条目后在右侧查看摘要、展示名和 OCR 预览。
                </CardDescription>
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="gap-2"
                    onClick={() => void handleManualRefresh()}
                    disabled={manualRefreshing || isBusy}
                  >
                    <RefreshCw className={cn("size-4", manualRefreshing ? "animate-spin" : "")} />
                    刷新状态
                  </Button>
                  {canWritePapers ? (
                    <Button
                      type="button"
                      size="sm"
                      className="gap-2"
                      onClick={() => setUploadDialogOpen(true)}
                      disabled={isBusy}
                    >
                      <Upload className="size-4" />
                      上传 PDF
                    </Button>
                  ) : null}
                </div>
              </div>
            </div>
          </CardHeader>
          <CardContent className="grid min-h-0 gap-3 overflow-y-auto">
            {loadingList ? (
              <div className="flex items-center gap-2 rounded-xl border border-dashed border-slate-200 px-4 py-6 text-sm text-slate-500">
                <LoaderCircle className="size-4 animate-spin" />
                正在加载论文列表...
              </div>
            ) : null}

            {listError ? <p className="text-sm text-destructive">{listError}</p> : null}

            {!loadingList && papers.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-4 py-10 text-sm text-slate-500">
                还没有归档论文，先上传一个 PDF 开始。
              </div>
            ) : null}

            {sortedPapers.map((paper) => (
              <button
                key={paper.id}
                type="button"
                onClick={() => onSelectedPaperChange(paper.id)}
                disabled={isBusy}
                className={cn(
                  "grid gap-3 rounded-2xl border px-4 py-4 text-left transition disabled:cursor-not-allowed disabled:opacity-70",
                  selectedPaperId === paper.id
                    ? "border-slate-400 bg-slate-200 text-slate-900 shadow-sm"
                    : "border-slate-200 bg-slate-100 text-slate-900 hover:border-slate-300 hover:bg-slate-50"
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-medium leading-6">{paper.title || `未命名论文 #${paper.id}`}</p>
                    <p className={cn("mt-1 text-xs", selectedPaperId === paper.id ? "text-slate-600" : "text-slate-500")}>
                      原始文件名：{paper.original_filename || "-"}
                    </p>
                    <p className={cn("mt-1 text-xs", selectedPaperId === paper.id ? "text-slate-600" : "text-slate-500")}>
                      论文时间：{paper.published_at ? formatTime(paper.published_at) : "-"}
                    </p>
                    <p className={cn("mt-1 text-xs", selectedPaperId === paper.id ? "text-slate-600" : "text-slate-500")}>
                      更新于 {formatTime(paper.updated_at)}
                    </p>
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-1.5">
                    <PhaseBadge
                      label="OCR"
                      status={paper.ocr_status}
                      className={selectedPaperId === paper.id ? "bg-white/80 text-slate-700 ring-slate-300" : undefined}
                    />
                    <PhaseBadge
                      label="摘要"
                      status={paper.summary_status}
                      className={selectedPaperId === paper.id ? "bg-white/80 text-slate-700 ring-slate-300" : undefined}
                    />
                  </div>
                </div>
                <p className={cn("text-sm leading-6", selectedPaperId === paper.id ? "text-slate-700" : "text-slate-600")}>
                  {excerpt(paper.abstract_raw)}
                </p>
              </button>
            ))}
          </CardContent>
        </Card>
        <Card className="min-h-0 border-none bg-white/85 shadow-sm ring-1 ring-slate-200 backdrop-blur">
          <CardHeader className="gap-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <ScanText className="size-4" />
                  OCR / 摘要预览
                </CardTitle>
                <CardDescription>查看当前选中论文的展示名、原始文件名、摘要信息和 OCR 文本内容。</CardDescription>
              </div>
              <Tabs
                value={previewTab}
                onValueChange={setPreviewTab}
                items={[
                  { value: "summary", label: "摘要和文档信息" },
                  { value: "ocr", label: "OCR 文本预览" },
                ]}
              />
            </div>
          </CardHeader>
          <CardContent className="min-h-0 overflow-hidden">
            {loadingDetail ? (
              <div className="flex items-center gap-2 rounded-xl border border-dashed border-slate-200 px-4 py-6 text-sm text-slate-500">
                <LoaderCircle className="size-4 animate-spin" />
                正在加载论文详情...
              </div>
            ) : null}

            {detailError ? <p className="text-sm text-destructive">{detailError}</p> : null}

            {!selectedPaperId && !loadingDetail ? (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-6 py-12 text-center text-sm text-slate-500">
                从左侧选择一篇论文，即可查看 OCR 预览和摘要结果。
              </div>
            ) : null}

            {paperDetail && !loadingDetail ? (
              <div className="relative h-full overflow-y-auto pr-1">
                {actionLoading ? (
                  <div className="absolute inset-0 z-10 flex items-center justify-center rounded-2xl bg-white/65 backdrop-blur-[1px]">
                    <div className="flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm text-slate-600 shadow-sm ring-1 ring-slate-200">
                      <LoaderCircle className="size-4 animate-spin" />
                      正在提交请求...
                    </div>
                  </div>
                ) : null}
                <TabPanel active={previewTab === "summary"} className="grid gap-5">
                  <div className="grid gap-4 rounded-2xl border border-slate-200 bg-slate-50/80 p-5">
                    <div className="flex items-start justify-between gap-4">
                      <div className="grid min-w-0 flex-1 gap-3">
                        <div>
                          <h2 className="text-xl font-semibold text-slate-950">{paperDetail.title || `未命名论文 #${paperDetail.id}`}</h2>
                          <p className="mt-1 text-sm text-slate-500">
                            原始文件名：{paperDetail.original_filename || "-"} | 更新时间：{formatTime(paperDetail.updated_at)}
                          </p>
                        </div>
                        <div className="flex flex-wrap gap-3">
                          {canWritePapers ? (
                            <Button type="button" variant="outline" size="sm" className="gap-2" onClick={() => setMetadataDialogOpen(true)} disabled={isBusy}>
                              <FilePenLine className="size-4" />
                              编辑文档信息
                            </Button>
                          ) : null}
                          {canDeletePapers ? (
                            <Button
                              type="button"
                              variant="destructive"
                              size="sm"
                              className="gap-2"
                              onClick={() => setDeleteConfirmOpen(true)}
                              disabled={hasActiveJob || isBusy}
                            >
                              <Trash2 className="size-4" />
                              删除文档
                            </Button>
                          ) : null}
                        </div>
                        {actionError ? <p className="text-sm text-destructive">{actionError}</p> : null}
                      </div>
                      <div className="grid shrink-0 gap-2">
                        <PhaseBadge
                          label="OCR"
                          status={paperDetail.ocr_status}
                          action={
                            canWritePapers
                              ? {
                                  tooltip: "重新执行 OCR",
                                  ariaLabel: "重新执行 OCR",
                                  disabled: hasActiveJob || isBusy,
                                  loading: actionLoading === "rerun-ocr",
                                  onClick: () => setConfirmAction("rerun-ocr"),
                                }
                              : undefined
                          }
                        />
                        <PhaseBadge
                          label="摘要"
                          status={paperDetail.summary_status}
                          action={
                            canWritePapers
                              ? {
                                  tooltip: "重新生成摘要",
                                  ariaLabel: "重新生成摘要",
                                  disabled: hasActiveJob || isBusy,
                                  loading: actionLoading === "regenerate-summary",
                                  onClick: () => setConfirmAction("regenerate-summary"),
                                }
                              : undefined
                          }
                        />
                      </div>
                    </div>

                    <section className="grid gap-3 md:grid-cols-2">
                      <SummaryBlock
                        label="作者"
                        value={paperDetail.authors || "-"}
                        sourceLabel={getMetadataSourceLabel({
                          currentValue: paperDetail.authors,
                          extractedValue: paperDetail.structured_summary?.authors,
                        })}
                      />
                      <SummaryBlock
                        label="DOI"
                        value={paperDetail.doi || "-"}
                        sourceLabel={getMetadataSourceLabel({
                          currentValue: paperDetail.doi,
                          extractedValue: paperDetail.structured_summary?.doi,
                        })}
                      />
                      <SummaryBlock
                        label="来源链接"
                        value={paperDetail.source_url || "-"}
                        sourceLabel={getMetadataSourceLabel({
                          currentValue: paperDetail.source_url,
                          extractedValue: paperDetail.structured_summary?.source_url,
                        })}
                      />
                      <SummaryBlock
                        label="发布时间"
                        value={paperDetail.published_at ? formatTime(paperDetail.published_at) : "-"}
                        sourceLabel={getMetadataSourceLabel({
                          currentValue: paperDetail.published_at,
                          extractedValue: paperDetail.structured_summary?.published_at,
                          isDate: true,
                        })}
                      />
                    </section>

                    <section className="grid gap-2">
                      <h3 className="text-sm font-medium text-slate-700">中文摘要</h3>
                      <p className="rounded-2xl bg-white px-4 py-4 leading-7 text-slate-700 shadow-sm ring-1 ring-slate-200">
                        {paperDetail.abstract_raw || "当前还没有生成摘要，可先检查模型配置是否已完成。"}
                      </p>
                    </section>

                    {paperDetail.structured_summary ? (
                      <section className="grid gap-3">
                        <h3 className="text-sm font-medium text-slate-700">结构化信息</h3>
                        <div className="grid gap-3 md:grid-cols-2">
                          <SummaryBlock label="研究问题" value={paperDetail.structured_summary.research_question} />
                          <SummaryBlock label="方法" value={paperDetail.structured_summary.method} />
                          <SummaryBlock label="主要发现" value={paperDetail.structured_summary.findings} />
                          <SummaryBlock label="局限性" value={paperDetail.structured_summary.limitations} />
                        </div>
                        {paperDetail.structured_summary.key_points.length > 0 ? (
                          <div className="rounded-2xl bg-white px-4 py-4 shadow-sm ring-1 ring-slate-200">
                            <h4 className="text-sm font-medium text-slate-700">要点总结</h4>
                            <ul className="mt-3 grid gap-2 text-sm text-slate-600">
                              {paperDetail.structured_summary.key_points.map((point) => (
                                <li key={point}>- {point}</li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                      </section>
                    ) : null}
                  </div>
                </TabPanel>

                <TabPanel active={previewTab === "ocr"} className="grid gap-3">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h3 className="text-sm font-medium text-slate-700">文本 / OCR 预览</h3>
                      <p className="text-xs text-slate-500">
                        总页数：{paperDetail.extraction_metadata?.page_count ?? "-"}，OCR 页：{" "}
                        {(paperDetail.extraction_metadata?.used_ocr_pages || []).join(", ") || "-"}
                      </p>
                    </div>
                  </div>
                  <div className="min-h-[420px] rounded-2xl border border-slate-200 bg-slate-950 px-5 py-5 font-mono text-sm leading-7 text-slate-200 shadow-inner">
                    <pre className="overflow-x-auto whitespace-pre-wrap break-words">
                      {paperDetail.preview_text || "当前还没有可预览的文本，任务完成后会在这里显示。"}
                    </pre>
                  </div>
                </TabPanel>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </>
  );
}

function SummaryBlock({
  label,
  value,
  sourceLabel = null,
}: {
  label: string;
  value: string;
  sourceLabel?: MetadataSourceLabel;
}) {
  return (
    <div className="rounded-2xl bg-white px-4 py-4 shadow-sm ring-1 ring-slate-200">
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-sm font-medium text-slate-700">{label}</h4>
        {sourceLabel ? (
          <span
            className={cn(
              "rounded-full px-2.5 py-1 text-xs ring-1",
              sourceLabel === "摘要自动提取"
                ? "bg-emerald-500/10 text-emerald-700 ring-emerald-500/20"
                : "bg-slate-200/80 text-slate-600 ring-slate-300"
            )}
          >
            来源：{sourceLabel}
          </span>
        ) : null}
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-600">{value || "-"}</p>
    </div>
  );
}
