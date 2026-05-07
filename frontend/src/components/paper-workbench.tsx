"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { FilePenLine, LoaderCircle, RefreshCw, ScanText, Trash2, Upload } from "lucide-react";

import { PaperMetadataDialog } from "@/components/paper-metadata-dialog";
import { PaperUploadDialog } from "@/components/paper-upload-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog } from "@/components/ui/dialog";
import { TabPanel, Tabs } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import {
  deletePaper,
  fetchPaper,
  fetchPapers,
  regeneratePaperSummary,
  rerunPaperOcr,
  type JobPublic,
  type PaperDetail,
  type PaperListItem,
  type UploadAcceptedResponse,
} from "@/lib/papers";

const ACTIVE_STATUSES = new Set(["queued", "processing"]);

function statusLabel(status: string | null): string {
  if (status === "queued") return "排队中";
  if (status === "processing") return "处理中";
  if (status === "completed") return "已完成";
  if (status === "failed") return "失败";
  if (status === "uploaded") return "已上传";
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

function excerpt(value: string | null, maxLength = 96): string {
  const text = (value || "").trim();
  if (!text) return "尚未生成摘要";
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

type PaperWorkbenchProps = {
  selectedPaperId: number | null;
  onSelectedPaperChange: (paperId: number | null) => void;
};

export function PaperWorkbench({ selectedPaperId, onSelectedPaperChange }: PaperWorkbenchProps) {
  const [papers, setPapers] = useState<PaperListItem[]>([]);
  const [paperDetail, setPaperDetail] = useState<PaperDetail | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [metadataDialogOpen, setMetadataDialogOpen] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [previewTab, setPreviewTab] = useState<"summary" | "ocr">("summary");
  const [actionLoading, setActionLoading] = useState<"rerun-ocr" | "regenerate-summary" | "delete" | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const loadPapers = useCallback(
    async (preferredPaperId?: number | null) => {
      setLoadingList(true);
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
        setLoadingList(false);
      }
    },
    [onSelectedPaperChange, selectedPaperId]
  );

  const loadDetail = useCallback(async (paperId?: number | null) => {
    const targetPaperId = paperId ?? selectedPaperId;
    if (!targetPaperId) {
      setPaperDetail(null);
      return;
    }

    setLoadingDetail(true);
    try {
      const detail = await fetchPaper(targetPaperId);
      setPaperDetail(detail);
      setDetailError(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "获取论文详情失败";
      setDetailError(message);
    } finally {
      setLoadingDetail(false);
    }
  }, [selectedPaperId]);

  useEffect(() => {
    void loadPapers();
  }, [loadPapers]);

  useEffect(() => {
    void loadDetail();
  }, [loadDetail]);

  const shouldPoll = useMemo(() => {
    if (papers.some((paper) => ACTIVE_STATUSES.has(paper.status ?? ""))) {
      return true;
    }
    return ACTIVE_STATUSES.has(paperDetail?.latest_job?.status ?? "");
  }, [paperDetail?.latest_job?.status, papers]);

  useEffect(() => {
    if (!shouldPoll) return;

    const timer = window.setTimeout(() => {
      void loadPapers(selectedPaperId);
      void loadDetail();
    }, 3000);

    return () => window.clearTimeout(timer);
  }, [loadDetail, loadPapers, selectedPaperId, shouldPoll]);

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
    return ACTIVE_STATUSES.has(paperDetail.status ?? "") || ACTIVE_STATUSES.has(paperDetail.latest_job?.status ?? "");
  }, [paperDetail]);

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
      const nextPaperId = remaining[0]?.id ?? null;

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

      <div className="grid h-full min-h-0 gap-6 xl:grid-cols-[minmax(320px,420px)_minmax(0,1fr)]">
        <Card className="min-h-0 border-none bg-white/80 shadow-sm ring-1 ring-slate-200 backdrop-blur">
          <CardHeader className="gap-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <CardTitle>论文列表</CardTitle>
                <CardDescription>按更新时间倒序展示，点击左侧条目后在右侧查看摘要、展示名和 OCR 预览。</CardDescription>
              </div>
              <Button type="button" size="sm" className="gap-2" onClick={() => setUploadDialogOpen(true)} disabled={isBusy}>
                <Upload className="size-4" />
                上传 PDF
              </Button>
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

            {papers.map((paper) => (
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
                      更新于 {formatTime(paper.updated_at)}
                    </p>
                  </div>
                  <span
                    className={cn(
                      "min-w-[72px] shrink-0 whitespace-nowrap rounded-full px-3 py-1 text-center text-xs font-medium ring-1",
                      selectedPaperId === paper.id ? "bg-white/80 text-slate-700 ring-slate-300" : getStatusClassName(paper.status)
                    )}
                  >
                    {statusLabel(paper.status)}
                  </span>
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
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="grid gap-3">
                        <div>
                          <h2 className="text-xl font-semibold text-slate-950">{paperDetail.title || `未命名论文 #${paperDetail.id}`}</h2>
                          <p className="mt-1 text-sm text-slate-500">
                            原始文件名：{paperDetail.original_filename || "-"} | 更新时间：{formatTime(paperDetail.updated_at)}
                          </p>
                        </div>
                        <div className="flex flex-wrap gap-3">
                          <Button type="button" variant="outline" size="sm" className="gap-2" onClick={() => setMetadataDialogOpen(true)} disabled={isBusy}>
                            <FilePenLine className="size-4" />
                            编辑文档信息
                          </Button>
                          <Button type="button" variant="outline" size="sm" className="gap-2" onClick={() => void handleRerunOcr()} disabled={hasActiveJob || isBusy}>
                            <RefreshCw className={cn("size-4", actionLoading === "rerun-ocr" ? "animate-spin" : "")} />
                            重新 OCR
                          </Button>
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            className="gap-2"
                            onClick={() => void handleRegenerateSummary()}
                            disabled={hasActiveJob || isBusy}
                          >
                            <RefreshCw className={cn("size-4", actionLoading === "regenerate-summary" ? "animate-spin" : "")} />
                            重新生成摘要
                          </Button>
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
                        </div>
                        {actionError ? <p className="text-sm text-destructive">{actionError}</p> : null}
                      </div>
                      <div className="grid gap-2">
                        <span className={cn("w-fit rounded-full px-3 py-1 text-xs font-medium ring-1", getStatusClassName(paperDetail.status))}>
                          {statusLabel(paperDetail.status)}
                        </span>
                        {paperDetail.latest_job ? (
                          <span
                            className={cn(
                              "w-fit rounded-full px-3 py-1 text-xs font-medium ring-1",
                              getStatusClassName(paperDetail.latest_job.status)
                            )}
                          >
                            最近任务：{statusLabel(paperDetail.latest_job.status)}
                          </span>
                        ) : null}
                      </div>
                    </div>

                    <section className="grid gap-3 md:grid-cols-2">
                      <SummaryBlock label="作者" value={paperDetail.authors || "-"} />
                      <SummaryBlock label="DOI" value={paperDetail.doi || "-"} />
                      <SummaryBlock label="来源链接" value={paperDetail.source_url || "-"} />
                      <SummaryBlock label="发布时间" value={paperDetail.published_at ? formatTime(paperDetail.published_at) : "-"} />
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

function SummaryBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-white px-4 py-4 shadow-sm ring-1 ring-slate-200">
      <h4 className="text-sm font-medium text-slate-700">{label}</h4>
      <p className="mt-2 text-sm leading-6 text-slate-600">{value || "-"}</p>
    </div>
  );
}
