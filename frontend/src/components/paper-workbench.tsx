"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { LoaderCircle, PencilLine, Save, ScanText, Upload, X } from "lucide-react";

import { PaperUploadDialog } from "@/components/paper-upload-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { TabPanel, Tabs } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import {
  fetchPaper,
  fetchPapers,
  updatePaperTitle,
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
  const [previewTab, setPreviewTab] = useState<"summary" | "ocr">("summary");
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const [titleError, setTitleError] = useState<string | null>(null);
  const [titleSaving, setTitleSaving] = useState(false);

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

  useEffect(() => {
    setTitleDraft(paperDetail?.title ?? "");
    setEditingTitle(false);
    setTitleError(null);
  }, [paperDetail?.id, paperDetail?.title]);

  async function handleSaveTitle() {
    if (!paperDetail) {
      return;
    }

    setTitleSaving(true);
    setTitleError(null);
    try {
      const nextDetail = await updatePaperTitle(paperDetail.id, titleDraft);
      setPaperDetail(nextDetail);
      setEditingTitle(false);
      await loadPapers(nextDetail.id);
    } catch (error) {
      const message = error instanceof Error ? error.message : "保存失败";
      setTitleError(message);
    } finally {
      setTitleSaving(false);
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

      <div className="grid h-full min-h-0 gap-6 xl:grid-cols-[minmax(320px,420px)_minmax(0,1fr)]">
        <Card className="min-h-0 border-none bg-white/80 shadow-sm ring-1 ring-slate-200 backdrop-blur">
          <CardHeader className="gap-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <CardTitle>论文列表</CardTitle>
                <CardDescription>按更新时间倒序展示，点击左侧条目后在右侧查看摘要、展示名和 OCR 预览。</CardDescription>
              </div>
              <Button type="button" size="sm" className="gap-2" onClick={() => setUploadDialogOpen(true)}>
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
                className={cn(
                  "grid gap-3 rounded-2xl border px-4 py-4 text-left transition hover:-translate-y-0.5 hover:shadow-sm",
                  selectedPaperId === paper.id
                    ? "border-slate-900 bg-slate-950 text-slate-50 shadow-sm"
                    : "border-slate-200 bg-white text-slate-900"
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-medium leading-6">{paper.title || `未命名论文 #${paper.id}`}</p>
                    <p className={cn("mt-1 text-xs", selectedPaperId === paper.id ? "text-slate-300" : "text-slate-500")}>
                      原始文件名：{paper.original_filename || "-"}
                    </p>
                    <p className={cn("mt-1 text-xs", selectedPaperId === paper.id ? "text-slate-300" : "text-slate-500")}>
                      更新于 {formatTime(paper.updated_at)}
                    </p>
                  </div>
                  <span
                    className={cn(
                      "rounded-full px-2.5 py-1 text-xs font-medium ring-1",
                      selectedPaperId === paper.id ? "bg-white/10 text-white ring-white/15" : getStatusClassName(paper.status)
                    )}
                  >
                    {statusLabel(paper.status)}
                  </span>
                </div>
                <p className={cn("text-sm leading-6", selectedPaperId === paper.id ? "text-slate-200" : "text-slate-600")}>
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
              <div className="h-full overflow-y-auto pr-1">
                <TabPanel active={previewTab === "summary"} className="grid gap-5">
                  <div className="grid gap-4 rounded-2xl border border-slate-200 bg-slate-50/80 p-5">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="grid gap-3">
                        {!editingTitle ? (
                          <>
                            <div>
                              <h2 className="text-xl font-semibold text-slate-950">{paperDetail.title || `未命名论文 #${paperDetail.id}`}</h2>
                              <p className="mt-1 text-sm text-slate-500">
                                原始文件名：{paperDetail.original_filename || "-"} | 更新时间：{formatTime(paperDetail.updated_at)}
                              </p>
                            </div>
                            <div className="flex flex-wrap gap-3">
                              <Button type="button" variant="outline" size="sm" className="gap-2" onClick={() => setEditingTitle(true)}>
                                <PencilLine className="size-4" />
                                编辑展示名
                              </Button>
                            </div>
                          </>
                        ) : (
                          <div className="grid gap-3">
                            <label className="grid gap-2 text-sm font-medium text-slate-700">
                              展示名
                              <input
                                value={titleDraft}
                                onChange={(event) => setTitleDraft(event.target.value)}
                                className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:ring-4 focus:ring-slate-200"
                              />
                            </label>
                            {titleError ? <p className="text-sm text-destructive">{titleError}</p> : null}
                            <div className="flex flex-wrap gap-3">
                              <Button type="button" size="sm" className="gap-2" onClick={() => void handleSaveTitle()} disabled={titleSaving}>
                                <Save className="size-4" />
                                {titleSaving ? "保存中..." : "保存"}
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                className="gap-2"
                                onClick={() => {
                                  setEditingTitle(false);
                                  setTitleDraft(paperDetail.title ?? "");
                                  setTitleError(null);
                                }}
                                disabled={titleSaving}
                              >
                                <X className="size-4" />
                                取消
                              </Button>
                            </div>
                          </div>
                        )}
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
