"use client";

import { useEffect, useState } from "react";

import { Dialog } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { updatePaperMetadata, type PaperDetail } from "@/lib/papers";

type PaperMetadataDialogProps = {
  open: boolean;
  paper: PaperDetail | null;
  onClose: () => void;
  onSaved: (detail: PaperDetail) => void;
};

function toDateInputValue(value: string | null): string {
  if (!value) {
    return "";
  }
  return value.split("T")[0] ?? "";
}

export function PaperMetadataDialog({ open, paper, onClose, onSaved }: PaperMetadataDialogProps) {
  const [title, setTitle] = useState("");
  const [authors, setAuthors] = useState("");
  const [doi, setDoi] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [publishedAt, setPublishedAt] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      return;
    }

    setTitle(paper?.title ?? "");
    setAuthors(paper?.authors ?? "");
    setDoi(paper?.doi ?? "");
    setSourceUrl(paper?.source_url ?? "");
    setPublishedAt(toDateInputValue(paper?.published_at ?? null));
    setSaving(false);
    setError(null);
  }, [open, paper]);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!paper) {
      return;
    }

    setSaving(true);
    setError(null);
    try {
      const nextDetail = await updatePaperMetadata(paper.id, {
        title,
        authors,
        doi,
        source_url: sourceUrl,
        published_at: publishedAt ? `${publishedAt}T00:00:00Z` : null,
      });
      onSaved(nextDetail);
      onClose();
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : "保存失败";
      setError(message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="编辑文档信息"
      description="可更新展示名、作者、DOI、来源链接和发布时间。原始文件名仍保持不变。"
    >
      <form className="grid gap-4 p-6" onSubmit={handleSubmit}>
        <label className="grid gap-2 text-sm font-medium text-slate-700">
          <span>展示名</span>
          <Input value={title} onChange={(event) => setTitle(event.target.value)} />
        </label>

        <label className="grid gap-2 text-sm font-medium text-slate-700">
          <span>作者</span>
          <Input value={authors} onChange={(event) => setAuthors(event.target.value)} placeholder="可选" />
        </label>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="grid gap-2 text-sm font-medium text-slate-700">
            <span>DOI</span>
            <Input value={doi} onChange={(event) => setDoi(event.target.value)} placeholder="可选" />
          </label>
          <div className="grid gap-2">
            <Label htmlFor="paper-published-at">发布时间</Label>
            <Input id="paper-published-at" type="date" value={publishedAt} onChange={(event) => setPublishedAt(event.target.value)} />
          </div>
        </div>

        <label className="grid gap-2 text-sm font-medium text-slate-700">
          <span>来源链接</span>
          <Input value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} placeholder="https://example.com/paper" />
        </label>

        {error ? <p className="text-sm text-destructive">{error}</p> : null}

        <div className="flex justify-end gap-3">
          <Button type="button" variant="outline" onClick={onClose} disabled={saving}>
            取消
          </Button>
          <Button type="submit" disabled={saving}>
            {saving ? "保存中..." : "保存"}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
