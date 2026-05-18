"use client";

import { useEffect, useMemo, useState } from "react";
import { FileQuestion, PencilLine, Plus, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  fetchOrganizationQuestionSet,
  updateOrganizationQuestionSet,
  type OrganizationQuestionItem,
} from "@/lib/org-settings";
import { useHasPermission } from "@/lib/session";

type EditorState = {
  mode: "create" | "edit";
  itemId: string | null;
};

function formatTime(value: string | null): string {
  if (!value) {
    return "-";
  }
  return new Date(value).toLocaleString("zh-CN");
}

function createQuestionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `question-${Date.now()}`;
}

export function OrgQuestionSetPanel() {
  const canWrite = useHasPermission("org_settings:write");
  const [questions, setQuestions] = useState<OrganizationQuestionItem[]>([]);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editorState, setEditorState] = useState<EditorState | null>(null);
  const [deleteItem, setDeleteItem] = useState<OrganizationQuestionItem | null>(null);
  const [draftQuestion, setDraftQuestion] = useState("");

  const editingItem = useMemo(
    () => questions.find((item) => item.id === editorState?.itemId) ?? null,
    [editorState?.itemId, questions]
  );

  useEffect(() => {
    void loadQuestionSet();
  }, []);

  useEffect(() => {
    if (!editorState) {
      setDraftQuestion("");
      return;
    }
    setDraftQuestion(editingItem?.question ?? "");
  }, [editingItem, editorState]);

  async function loadQuestionSet() {
    setLoading(true);
    try {
      const data = await fetchOrganizationQuestionSet();
      setQuestions(data.questions);
      setUpdatedAt(data.updated_at);
      setError(null);
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : "获取组织问题集失败";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  async function saveQuestions(nextQuestions: OrganizationQuestionItem[]) {
    setSaving(true);
    setError(null);
    try {
      const data = await updateOrganizationQuestionSet({ questions: nextQuestions });
      setQuestions(data.questions);
      setUpdatedAt(data.updated_at);
      setEditorState(null);
      setDeleteItem(null);
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "保存组织问题集失败";
      setError(message);
    } finally {
      setSaving(false);
    }
  }

  async function handleSaveDraft() {
    const normalizedQuestion = draftQuestion.trim();
    if (!normalizedQuestion) {
      setError("问题内容不能为空");
      return;
    }

    if (editorState?.mode === "edit" && editingItem) {
      await saveQuestions(
        questions.map((item) =>
          item.id === editingItem.id
            ? {
                ...item,
                question: normalizedQuestion,
              }
            : item
        )
      );
      return;
    }

    await saveQuestions([
      ...questions,
      {
        id: createQuestionId(),
        question: normalizedQuestion,
      },
    ]);
  }

  async function handleDeleteItem() {
    if (!deleteItem) {
      return;
    }
    await saveQuestions(questions.filter((item) => item.id !== deleteItem.id));
  }

  return (
    <>
      <Dialog
        open={editorState !== null}
        onClose={() => {
          if (saving) {
            return;
          }
          setEditorState(null);
        }}
        title={editorState?.mode === "edit" ? "编辑问题" : "新增问题"}
        description="更新后不会自动同步历史论文的抽取结果；如需让已有论文使用最新问题集，请重新执行该论文的问题集抽取。"
        className="max-w-xl"
      >
        <div className="grid gap-4 p-6">
          <label className="grid gap-2 text-sm font-medium text-slate-700">
            <span>问题内容</span>
            <Input value={draftQuestion} onChange={(event) => setDraftQuestion(event.target.value)} placeholder="例如：这篇论文解决了什么研究问题？" />
          </label>
          <div className="flex justify-end gap-3">
            <Button type="button" variant="outline" onClick={() => setEditorState(null)} disabled={saving}>
              取消
            </Button>
            <Button type="button" onClick={() => void handleSaveDraft()} disabled={saving}>
              {saving ? "保存中..." : editorState?.mode === "edit" ? "保存修改" : "确认新增"}
            </Button>
          </div>
        </div>
      </Dialog>

      <Dialog
        open={deleteItem !== null}
        onClose={() => {
          if (saving) {
            return;
          }
          setDeleteItem(null);
        }}
        title="确认删除问题"
        description="删除后不会自动同步历史论文的抽取结果；如需让已有论文使用最新问题集，请重新执行该论文的问题集抽取。"
        className="max-w-lg"
      >
        <div className="grid gap-4 p-6">
          <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-4 text-sm text-amber-900">
            {deleteItem ? `即将删除问题：${deleteItem.question}` : "即将删除当前问题。"}
          </div>
          <div className="flex justify-end gap-3">
            <Button type="button" variant="outline" onClick={() => setDeleteItem(null)} disabled={saving}>
              取消
            </Button>
            <Button type="button" variant="destructive" onClick={() => void handleDeleteItem()} disabled={saving}>
              {saving ? "删除中..." : "确认删除"}
            </Button>
          </div>
        </div>
      </Dialog>

      <Card className="h-full min-h-0 border-none bg-white/85 shadow-sm ring-1 ring-slate-200 backdrop-blur">
        <CardHeader className="gap-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="grid gap-2">
              <CardTitle className="flex items-center gap-2 text-lg">
                <FileQuestion className="size-4" />
                组织级问题集
              </CardTitle>
              <p className="text-xs text-slate-500">最近更新：{formatTime(updatedAt)}</p>
            </div>
            {canWrite ? (
              <Button type="button" size="sm" className="gap-2" onClick={() => setEditorState({ mode: "create", itemId: null })} disabled={loading || saving}>
                <Plus className="size-4" />
                新增问题
              </Button>
            ) : null}
          </div>
        </CardHeader>
        <CardContent className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto">
          {loading ? <p className="text-sm text-slate-500">正在加载组织问题集...</p> : null}
          {error ? <p className="text-sm text-destructive">{error}</p> : null}
          {!loading && questions.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-10 text-sm text-slate-500">
              当前组织还没有配置问题集。
            </div>
          ) : null}
          {questions.map((item, index) => (
            <div key={item.id} className="rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-4 shadow-sm">
              <div className="flex items-start justify-between gap-3">
                <div className="grid gap-1">
                  <p className="text-xs font-medium tracking-wide text-slate-500 uppercase">问题 {index + 1}</p>
                  <p className="text-sm leading-6 text-slate-700">{item.question}</p>
                </div>
                {canWrite ? (
                  <div className="flex items-center gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="icon-sm"
                      aria-label={`编辑问题 ${index + 1}`}
                      onClick={() => setEditorState({ mode: "edit", itemId: item.id })}
                      disabled={saving}
                    >
                      <PencilLine className="size-3.5" />
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="icon-sm"
                      aria-label={`删除问题 ${index + 1}`}
                      onClick={() => setDeleteItem(item)}
                      disabled={saving}
                    >
                      <Trash2 className="size-3.5" />
                    </Button>
                  </div>
                ) : null}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </>
  );
}
