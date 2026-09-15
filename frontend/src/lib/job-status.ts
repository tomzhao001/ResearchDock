const ACTIVE_STATUSES = new Set(["queued", "processing", "cancel_requested"]);
const STOPPABLE_STATUSES = new Set(["queued", "processing"]);
const HIDDEN_TASK_LIST_STATUSES = new Set(["completed", "cancelled"]);
const DELETABLE_STATUSES = new Set(["completed", "failed", "cancelled"]);

const STATUS_LABELS: Record<string, string> = {
  queued: "排队中",
  processing: "处理中",
  cancel_requested: "取消中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

const STATUS_CLASS_NAMES: Record<string, string> = {
  queued: "bg-sky-500/15 text-sky-700 ring-sky-500/20",
  processing: "bg-amber-500/15 text-amber-700 ring-amber-500/20",
  cancel_requested: "bg-amber-500/15 text-amber-700 ring-amber-500/20",
  completed: "bg-emerald-500/15 text-emerald-700 ring-emerald-500/20",
  failed: "bg-rose-500/15 text-rose-700 ring-rose-500/20",
  cancelled: "bg-slate-200/80 text-slate-600 ring-slate-300",
};

const EMPTY_STATUS_CLASS_NAME = "bg-slate-200/80 text-slate-600 ring-slate-300";

export function getJobStatusLabel(
  status: string | null,
  options?: { variant?: "upload"; emptyLabel?: string }
): string {
  if (!status) {
    return options?.emptyLabel ?? "";
  }
  return STATUS_LABELS[status] ?? status;
}

export function getJobStatusClassName(status: string | null): string {
  if (!status) {
    return EMPTY_STATUS_CLASS_NAME;
  }
  return STATUS_CLASS_NAMES[status] ?? EMPTY_STATUS_CLASS_NAME;
}

export function isActiveJobStatus(status: string | null): boolean {
  return status !== null && ACTIVE_STATUSES.has(status);
}

export function isDeletableJobStatus(status: string | null): boolean {
  return status !== null && DELETABLE_STATUSES.has(status);
}

export function isHiddenTaskListJobStatus(status: string | null): boolean {
  return status !== null && HIDDEN_TASK_LIST_STATUSES.has(status);
}

export function isStoppableJobStatus(status: string | null): boolean {
  return status !== null && STOPPABLE_STATUSES.has(status);
}
