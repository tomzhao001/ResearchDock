import { apiFetch, apiJson, readApiErrorMessage, websocketUrl } from "@/lib/api";

export type JobPublic = {
  id: number;
  job_type: string | null;
  paper_id: number | null;
  celery_task_id: string | null;
  status: string | null;
  error_message: string | null;
  retry_count: number;
  cancel_requested_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  deleted_at: string | null;
  created_at: string;
};

export type UploadAcceptedResponse = {
  paper_id: number;
  job_id: number;
  filename: string;
  status: string;
};

export type PaperListItem = {
  id: number;
  organization_id?: number;
  title: string | null;
  authors: string | null;
  original_filename: string | null;
  abstract_raw: string | null;
  published_at: string | null;
  status: string | null;
  ocr_status: string | null;
  summary_status: string | null;
  question_set_status: string | null;
  created_at: string;
  updated_at: string;
};

export type PaperDetail = {
  id: number;
  organization_id?: number;
  title: string | null;
  authors: string | null;
  abstract_raw: string | null;
  source_url: string | null;
  pdf_url: string | null;
  doi: string | null;
  published_at: string | null;
  status: string | null;
  ocr_status: string | null;
  summary_status: string | null;
  question_set_status: string | null;
  created_at: string;
  updated_at: string;
  original_filename: string | null;
  preview_text: string | null;
  extraction_metadata: {
    page_count?: number | null;
    block_count?: number | null;
    table_count?: number | null;
    picture_count?: number | null;
  } | null;
  structured_summary: {
    abstract_cn?: string;
    key_points: string[];
    research_question: string;
    method: string;
    findings: string;
    limitations: string;
    authors?: string | null;
    doi?: string | null;
    source_url?: string | null;
    published_at?: string | null;
  } | null;
  question_set_extraction: {
    generated_at?: string;
    model_name?: string;
    questions: Array<{ id: string; question: string; answer: string }>;
  } | null;
  latest_job: JobPublic | null;
  latest_ocr_job: JobPublic | null;
  latest_summary_job: JobPublic | null;
  latest_question_set_job: JobPublic | null;
};

export type TaskStatusEvent = {
  type?: "task-status";
  paper_id: number;
  job_id?: number | null;
  job_type?: string | null;
  job_status?: string | null;
  paper_status?: string | null;
  ocr_status?: string | null;
  summary_status?: string | null;
  question_set_status?: string | null;
  error_message?: string | null;
  updated_at?: string;
  job?: JobPublic | null;
  paper_list_item: PaperListItem;
  paper_detail: PaperDetail;
};

export type UploadConflictDetail = {
  message: string;
  existing_paper_id: number;
  filename: string;
};

export class UploadConflictError extends Error {
  detail: UploadConflictDetail;

  constructor(detail: UploadConflictDetail) {
    super(detail.message);
    this.name = "UploadConflictError";
    this.detail = detail;
  }
}

type JobAcceptedResponse = {
  paper_id: number;
  job_id: number;
  job_type: string;
  status: string;
};

async function apiList<T>(input: string): Promise<T[]> {
  const payload = await apiJson<{ items: T[] }>(input);
  return payload.items ?? [];
}

export async function uploadPaper(file: File, overwrite = false): Promise<UploadAcceptedResponse> {
  const body = new FormData();
  body.append("file", file);
  body.append("overwrite", overwrite ? "true" : "false");
  const response = await apiFetch("/api/papers/upload", { method: "POST", body });
  if (response.status === 409) {
    const payload = (await response.json().catch(() => null)) as { detail?: UploadConflictDetail } | null;
    if (payload?.detail && typeof payload.detail === "object") {
      throw new UploadConflictError(payload.detail);
    }
    throw new UploadConflictError({
      message: "已有相同文件名的文档，是否需要覆盖上传？",
      existing_paper_id: 0,
      filename: file.name,
    });
  }
  if (!response.ok) {
    throw new Error(await readApiErrorMessage(response));
  }
  return (await response.json()) as UploadAcceptedResponse;
}

export function fetchJob(jobId: number): Promise<JobPublic> {
  return apiJson<JobPublic>(`/api/jobs/${jobId}`);
}

export function fetchJobs(): Promise<JobPublic[]> {
  return apiList<JobPublic>("/api/jobs");
}

export function cancelJob(jobId: number): Promise<JobPublic> {
  return apiJson<JobPublic>(`/api/jobs/${jobId}/cancel`, { method: "POST" });
}

export async function deleteJob(jobId: number): Promise<void> {
  await apiJson(`/api/jobs/${jobId}`, { method: "DELETE" });
}

export function fetchPapers(): Promise<PaperListItem[]> {
  return apiList<PaperListItem>("/api/papers");
}

export function fetchPaper(paperId: number): Promise<PaperDetail> {
  return apiJson<PaperDetail>(`/api/papers/${paperId}`);
}

export async function deletePaper(paperId: number): Promise<void> {
  await apiJson(`/api/papers/${paperId}`, { method: "DELETE" });
}

export function reparsePaperDocument(paperId: number): Promise<JobAcceptedResponse> {
  return apiJson<JobAcceptedResponse>(`/api/papers/${paperId}/reparse-document`, { method: "POST" });
}

export function regeneratePaperSummary(paperId: number): Promise<JobAcceptedResponse> {
  return apiJson<JobAcceptedResponse>(`/api/papers/${paperId}/regenerate-summary`, { method: "POST" });
}

export function regeneratePaperQuestionSet(paperId: number): Promise<JobAcceptedResponse> {
  return apiJson<JobAcceptedResponse>(`/api/papers/${paperId}/regenerate-question-set`, { method: "POST" });
}

export function updatePaperMetadata(
  paperId: number,
  patch: {
    title: string;
    authors: string;
    doi: string;
    source_url: string;
    published_at: string | null;
  }
): Promise<PaperDetail> {
  return apiJson<PaperDetail>(`/api/papers/${paperId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
}

export function subscribeTaskStatusEvents(options: {
  onEvent: (event: TaskStatusEvent) => void;
}): () => void {
  const socket = new WebSocket(websocketUrl("/api/ws/tasks"));
  socket.onmessage = (message) => {
    try {
      options.onEvent(JSON.parse(message.data) as TaskStatusEvent);
    } catch {
      return;
    }
  };
  return () => socket.close();
}
