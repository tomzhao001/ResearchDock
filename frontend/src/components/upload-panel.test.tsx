import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { UploadPanel } from "@/components/upload-panel";
import { UploadConflictError, fetchJob, subscribeTaskStatusEvents, uploadPaper } from "@/lib/papers";

vi.mock("@/lib/papers", () => ({
  UploadConflictError: class UploadConflictError extends Error {
    detail: { message: string; existing_paper_id: number; filename: string };

    constructor(detail: { message: string; existing_paper_id: number; filename: string }) {
      super(detail.message);
      this.detail = detail;
      this.name = "UploadConflictError";
    }
  },
  uploadPaper: vi.fn(),
  fetchJob: vi.fn(),
  subscribeTaskStatusEvents: vi.fn(() => vi.fn()),
}));

describe("UploadPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("上传后会订阅任务状态并更新界面", async () => {
    const user = userEvent.setup();
    let handleTaskEvent: (event: { job: Record<string, unknown> | null }) => void = () => {};

    vi.mocked(uploadPaper).mockResolvedValue({
      paper_id: 1,
      job_id: 9,
      filename: "paper.pdf",
      status: "queued",
    });
    vi.mocked(fetchJob)
      .mockResolvedValueOnce({
        id: 9,
        job_type: "pdf_ingest",
        paper_id: 1,
        status: "processing",
        error_message: null,
        retry_count: 0,
        started_at: "2026-05-06T05:00:00Z",
        finished_at: null,
        created_at: "2026-05-06T05:00:00Z",
      });
    vi.mocked(subscribeTaskStatusEvents).mockImplementation(({ onEvent }) => {
      handleTaskEvent = onEvent as (event: { job: Record<string, unknown> | null }) => void;
      return vi.fn();
    });

    render(<UploadPanel />);

    const input = screen.getByLabelText("选择 PDF");
    await user.upload(input, new File(["pdf"], "paper.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "上传 PDF" }));

    await waitFor(() => expect(uploadPaper).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(fetchJob).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByText(/当前上传已创建任务 #9，状态为 处理中/)).toBeInTheDocument());
    expect(subscribeTaskStatusEvents).toHaveBeenCalledTimes(1);

    handleTaskEvent({
      job: {
        id: 9,
        job_type: "pdf_ingest",
        paper_id: 1,
        status: "completed",
        error_message: null,
        retry_count: 0,
        started_at: "2026-05-06T05:00:00Z",
        finished_at: "2026-05-06T05:01:00Z",
        created_at: "2026-05-06T05:00:00Z",
      },
    });

    await waitFor(() => expect(screen.getByText(/当前上传已创建任务 #9，状态为 已完成/)).toBeInTheDocument());
  });

  it("上传失败时显示错误", async () => {
    const user = userEvent.setup();
    vi.mocked(uploadPaper).mockRejectedValue(new Error("Only PDF files are supported"));

    render(<UploadPanel />);

    const input = screen.getByLabelText("选择 PDF");
    await user.upload(input, new File(["oops"], "bad.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "上传 PDF" }));

    expect(await screen.findByText("Only PDF files are supported")).toBeInTheDocument();
  });

  it("遇到同名文件时提示是否覆盖上传", async () => {
    const user = userEvent.setup();
    vi.mocked(uploadPaper).mockRejectedValue(
      new UploadConflictError({
        message: "已有相同文件名的文档，是否需要覆盖上传？",
        existing_paper_id: 2,
        filename: "paper.pdf",
      })
    );

    render(<UploadPanel />);

    const input = screen.getByLabelText("选择 PDF");
    await user.upload(input, new File(["pdf"], "paper.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "上传 PDF" }));

    expect(await screen.findByText("已有相同原始文件名的文档：paper.pdf。")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "确认覆盖上传" })).toBeInTheDocument();
  });
});
