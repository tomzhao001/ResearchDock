import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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
    await user.click(screen.getByRole("button", { name: "批量上传 PDF" }));

    await waitFor(() => expect(uploadPaper).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(fetchJob).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByText(/当前批次已提交 1 \/ 1 个文件/)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText(/最近创建的任务 #9 状态为 处理中/)).toBeInTheDocument());
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

    await waitFor(() => expect(screen.getByText(/最近创建的任务 #9 状态为 已完成/)).toBeInTheDocument());
  });

  it("批量上传时会按选择顺序依次调用接口", async () => {
    const user = userEvent.setup();
    vi.mocked(uploadPaper)
      .mockResolvedValueOnce({
        paper_id: 1,
        job_id: 9,
        filename: "a.pdf",
        status: "queued",
      })
      .mockResolvedValueOnce({
        paper_id: 2,
        job_id: 10,
        filename: "b.pdf",
        status: "queued",
      });
    vi.mocked(fetchJob)
      .mockResolvedValueOnce({
        id: 9,
        job_type: "pdf_ingest",
        paper_id: 1,
        status: "queued",
        error_message: null,
        retry_count: 0,
        started_at: null,
        finished_at: null,
        created_at: "2026-05-06T05:00:00Z",
      })
      .mockResolvedValueOnce({
        id: 10,
        job_type: "pdf_ingest",
        paper_id: 2,
        status: "queued",
        error_message: null,
        retry_count: 0,
        started_at: null,
        finished_at: null,
        created_at: "2026-05-06T05:01:00Z",
      });

    render(<UploadPanel />);

    const input = screen.getByLabelText("选择 PDF");
    await user.upload(input, [
      new File(["first"], "a.pdf", { type: "application/pdf" }),
      new File(["second"], "b.pdf", { type: "application/pdf" }),
    ]);
    await user.click(screen.getByRole("button", { name: "批量上传 PDF" }));

    await waitFor(() => expect(uploadPaper).toHaveBeenCalledTimes(2));
    expect(vi.mocked(uploadPaper).mock.calls[0]?.[0]).toMatchObject({ name: "a.pdf" });
    expect(vi.mocked(uploadPaper).mock.calls[1]?.[0]).toMatchObject({ name: "b.pdf" });
    expect(screen.getByText("当前文件顺序：a.pdf -> b.pdf")).toBeInTheDocument();
  });

  it("选择了非 PDF 文件时会提示错误", async () => {
    render(<UploadPanel />);

    const input = screen.getByLabelText("选择 PDF");
    fireEvent.change(input, {
      target: {
        files: [new File(["oops"], "bad.txt", { type: "text/plain" })],
      },
    });

    expect(screen.getByText("仅支持上传 PDF 文件，请重新选择。")).toBeInTheDocument();
    expect(uploadPaper).not.toHaveBeenCalled();
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
    await user.click(screen.getByRole("button", { name: "批量上传 PDF" }));

    expect(await screen.findByText("已有相同原始文件名的文档：paper.pdf。")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "确认覆盖上传" })).toBeInTheDocument();
  });
});
