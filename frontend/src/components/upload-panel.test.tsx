import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { UploadPanel } from "@/components/upload-panel";
import { fetchJob, uploadPaper } from "@/lib/papers";

vi.mock("@/lib/papers", () => ({
  uploadPaper: vi.fn(),
  fetchJob: vi.fn(),
}));

describe("UploadPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("上传后会轮询任务直至完成", async () => {
    const user = userEvent.setup();

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
      })
      .mockResolvedValueOnce({
        id: 9,
        job_type: "pdf_ingest",
        paper_id: 1,
        status: "completed",
        error_message: null,
        retry_count: 0,
        started_at: "2026-05-06T05:00:00Z",
        finished_at: "2026-05-06T05:01:00Z",
        created_at: "2026-05-06T05:00:00Z",
      });

    render(<UploadPanel pollDelayMs={1} />);

    const input = screen.getByLabelText("选择 PDF");
    await user.upload(input, new File(["pdf"], "paper.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "上传 PDF" }));

    await waitFor(() => expect(uploadPaper).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(fetchJob).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByText("已完成")).toBeInTheDocument());
    expect(screen.getByText("paper.pdf")).toBeInTheDocument();
  });

  it("上传失败时显示错误", async () => {
    const user = userEvent.setup();
    vi.mocked(uploadPaper).mockRejectedValue(new Error("Only PDF files are supported"));

    render(<UploadPanel pollDelayMs={1} />);

    const input = screen.getByLabelText("选择 PDF");
    await user.upload(input, new File(["oops"], "bad.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "上传 PDF" }));

    expect(await screen.findByText("Only PDF files are supported")).toBeInTheDocument();
  });
});
