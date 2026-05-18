import { type ReactNode, useState } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { PaperWorkbench } from "@/components/paper-workbench";

vi.mock("@/components/paper-upload-dialog", () => ({
  PaperUploadDialog: () => null,
}));

vi.mock("@/components/paper-metadata-dialog", () => ({
  PaperMetadataDialog: () => null,
}));

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open?: boolean; children?: ReactNode }) => (open ? <>{children}</> : null),
}));

vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/session", () => ({
  useHasPermission: () => true,
}));

const fetchPapers = vi.fn();
const fetchPaper = vi.fn();
const deletePaper = vi.fn();
const regeneratePaperQuestionSet = vi.fn();
const regeneratePaperSummary = vi.fn();
const rerunPaperOcr = vi.fn();
const subscribeTaskStatusEvents = vi.fn(() => vi.fn());

vi.mock("@/lib/papers", () => ({
  deletePaper: (...args: unknown[]) => deletePaper(...args),
  fetchPaper: (...args: unknown[]) => fetchPaper(...args),
  fetchPapers: (...args: unknown[]) => fetchPapers(...args),
  regeneratePaperQuestionSet: (...args: unknown[]) => regeneratePaperQuestionSet(...args),
  regeneratePaperSummary: (...args: unknown[]) => regeneratePaperSummary(...args),
  rerunPaperOcr: (...args: unknown[]) => rerunPaperOcr(...args),
  subscribeTaskStatusEvents: (...args: unknown[]) => subscribeTaskStatusEvents(...args),
}));

function makePaperListItem({
  id,
  title,
  authors = null,
  publishedAt,
}: {
  id: number;
  title: string;
  authors?: string | null;
  publishedAt: string | null;
}) {
  return {
    id,
    title,
    authors,
    original_filename: `${title}.pdf`,
    abstract_raw: `${title} abstract`,
    published_at: publishedAt,
    status: "completed",
    ocr_status: "completed",
    summary_status: "completed",
    question_set_status: "completed",
    created_at: "2026-05-10T00:00:00Z",
    updated_at: "2026-05-11T00:00:00Z",
  };
}

function makePaperDetail(id: number, title: string) {
  return {
    id,
    title,
    authors: null,
    abstract_raw: `${title} abstract`,
    source_url: null,
    pdf_url: null,
    doi: null,
    published_at: null,
    status: "completed",
    ocr_status: "completed",
    summary_status: "completed",
    question_set_status: "completed",
    created_at: "2026-05-10T00:00:00Z",
    updated_at: "2026-05-11T00:00:00Z",
    original_filename: `${title}.pdf`,
    preview_text: `${title} preview`,
    extraction_metadata: null,
    structured_summary: null,
    question_set_extraction: null,
    latest_job: null,
    latest_ocr_job: null,
    latest_summary_job: null,
    latest_question_set_job: null,
  };
}

function expectPaperOrder(expectedTitles: string[]) {
  const paperButtons = screen
    .getAllByRole("button")
    .filter((button) => button.textContent?.includes("作者："));

  expect(paperButtons).toHaveLength(expectedTitles.length);
  expect(paperButtons.map((button) => button.textContent || "")).toEqual(
    expectedTitles.map((title) => expect.stringContaining(title))
  );
}

describe("PaperWorkbench", () => {
  it("支持按论文时间和字母切换排序方向", async () => {
    const user = userEvent.setup();
    const papers = [
      makePaperListItem({ id: 1, title: "Alpha", publishedAt: null }),
      makePaperListItem({ id: 2, title: "Zebra", publishedAt: "2023-01-01T00:00:00Z" }),
      makePaperListItem({ id: 3, title: "Beta", publishedAt: "2024-01-01T00:00:00Z" }),
    ];

    fetchPapers.mockResolvedValue(papers);
    fetchPaper.mockImplementation(async (paperId: number) => {
      const paper = papers.find((item) => item.id === paperId);
      return makePaperDetail(paperId, paper?.title ?? "Unknown");
    });

    function Harness() {
      const [selectedPaperId, setSelectedPaperId] = useState<number | null>(null);
      return <PaperWorkbench selectedPaperId={selectedPaperId} onSelectedPaperChange={setSelectedPaperId} />;
    }

    render(<Harness />);

    await waitFor(() => expect(fetchPapers).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByRole("button", { name: /Alpha/ })).toBeInTheDocument());
    await waitFor(() => expectPaperOrder(["Beta", "Zebra", "Alpha"]));

    await user.click(screen.getByRole("button", { name: "当前按论文时间排序，点击切换到字母" }));
    await waitFor(() => expectPaperOrder(["Zebra", "Beta", "Alpha"]));

    await user.click(screen.getByRole("button", { name: "当前倒序，点击切换到正序" }));
    await waitFor(() => expectPaperOrder(["Alpha", "Beta", "Zebra"]));
  });

  it("为自动提取和手动编辑的元数据显示来源", async () => {
    const papers = [makePaperListItem({ id: 1, title: "Alpha", publishedAt: "2024-05-01T00:00:00Z" })];

    fetchPapers.mockResolvedValue(papers);
    fetchPaper.mockResolvedValue({
      ...makePaperDetail(1, "Alpha"),
      authors: "Alice Example; Bob Example",
      doi: "10.3000/manual",
      source_url: "https://example.com/papers/alpha",
      published_at: "2024-05-01T00:00:00Z",
      structured_summary: {
        abstract_cn: "这是一段中文摘要。",
        key_points: ["要点一"],
        research_question: "研究问题",
        method: "研究方法",
        findings: "主要发现",
        limitations: "局限性",
        authors: "Alice Example; Bob Example",
        doi: "10.1000/extracted",
        source_url: "https://example.com/papers/alpha",
        published_at: "2024-05-01",
      },
    });

    function Harness() {
      const [selectedPaperId, setSelectedPaperId] = useState<number | null>(1);
      return <PaperWorkbench selectedPaperId={selectedPaperId} onSelectedPaperChange={setSelectedPaperId} />;
    }

    render(<Harness />);

    await waitFor(() => expect(fetchPapers).toHaveBeenCalled());
    await waitFor(() => expect(fetchPaper).toHaveBeenCalledWith(1));

    expect(screen.getAllByText("来源：摘要自动提取")).toHaveLength(3);
    expect(screen.getByText("来源：手动编辑")).toBeInTheDocument();
  });

  it("按新的顺序展示详情标签页并移除旧说明文案", async () => {
    const papers = [makePaperListItem({ id: 1, title: "Alpha", authors: "Alice Example", publishedAt: "2024-05-01T00:00:00Z" })];

    fetchPapers.mockResolvedValue(papers);
    fetchPaper.mockResolvedValue(makePaperDetail(1, "Alpha"));

    function Harness() {
      const [selectedPaperId, setSelectedPaperId] = useState<number | null>(1);
      return <PaperWorkbench selectedPaperId={selectedPaperId} onSelectedPaperChange={setSelectedPaperId} />;
    }

    render(<Harness />);

    await waitFor(() => expect(fetchPaper).toHaveBeenCalledWith(1));

    expect(screen.getByRole("button", { name: "文档信息" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "OCR文本" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "摘要" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "问题集结果" })).toBeInTheDocument();
    expect(screen.queryByText("OCR / 摘要预览")).not.toBeInTheDocument();
    expect(screen.queryByText("摘要和文档信息")).not.toBeInTheDocument();
    expect(screen.queryByText("查看当前选中论文的展示名、原始文件名、摘要信息和 OCR 文本内容。")).not.toBeInTheDocument();
    expect(screen.getAllByText("原始文件名：Alpha.pdf").length).toBeGreaterThan(0);
    expect(screen.getByText("作者：Alice Example")).toBeInTheDocument();
    expect(screen.queryByText(/Alpha abstract/)).not.toBeInTheDocument();
  });

  it("展示问题集阶段和结果标签页", async () => {
    const papers = [makePaperListItem({ id: 1, title: "Alpha", publishedAt: "2024-05-01T00:00:00Z" })];

    fetchPapers.mockResolvedValue(papers);
    fetchPaper.mockResolvedValue({
      ...makePaperDetail(1, "Alpha"),
      question_set_extraction: {
        generated_at: "2026-05-16T00:00:00Z",
        model_name: "test-model",
        questions: [{ id: "q1", question: "这篇论文研究了什么？", answer: "回答内容" }],
      },
      latest_question_set_job: {
        id: 3,
        job_type: "paper_question_set",
        paper_id: 1,
        status: "completed",
        error_message: null,
        retry_count: 0,
        started_at: null,
        finished_at: null,
        created_at: "2026-05-16T00:00:00Z",
      },
    });

    function Harness() {
      const [selectedPaperId, setSelectedPaperId] = useState<number | null>(1);
      return <PaperWorkbench selectedPaperId={selectedPaperId} onSelectedPaperChange={setSelectedPaperId} />;
    }

    render(<Harness />);

    await waitFor(() => expect(fetchPaper).toHaveBeenCalledWith(1));
    expect(screen.getAllByText(/问题集/).length).toBeGreaterThan(0);

    await userEvent.setup().click(screen.getByRole("button", { name: "问题集结果" }));
    expect(screen.getByText("这篇论文研究了什么？")).toBeInTheDocument();
    expect(screen.getByText("回答内容")).toBeInTheDocument();
  });
});
