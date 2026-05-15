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
const regeneratePaperSummary = vi.fn();
const rerunPaperOcr = vi.fn();
const subscribeTaskStatusEvents = vi.fn(() => vi.fn());

vi.mock("@/lib/papers", () => ({
  deletePaper: (...args: unknown[]) => deletePaper(...args),
  fetchPaper: (...args: unknown[]) => fetchPaper(...args),
  fetchPapers: (...args: unknown[]) => fetchPapers(...args),
  regeneratePaperSummary: (...args: unknown[]) => regeneratePaperSummary(...args),
  rerunPaperOcr: (...args: unknown[]) => rerunPaperOcr(...args),
  subscribeTaskStatusEvents: (...args: unknown[]) => subscribeTaskStatusEvents(...args),
}));

function makePaperListItem({
  id,
  title,
  publishedAt,
}: {
  id: number;
  title: string;
  publishedAt: string | null;
}) {
  return {
    id,
    title,
    original_filename: `${title}.pdf`,
    abstract_raw: `${title} abstract`,
    published_at: publishedAt,
    status: "completed",
    ocr_status: "completed",
    summary_status: "completed",
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
    created_at: "2026-05-10T00:00:00Z",
    updated_at: "2026-05-11T00:00:00Z",
    original_filename: `${title}.pdf`,
    preview_text: `${title} preview`,
    extraction_metadata: null,
    structured_summary: null,
    latest_job: null,
    latest_ocr_job: null,
    latest_summary_job: null,
  };
}

function expectPaperOrder(expectedTitles: string[]) {
  const paperButtons = screen
    .getAllByRole("button")
    .filter((button) => button.textContent?.includes("原始文件名："));

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

    await user.click(screen.getByRole("button", { name: "论文时间" }));
    await waitFor(() => expectPaperOrder(["Zebra", "Beta", "Alpha"]));

    await user.click(screen.getByRole("button", { name: "倒序" }));
    await waitFor(() => expectPaperOrder(["Alpha", "Beta", "Zebra"]));
  });
});
