import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TaskListPopover } from "@/components/task-list-popover";
import { fetchJobs, subscribeTaskStatusEvents } from "@/lib/papers";
import { SessionProvider, type SessionUser } from "@/lib/session";

vi.mock("@/lib/papers", () => ({
  deleteJob: vi.fn(),
  fetchJobs: vi.fn(),
  subscribeTaskStatusEvents: vi.fn(() => vi.fn()),
}));

const baseSession: SessionUser = {
  id: 1,
  username: "admin",
  role: "org_owner",
  permissions: ["jobs:read", "jobs:manage", "papers:read"],
  organization: {
    id: 1,
    name: "Test Org",
    slug: "test-org",
  },
};

function renderPopover(session: SessionUser) {
  return render(
    <SessionProvider value={session}>
      <TaskListPopover onOpenPaper={() => {}} />
    </SessionProvider>
  );
}

describe("TaskListPopover permissions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchJobs).mockResolvedValue([
      {
        id: 1,
        job_type: "pdf_ingest",
        paper_id: 1,
        status: "failed",
        error_message: null,
        retry_count: 0,
        started_at: null,
        finished_at: null,
        created_at: "2026-05-14T00:00:00Z",
      },
    ]);
    vi.mocked(subscribeTaskStatusEvents).mockReturnValue(vi.fn());
  });

  it("没有 jobs:manage 权限时不显示删除按钮", async () => {
    const user = userEvent.setup();
    renderPopover({ ...baseSession, permissions: ["jobs:read"] });

    await user.click(screen.getByRole("button", { name: /任务/ }));

    await waitFor(() => expect(fetchJobs).toHaveBeenCalled());
    expect(screen.queryByLabelText("删除任务 1")).not.toBeInTheDocument();
  });

  it("有 jobs:manage 权限时显示删除按钮", async () => {
    const user = userEvent.setup();
    renderPopover(baseSession);

    await user.click(screen.getByRole("button", { name: /任务/ }));

    await waitFor(() => expect(fetchJobs).toHaveBeenCalled());
    expect(screen.getByLabelText("删除任务 1")).toBeInTheDocument();
  });
});
