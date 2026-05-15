import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { OrgQuestionSetPanel } from "@/components/org-question-set-panel";

vi.mock("@/lib/session", () => ({
  useHasPermission: (permission: string) => permission === "org_settings:write",
}));

const fetchOrganizationQuestionSet = vi.fn();
const updateOrganizationQuestionSet = vi.fn();

vi.mock("@/lib/org-settings", () => ({
  fetchOrganizationQuestionSet: () => fetchOrganizationQuestionSet(),
  updateOrganizationQuestionSet: (payload: unknown) => updateOrganizationQuestionSet(payload),
}));

describe("OrgQuestionSetPanel", () => {
  it("支持新增问题并显示重新提取提示", async () => {
    const user = userEvent.setup();
    fetchOrganizationQuestionSet.mockResolvedValue({
      organization_id: 1,
      questions: [],
      updated_at: "2026-05-16T00:00:00Z",
    });
    updateOrganizationQuestionSet.mockResolvedValue({
      organization_id: 1,
      questions: [{ id: "q1", question: "新增的问题" }],
      updated_at: "2026-05-16T00:10:00Z",
    });

    render(<OrgQuestionSetPanel />);

    await waitFor(() => expect(fetchOrganizationQuestionSet).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: /新增问题/ }));

    expect(screen.getByText(/不会自动同步历史论文的抽取结果/)).toBeInTheDocument();
    await user.type(screen.getByPlaceholderText("例如：这篇论文解决了什么研究问题？"), "新增的问题");
    await user.click(screen.getByRole("button", { name: "确认新增" }));

    await waitFor(() =>
      expect(updateOrganizationQuestionSet).toHaveBeenCalledWith({
        questions: [{ id: expect.any(String), question: "新增的问题" }],
      })
    );
    expect(screen.getByText("新增的问题")).toBeInTheDocument();
  });

  it("支持删除问题并显示重跑提示", async () => {
    const user = userEvent.setup();
    fetchOrganizationQuestionSet.mockResolvedValue({
      organization_id: 1,
      questions: [{ id: "q1", question: "要删除的问题" }],
      updated_at: "2026-05-16T00:00:00Z",
    });
    updateOrganizationQuestionSet.mockResolvedValue({
      organization_id: 1,
      questions: [],
      updated_at: "2026-05-16T00:12:00Z",
    });

    render(<OrgQuestionSetPanel />);

    await waitFor(() => expect(fetchOrganizationQuestionSet).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: "删除问题 1" }));

    expect(screen.getByText(/请重新执行该论文的问题集抽取/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "确认删除" }));

    await waitFor(() => expect(updateOrganizationQuestionSet).toHaveBeenCalledWith({ questions: [] }));
  });
});
