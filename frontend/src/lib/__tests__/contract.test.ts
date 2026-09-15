import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

async function importLib(name: string) {
  return import(`../${name}.ts`);
}

const PAPERS_FUNCTIONS = [
  "uploadPaper",
  "fetchJob",
  "fetchJobs",
  "cancelJob",
  "deleteJob",
  "fetchPapers",
  "fetchPaper",
  "deletePaper",
  "reparsePaperDocument",
  "regeneratePaperSummary",
  "regeneratePaperQuestionSet",
  "updatePaperMetadata",
  "subscribeTaskStatusEvents",
] as const;

const CHAT_FUNCTIONS = [
  "fetchChatTopics",
  "createChatTopic",
  "fetchTopicMessages",
  "subscribeChatProgressEvents",
  "streamTopicMessage",
] as const;

const JOB_STATUS_FUNCTIONS = [
  "getJobStatusLabel",
  "getJobStatusClassName",
  "isActiveJobStatus",
  "isDeletableJobStatus",
  "isHiddenTaskListJobStatus",
  "isStoppableJobStatus",
] as const;

describe("@/lib/utils", () => {
  it("exports cn as a function", async () => {
    const mod = await importLib("utils");
    expect(typeof mod.cn).toBe("function");
  });
});

describe("@/lib/session", () => {
  it("exports SessionProvider and useHasPermission", async () => {
    const mod = await importLib("session");
    expect(typeof mod.SessionProvider).toBe("function");
    expect(typeof mod.useHasPermission).toBe("function");
  });
});

describe("@/lib/job-status", () => {
  it("exports six status helpers", async () => {
    const mod = await importLib("job-status");
    for (const name of JOB_STATUS_FUNCTIONS) {
      expect(typeof mod[name]).toBe("function");
    }
  });

  it("maps processing and cancel_requested labels", async () => {
    const { getJobStatusLabel } = await importLib("job-status");
    expect(getJobStatusLabel("processing")).toBe("处理中");
    expect(getJobStatusLabel("cancel_requested")).toBe("取消中");
    expect(getJobStatusLabel(null, { variant: "upload", emptyLabel: "未开始" })).toBe("未开始");
  });

  it("classifies active / hidden / deletable / stoppable statuses", async () => {
    const {
      getJobStatusClassName,
      isActiveJobStatus,
      isDeletableJobStatus,
      isHiddenTaskListJobStatus,
      isStoppableJobStatus,
    } = await importLib("job-status");

    expect(typeof getJobStatusClassName("processing")).toBe("string");
    expect(isActiveJobStatus("processing")).toBe(true);
    expect(isActiveJobStatus("cancel_requested")).toBe(true);
    expect(isHiddenTaskListJobStatus("completed")).toBe(true);
    expect(isHiddenTaskListJobStatus("cancelled")).toBe(true);
    expect(isDeletableJobStatus("failed")).toBe(true);
    expect(isStoppableJobStatus("processing")).toBe(true);
    expect(isStoppableJobStatus("cancel_requested")).toBe(false);
  });
});

describe("@/lib/api", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }))
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("exports apiFetch and always sends credentials include", async () => {
    const { apiFetch } = await importLib("api");
    expect(typeof apiFetch).toBe("function");

    await apiFetch("/api/auth/me");
    await apiFetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: "admin", password: "secret" }),
    });

    const fetchMock = vi.mocked(fetch);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0]?.[1]).toEqual(expect.objectContaining({ credentials: "include" }));
    expect(fetchMock.mock.calls[1]?.[1]).toEqual(expect.objectContaining({ credentials: "include" }));
  });
});

describe("@/lib/chat", () => {
  it("exports five chat API functions", async () => {
    const mod = await importLib("chat");
    for (const name of CHAT_FUNCTIONS) {
      expect(typeof mod[name]).toBe("function");
    }
  });
});

describe("@/lib/org-settings", () => {
  it("exports fetchOrganizationQuestionSet and updateOrganizationQuestionSet", async () => {
    const mod = await importLib("org-settings");
    expect(typeof mod.fetchOrganizationQuestionSet).toBe("function");
    expect(typeof mod.updateOrganizationQuestionSet).toBe("function");
  });
});

describe("@/lib/papers", () => {
  it("exports paper/job functions plus UploadConflictError", async () => {
    const mod = await importLib("papers");
    for (const name of PAPERS_FUNCTIONS) {
      expect(typeof mod[name]).toBe("function");
    }
    expect(typeof mod.UploadConflictError).toBe("function");
  });

  it("UploadConflictError carries detail and is instanceof Error", async () => {
    const { UploadConflictError } = await importLib("papers");
    const error = new UploadConflictError({
      message: "已有相同文件名的文档，是否需要覆盖上传？",
      existing_paper_id: 2,
      filename: "paper.pdf",
    });
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe("UploadConflictError");
    expect(error.message).toBe("已有相同文件名的文档，是否需要覆盖上传？");
    expect(error.detail).toEqual({
      message: "已有相同文件名的文档，是否需要覆盖上传？",
      existing_paper_id: 2,
      filename: "paper.pdf",
    });
  });
});
