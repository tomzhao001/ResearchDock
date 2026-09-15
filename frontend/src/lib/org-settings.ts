import { apiJson } from "@/lib/api";

export type OrganizationQuestionItem = {
  id: string;
  question: string;
};

export type OrganizationQuestionSet = {
  organization_id: number;
  questions: OrganizationQuestionItem[];
  updated_at: string | null;
};

export function fetchOrganizationQuestionSet(): Promise<OrganizationQuestionSet> {
  return apiJson<OrganizationQuestionSet>("/api/org-settings/questions");
}

export function updateOrganizationQuestionSet(payload: {
  questions: OrganizationQuestionItem[];
}): Promise<OrganizationQuestionSet> {
  return apiJson<OrganizationQuestionSet>("/api/org-settings/questions", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
