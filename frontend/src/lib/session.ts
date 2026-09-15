"use client";

import { createContext, createElement, useContext, type ReactNode } from "react";

export type SessionUser = {
  id: number;
  username: string;
  role: string;
  permissions: string[];
  organization: {
    id: number;
    name: string;
    slug: string;
  };
};

const SessionContext = createContext<SessionUser | null>(null);

export function SessionProvider({
  value,
  children,
}: {
  value: SessionUser;
  children: ReactNode;
}) {
  return createElement(SessionContext.Provider, { value }, children);
}

export function useHasPermission(permission: string): boolean {
  const session = useContext(SessionContext);
  return session?.permissions.includes(permission) ?? false;
}
