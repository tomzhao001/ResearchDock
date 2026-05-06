"use client";

import { type ReactNode } from "react";

import { cn } from "@/lib/utils";

type TabsProps<T extends string> = {
  value: T;
  onValueChange: (value: T) => void;
  items: Array<{ value: T; label: string }>;
  className?: string;
};

export function Tabs<T extends string>({ value, onValueChange, items, className }: TabsProps<T>) {
  return (
    <div className={cn("inline-flex w-fit flex-wrap items-center gap-2 rounded-full border border-slate-200 bg-slate-100/80 p-1", className)}>
      {items.map((item) => (
        <button
          key={item.value}
          type="button"
          onClick={() => onValueChange(item.value)}
          className={cn(
            "rounded-full px-4 py-2 text-sm font-medium transition",
            value === item.value ? "bg-white text-slate-950 shadow-sm" : "text-slate-500 hover:text-slate-900"
          )}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

type TabPanelProps = {
  active: boolean;
  children: ReactNode;
  className?: string;
};

export function TabPanel({ active, children, className }: TabPanelProps) {
  if (!active) {
    return null;
  }
  return <div className={className}>{children}</div>;
}
