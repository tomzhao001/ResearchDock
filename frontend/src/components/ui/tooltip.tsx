"use client";

import { useState, type ReactNode } from "react";

type TooltipProps = {
  content: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom";
};

export function Tooltip({ content, children, side = "top" }: TooltipProps) {
  const [open, setOpen] = useState(false);

  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children}
      {open ? (
        <span
          role="tooltip"
          className={`pointer-events-none absolute left-1/2 z-50 -translate-x-1/2 rounded-md bg-slate-950 px-2 py-1 text-[11px] whitespace-nowrap text-white shadow-lg ${
            side === "bottom" ? "top-[calc(100%+8px)]" : "bottom-[calc(100%+8px)]"
          }`}
        >
          {content}
        </span>
      ) : null}
    </span>
  );
}
