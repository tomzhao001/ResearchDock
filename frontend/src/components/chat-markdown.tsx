"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";

type ChatMarkdownProps = {
  content: string;
  className?: string;
};

export function ChatMarkdown({ content, className }: ChatMarkdownProps) {
  return (
    <div className={cn("grid gap-3 text-sm leading-7 text-inherit", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1 className="text-lg font-semibold text-inherit">{children}</h1>,
          h2: ({ children }) => <h2 className="text-base font-semibold text-inherit">{children}</h2>,
          h3: ({ children }) => <h3 className="text-sm font-semibold text-inherit">{children}</h3>,
          p: ({ children }) => <p className="whitespace-pre-wrap text-inherit">{children}</p>,
          ul: ({ children }) => <ul className="list-disc space-y-1 pl-5 text-inherit">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5 text-inherit">{children}</ol>,
          li: ({ children }) => <li className="text-inherit">{children}</li>,
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-slate-300 pl-4 text-slate-600">{children}</blockquote>
          ),
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noreferrer"
              className="font-medium text-sky-700 underline underline-offset-4"
            >
              {children}
            </a>
          ),
          strong: ({ children }) => <strong className="font-semibold text-inherit">{children}</strong>,
          em: ({ children }) => <em className="italic text-inherit">{children}</em>,
          code: ({ className: codeClassName, children }) => {
            const isBlock = Boolean(codeClassName);
            if (isBlock) {
              return (
                <code className="block overflow-x-auto rounded-2xl bg-slate-950/95 px-4 py-3 font-mono text-[13px] leading-6 text-slate-50">
                  {children}
                </code>
              );
            }
            return (
              <code className="rounded bg-slate-200 px-1.5 py-0.5 font-mono text-[13px] text-slate-900">{children}</code>
            );
          },
          pre: ({ children }) => <pre className="overflow-x-auto">{children}</pre>,
          hr: () => <hr className="border-slate-200" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
