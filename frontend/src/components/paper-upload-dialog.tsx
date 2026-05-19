"use client";

import { UploadPanel, type UploadPanelProps } from "@/components/upload-panel";
import { Dialog } from "@/components/ui/dialog";

type PaperUploadDialogProps = Pick<UploadPanelProps, "onUploadAccepted" | "onJobUpdate"> & {
  open: boolean;
  onClose: () => void;
};

export function PaperUploadDialog({ open, onClose, onUploadAccepted, onJobUpdate }: PaperUploadDialogProps) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="上传 PDF"
      description="上传后的文档解析会继续在后台运行，详细状态请到右上角任务列表查看。"
    >
      <UploadPanel onUploadAccepted={onUploadAccepted} onJobUpdate={onJobUpdate} onCloseAfterUpload={onClose} />
    </Dialog>
  );
}
