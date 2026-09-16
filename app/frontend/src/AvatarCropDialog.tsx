import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type {
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
} from "react";

import {
  AVATAR_MAX_ZOOM,
  AVATAR_MIN_ZOOM,
  avatarCropGeometry,
  clampAvatarCropOffset,
  clampAvatarCropZoom,
  renderAvatarCrop,
} from "./avatarCrop";
import type { AvatarCropPoint } from "./avatarCrop";
import { localeSafeError, useI18n } from "./i18n";

interface AvatarCropDialogProps {
  sourceUrl: string;
  pending: boolean;
  onCancel: () => void;
  onConfirm: (avatarDataUrl: string) => Promise<void>;
  onSourceError: (message: string) => void;
}
export function AvatarCropDialog({
  sourceUrl,
  pending,
  onCancel,
  onConfirm,
  onSourceError,
}: AvatarCropDialogProps) {
  const { locale, tr } = useI18n();
  const titleId = useId();
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    offset: AvatarCropPoint;
  } | null>(null);
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);
  const [viewportSize, setViewportSize] = useState(0);
  const [zoom, setZoom] = useState(AVATAR_MIN_ZOOM);
  const [offset, setOffset] = useState<AvatarCropPoint>({ x: 0, y: 0 });
  const [processingError, setProcessingError] = useState<string | null>(null);

  const geometry = useMemo(
    () => dimensions && viewportSize > 0
      ? avatarCropGeometry(dimensions, viewportSize, zoom)
      : null,
    [dimensions, viewportSize, zoom],
  );
  const safeOffset = useMemo(
    () => geometry ? clampAvatarCropOffset(offset, geometry) : { x: 0, y: 0 },
    [geometry, offset],
  );

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return undefined;
    const measure = () => setViewportSize(Math.max(1, Math.min(viewport.clientWidth, viewport.clientHeight)));
    measure();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", measure);
      return () => window.removeEventListener("resize", measure);
    }
    const observer = new ResizeObserver(measure);
    observer.observe(viewport);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (geometry) setOffset((current) => clampAvatarCropOffset(current, geometry));
  }, [geometry]);

  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const focusFrame = window.requestAnimationFrame(() => closeRef.current?.focus());
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (!pending) onCancel();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = Array.from(dialogRef.current?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ) ?? []);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown, true);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      document.removeEventListener("keydown", handleKeyDown, true);
      previousFocus?.focus();
    };
  }, [onCancel, pending]);

  const moveBy = useCallback((x: number, y: number) => {
    if (!geometry) return;
    setOffset((current) => clampAvatarCropOffset({ x: current.x + x, y: current.y + y }, geometry));
  }, [geometry]);

  const handleCropKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 24 : 8;
    if (event.key === "ArrowLeft") moveBy(-step, 0);
    else if (event.key === "ArrowRight") moveBy(step, 0);
    else if (event.key === "ArrowUp") moveBy(0, -step);
    else if (event.key === "ArrowDown") moveBy(0, step);
    else if (event.key === "+" || event.key === "=") setZoom(clampAvatarCropZoom(zoom + 0.1));
    else if (event.key === "-" || event.key === "_") setZoom(clampAvatarCropZoom(zoom - 0.1));
    else return;
    event.preventDefault();
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!geometry) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture?.(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      offset: safeOffset,
    };
  };
  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !geometry) return;
    setOffset(clampAvatarCropOffset({
      x: drag.offset.x + event.clientX - drag.startX,
      y: drag.offset.y + event.clientY - drag.startY,
    }, geometry));
  };
  const endPointerDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
  };

  const confirm = async () => {
    const image = imageRef.current;
    if (!image || !dimensions || viewportSize < 1) return;
    setProcessingError(null);
    try {
      await onConfirm(renderAvatarCrop(image, viewportSize, zoom, safeOffset));
    } catch (reason) {
      const code = reason instanceof Error ? reason.message : "";
      setProcessingError(code === "AVATAR_IMAGE_EMPTY"
        ? tr("头像图片为空", "The profile image is empty")
        : code === "AVATAR_IMAGE_PROCESSING_FAILED"
          ? tr("无法处理头像图片", "Could not process the profile image")
          : localeSafeError(code, locale, {
            zh: "无法保存头像",
            en: "Could not save the profile photo",
          }));
    }
  };

  const imageStyle = geometry ? {
    width: `${dimensions!.width * geometry.scale}px`,
    height: `${dimensions!.height * geometry.scale}px`,
    transform: `translate(-50%, -50%) translate(${safeOffset.x}px, ${safeOffset.y}px)`,
  } : undefined;
  const previewRatio = viewportSize > 0 ? 72 / viewportSize : 1;
  const previewStyle = geometry ? {
    width: `${dimensions!.width * geometry.scale * previewRatio}px`,
    height: `${dimensions!.height * geometry.scale * previewRatio}px`,
    transform: `translate(-50%, -50%) translate(${safeOffset.x * previewRatio}px, ${safeOffset.y * previewRatio}px)`,
  } : undefined;

  return <div className="avatar-crop-backdrop" role="presentation" onPointerDown={(event) => {
    if (event.target === event.currentTarget && !pending) onCancel();
  }}>
    <section ref={dialogRef} className="avatar-crop-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId}>
      <header><h3 id={titleId}>{tr("裁剪头像", "Crop profile photo")}</h3><button ref={closeRef} type="button" className="avatar-crop-close" aria-label={tr("关闭", "Close")} disabled={pending} onClick={onCancel}><span aria-hidden="true">×</span></button></header>
      <div className="avatar-crop-workspace">
        <div ref={viewportRef} className="avatar-crop-viewport" role="group" tabIndex={0} aria-label={tr("拖动裁剪区域", "Drag to position the crop")} onKeyDown={handleCropKeyDown} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={endPointerDrag} onPointerCancel={endPointerDrag}>
          <img ref={imageRef} src={sourceUrl} alt="" draggable={false} style={imageStyle} onLoad={(event) => {
            const image = event.currentTarget;
            if (image.naturalWidth < 1 || image.naturalHeight < 1) return onSourceError(tr("头像图片为空", "The profile image is empty"));
            setDimensions({ width: image.naturalWidth, height: image.naturalHeight });
            setZoom(AVATAR_MIN_ZOOM);
            setOffset({ x: 0, y: 0 });
          }} onError={() => onSourceError(tr("无法读取头像图片", "Could not read the profile image"))} />
          <span className="avatar-crop-mask" aria-hidden="true" />
        </div>
        <div className="avatar-crop-controls">
          <label><span>{tr("照片大小", "Photo size")}</span><input type="range" min={AVATAR_MIN_ZOOM} max={AVATAR_MAX_ZOOM} step="0.01" value={zoom} disabled={!dimensions || pending} onChange={(event) => setZoom(clampAvatarCropZoom(Number(event.target.value)))} /></label>
          <div className="avatar-crop-preview-wrap"><span>{tr("预览", "Preview")}</span><div className="avatar-crop-preview" aria-hidden="true"><img src={sourceUrl} alt="" draggable={false} style={previewStyle} /></div></div>
        </div>
      </div>
      {processingError && <p className="avatar-crop-error" role="alert">{processingError}</p>}
      <footer><button type="button" className="secondary-button" disabled={pending} onClick={onCancel}>{tr("取消", "Cancel")}</button><button type="button" className="primary-button" disabled={!dimensions || pending} onClick={() => void confirm()}>{pending ? tr("正在保存", "Saving") : tr("保存", "Save")}</button></footer>
    </section>
  </div>;
}
