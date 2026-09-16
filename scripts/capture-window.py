from __future__ import annotations

import argparse
import ctypes
import time
from ctypes import wintypes
from pathlib import Path

from PIL import ImageGrab


class Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-screen-fallback",
        action="store_true",
        help="fall back to the visible screen region when HWND capture is blank",
    )
    args = parser.parse_args()
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()
    user32 = ctypes.windll.user32
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.SwitchToThisWindow.argtypes = [wintypes.HWND, wintypes.BOOL]
    user32.SwitchToThisWindow.restype = None
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(Rect)]
    user32.GetWindowRect.restype = wintypes.BOOL
    handles: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_window(handle: int, _parameter: int) -> bool:
        length = user32.GetWindowTextLengthW(handle)
        if length:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(handle, buffer, length + 1)
            process_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(handle, ctypes.byref(process_id))
            matches_process = args.pid is None or process_id.value == args.pid
            if (
                args.title in buffer.value
                and matches_process
                and user32.IsWindowVisible(handle)
            ):
                handles.append(handle)
        return True

    user32.EnumWindows(enum_window, 0)
    handle = handles[0] if handles else 0
    if not handle:
        raise SystemExit(f"window not found: {args.title}")
    user32.ShowWindow(handle, 9)
    user32.SwitchToThisWindow(handle, True)
    user32.SetWindowPos(handle, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)
    user32.SetForegroundWindow(handle)
    time.sleep(0.35)
    rect = Rect()
    if not user32.GetWindowRect(handle, ctypes.byref(rect)):
        raise SystemExit("GetWindowRect failed")
    # Pillow's HWND capture uses PrintWindow on current Windows builds, so the
    # requested application is captured even when foreground activation is
    # denied by the desktop focus policy.  The screen-region fallback keeps the
    # helper compatible with older Pillow versions.
    try:
        image = ImageGrab.grab(window=handle, include_layered_windows=True)
        minimum_luma, _ = image.convert("L").getextrema()
        if minimum_luma >= 250 and args.allow_screen_fallback:
            image = ImageGrab.grab(
                (rect.left, rect.top, rect.right, rect.bottom), all_screens=True
            )
    except TypeError:
        image = ImageGrab.grab((rect.left, rect.top, rect.right, rect.bottom), all_screens=True)
    user32.SetWindowPos(handle, -2, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
