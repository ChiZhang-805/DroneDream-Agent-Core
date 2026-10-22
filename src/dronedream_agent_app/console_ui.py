"""Human terminal presentation for the DroneDream product CLI.

The JSON transport remains the machine contract.  This module only renders a
bounded, already-redacted result when stdout is an interactive terminal.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TextIO

PINK = (255, 82, 171)
PURPLE = (154, 92, 255)
BLUE = (74, 190, 255)
BRAND_GRADIENT = (
    PINK,
    (230, 76, 220),
    PURPLE,
    (105, 122, 255),
    BLUE,
)
CYAN = (66, 220, 235)
GREEN = (84, 220, 132)
YELLOW = (255, 201, 74)
RED = (255, 92, 112)
WHITE = (240, 244, 255)
MUTED = (145, 156, 181)
SURFACE = (34, 39, 58)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# These Braille masks are sampled from the official transparent brand PNG.
# Keeping the sampled cells in source makes the installed wheel self-contained.
# The body mask is intentionally solid; the companion mask restores the logo's
# white wing, ear, and eye details over the pink-purple-blue fill.
_BRAND_MARK_DETAILED = (
    "    ⣠⣤⣶⣿⡿⠟⠋⠁",
    " ⢀⣴⣿⣿⣿⣿⠟     ⣀⣠⣤⡤",
    "⢠⣾⣿⣿⣿⣿⣿   ⣠⣴⣿⣿⣿⡟",
    "⣾⣿⣿⣿⣿⣿⣿⡀ ⣼⣿⣿⣿⣿⣿⠁",
    "⣿⣿⣿⣿⣿⣿⣿⣷⣄ ⢸⣿⣿⣿⣿⣤⣶",
    "⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦",
    "⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠉⠁",
    "  ⠙⢿⣿⣿⣿⣿⣿⣿⡿⠟⠛⠋⠁",
    "    ⠉⠛⠿⢿⣿⣯⡀",
)

_BRAND_MARK_DETAILED_WHITE = (
    "",
    "",
    "",
    " ⠐⠲⠦⣤⣀",
    "    ⠈⠙⢷⣆     ⢰⠦⣤⠂",
    "      ⠈⢿⣧    ⠈ ⠰⠆",
    "       ⣸⡟",
    "       ⣿",
    "       ⠘⠧⣄",
)

_BRAND_MARK_MEDIUM = (
    "    ⣠⣤⣶⣾⣿⠟⠋⠉",
    " ⢀⣴⣿⣿⣿⣿⡟⠁    ⣀⣤⣤⣶⠖",
    "⢠⣿⣿⣿⣿⣿⣿   ⣠⣴⣿⣿⣿⣿⠃",
    "⣿⣿⣿⣿⣿⣿⣿⣧⡀⠘⠛⣿⣿⣿⣿⣿ ⡀",
    "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣿⣿⣿⣿⣿⣿⣿⡀",
    "⠘⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠋",
    " ⠈⠻⢿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠟⠛⠉",
    "    ⠉⠛⠻⠿⣿⣿⣅",
)

_BRAND_MARK_MEDIUM_WHITE = (
    "",
    "",
    "  ⡀",
    "  ⠉⠙⠻⢶⣤⡀      ⢀⡀ ⡀",
    "      ⠙⢿⣦     ⠿⠙⢋⡀",
    "       ⢘⣿⡇",
    "       ⢸⡇",
    "       ⠈⠳⣤⡀",
)

_BRAND_WORDMARK_DETAILED = (
    "⣿⣿⣿⣿⣿⣷⣄                              ⢸⣿⣿⣿⣿⣿⣶⡄",
    "⠛⣿⣿⠛⠛⢿⣿⡆                             ⠈⠛⣿⣿⠛⠻⣿⣿⡀",
    " ⣿⣿  ⠘⣿⣇ ⣤⣤⣠⣤  ⢀⣴⣶⣤⡀  ⢠⣤⢀⣤⣤⡄   ⢀⣤⣶⣤⡀   ⣿⣿  ⢸⣿⡇ ⣤⡄⣠⣤  ⣠⣴⣶⣄   ⢀⣴⣶⣤⡀  ⢠⣤⢀⣤⣤⡀⢀⣤⣄",
    " ⣿⣿   ⣿⣿ ⣿⣿⣿⣿ ⢰⣿⣿⠿⣿⣿⡄ ⢸⣿⣾⠿⣿⣿⡆ ⢠⣿⣿⠿⣿⣿⡀  ⣿⣿  ⢸⣿⡇ ⣿⣿⣿⣿ ⣼⣿⡿⠿⣿⣧ ⢠⣿⡿⠿⢿⣿⡀ ⢸⣿⣿⢿⣿⣷⣿⢿⣿⣧",
    " ⣿⣿   ⣿⣿ ⣿⣿⠁  ⣿⣿⠁ ⠈⣿⣷ ⢸⣿⠇ ⠈⣿⣷ ⣼⣿⠁ ⠈⣿⣇  ⣿⣿  ⢸⣿⡇ ⣿⡟  ⢀⣿⡏  ⢹⣿⡄⠈⠛⠁ ⢸⣿⡇ ⢸⣿⠁ ⢹⣿⠃ ⢹⣿",
    " ⣿⣿   ⣿⣿ ⣿⣿   ⣿⡇   ⢹⣿ ⢸⣿   ⣿⣿ ⣿⣿⣿⣿⣿⣿⣿  ⣿⣿  ⢸⣿⡇ ⣿⡇  ⢸⣿⣿⣿⣿⣿⣿⡇⢀⣶⣿⣿⣿⣿⡇ ⢸⣿  ⢸⣿  ⢸⣿",
    " ⣿⣿  ⢰⣿⡏ ⣿⣿   ⣿⣧   ⣿⣿ ⢸⣿   ⣿⣿ ⣿⣿⠉⠉⠉⣉⠁  ⣿⣿  ⣸⣿⡇ ⣿⡇  ⠸⣿⡏⠉⠉⢉⡉ ⣼⣿⠋⠉⢹⣿⡇ ⢸⣿  ⢸⣿  ⢸⣿",
    "⣤⣿⣿⣶⣶⣿⣿⠃ ⣿⣿   ⢻⣿⣦⣤⣴⣿⡏ ⢸⣿   ⣿⣿ ⠸⣿⣧⣤⣴⣿⠇⢠⣤⣿⣿⣶⣶⣿⡿  ⣿⡇   ⢿⣿⣤⣠⣾⣿ ⢻⣿⣄⣠⣾⣿⣧⡄⢸⣿  ⢸⣿  ⢸⣿",
    "⣿⣿⣿⣿⡿⠟⠁  ⢿⡿    ⠻⣿⣿⣿⠟  ⢸⣿   ⢿⡿  ⠹⣿⣿⣿⠏ ⠸⣿⣿⣿⣿⡿⠟⠁  ⣿⡇   ⠈⠻⣿⣿⡿⠃ ⠈⢿⣿⣿⠏⠻⠿⠇⢸⣿  ⢸⣿  ⢸⣿",
)

_BRAND_WORDMARK_MEDIUM = (
    "⣿⣿⣿⣿⣿⣷⣄                              ⢸⣿⣿⣿⣿⣿⣶⡄",
    "⠉⣿⣿⠉⠙⢿⣿⡆                             ⠈⠉⣿⣿⠉⠙⢿⣿⡄",
    " ⣿⣿  ⠈⣿⣧ ⣶⣶⣴⣶  ⣴⣾⣿⣷⣄  ⢰⣶⣠⣶⣶⣦   ⣠⣾⣿⣷⣄   ⣿⣿  ⢸⣿⡇ ⣶⣆⣴⣶ ⢀⣴⣾⣿⣦⡀  ⣠⣾⣿⣷⣄  ⢰⣶⣠⣶⣶⡄⣰⣶⣶⡄",
    " ⣿⣿   ⣿⣿ ⣿⣿⠟⠛ ⣼⣿⠟⠉⠻⣿⣇ ⢸⣿⡟⠛⠻⣿⣇ ⢸⣿⠟⠉⠻⣿⡆  ⣿⣿  ⢸⣿⡇ ⣿⣿⠟⠛ ⣾⡿⠋⠙⢿⣷ ⠰⣿⠏⠉⢻⣿⡄ ⢸⣿⠟⠛⢿⣿⡟⠛⣿⣿",
    " ⣿⣿   ⣿⣿ ⣿⣿   ⣿⡏   ⢿⣿ ⢸⣿⠁  ⣿⣿ ⣿⣿⣤⣤⣤⣿⣷  ⣿⣿  ⢸⣿⡇ ⣿⡇  ⢰⣿⣧⣤⣤⣼⣿⡇ ⣠⣤⣤⣼⣿⡇ ⢸⣿  ⢸⣿  ⢸⣿",
    " ⣿⣿  ⢀⣿⡟ ⣿⣿   ⣿⣇   ⣼⣿ ⢸⣿   ⣿⣿ ⣿⣿⠛⠛⠛⠛⠛  ⣿⣿  ⢸⣿⡇ ⣿⡇  ⢸⣿⡟⠛⠛⠛⠛⠃⣸⣿⠟⠛⢻⣿⡇ ⢸⣿  ⢸⣿  ⢸⣿",
    "⣤⣿⣿⣤⣤⣾⣿⠇ ⣿⣿   ⢻⣿⣄⣀⣰⣿⡟ ⢸⣿   ⣿⣿ ⢹⣿⣦⣀⣠⣿⡆⢀⣤⣿⣿⣤⣴⣿⣿⠁ ⣿⡇   ⣿⣷⣄⣀⣼⣷ ⢿⣿⣀⣀⣼⣿⣇⡀⢸⣿  ⢸⣿  ⢸⣿",
    "⣿⣿⣿⣿⣿⠿⠋  ⣿⡿   ⠈⠻⣿⣿⣿⠟  ⢸⣿   ⢿⡿  ⠻⣿⣿⣿⠟ ⠸⣿⣿⣿⣿⡿⠟⠁  ⣿⡇   ⠈⠿⣿⣿⡿⠃ ⠘⢿⣿⣿⠏⠻⣿⠇⢸⣿  ⢸⣿  ⢸⣿",
)

_BRAND_MARK_COMPACT = (
    "      ╭──────╮",
    "   ╭──╯     ╱",
    " ╭─╯   ╭───╯       ╭╮ ╭╮",
    "╭╯  ╭──╯      ╭────╯╰─╯╰╮",
    "│  ╭╯         ╰╮  ◇    ╰─╮",
    "╰╮ ╰╮          ╰╮      ╭╯",
    " ╰╮ ╰╮    ╭────╯ ╭───╯",
    "  ╰──╰────╯      ╰╯",
)


def _visible_len(value: str) -> int:
    """Return the printable width for the mostly single-width console text."""
    return len(_ANSI.sub("", value))


@dataclass(slots=True)
class TerminalUI:
    stream: TextIO = sys.stdout
    color: bool | None = None
    width: int | None = None

    def __post_init__(self) -> None:
        if (self.stream is sys.stdout or self.stream is sys.stderr) and hasattr(
            self.stream, "reconfigure"
        ):
            self.stream.reconfigure(encoding="utf-8")
        if self.color is None:
            self.color = bool(
                hasattr(self.stream, "isatty")
                and self.stream.isatty()
                and os.environ.get("NO_COLOR") is None
            ) or os.environ.get("FORCE_COLOR") not in {None, "", "0"}
        if self.width is None:
            self.width = max(72, min(shutil.get_terminal_size((108, 24)).columns, 132))

    def rgb(self, text: object, color: tuple[int, int, int], *, bold: bool = False) -> str:
        value = str(text)
        if not self.color:
            return value
        weight = "1;" if bold else ""
        return f"\x1b[{weight}38;2;{color[0]};{color[1]};{color[2]}m{value}\x1b[0m"

    def style(
        self,
        text: object,
        *,
        foreground: tuple[int, int, int] = WHITE,
        background: tuple[int, int, int] | None = None,
        bold: bool = False,
    ) -> str:
        value = str(text)
        if not self.color:
            return value
        codes = ["1" if bold else "22", f"38;2;{foreground[0]};{foreground[1]};{foreground[2]}"]
        if background is not None:
            codes.append(f"48;2;{background[0]};{background[1]};{background[2]}")
        return f"\x1b[{';'.join(codes)}m{value}\x1b[0m"

    def gradient(
        self,
        text: str,
        colors: tuple[tuple[int, int, int], ...] = BRAND_GRADIENT,
        *,
        bold: bool = True,
        overlays: dict[int, tuple[str, tuple[int, int, int]]] | None = None,
        offset: int = 0,
        span: int | None = None,
    ) -> str:
        if overlays:
            characters = list(text)
            for index, (char, _) in overlays.items():
                if 0 <= index < len(characters):
                    characters[index] = char
            visible_text = "".join(characters)
        else:
            visible_text = text
        if not self.color or len(text) < 2:
            return visible_text
        segments = max(1, len(colors) - 1)
        canvas_width = max(2, span or len(text))
        painted: list[str] = []
        for index, char in enumerate(text):
            if overlays and index in overlays:
                overlay_char, overlay_color = overlays[index]
                painted.append(self.rgb(overlay_char, overlay_color, bold=bold))
                continue
            canvas_index = min(max(offset + index, 0), canvas_width - 1)
            position = canvas_index / (canvas_width - 1) * segments
            segment = min(int(position), segments - 1)
            amount = position - segment
            amount = amount * amount * (3 - 2 * amount)
            left, right = colors[segment], colors[segment + 1]
            mixed = tuple(
                round(left[channel] + (right[channel] - left[channel]) * amount)
                for channel in range(3)
            )
            painted.append(self.rgb(char, mixed, bold=bold))
        return "".join(painted)

    def banner(self) -> None:
        print(file=self.stream)
        if self.width >= 98:
            self._brand_lockup(
                _BRAND_MARK_DETAILED,
                _BRAND_MARK_DETAILED_WHITE,
                _BRAND_WORDMARK_DETAILED,
            )
        elif self.width >= 96:
            self._brand_lockup(
                _BRAND_MARK_MEDIUM,
                _BRAND_MARK_MEDIUM_WHITE,
                _BRAND_WORDMARK_MEDIUM,
            )
        else:
            mark_width = max(len(row) for row in _BRAND_MARK_COMPACT) + 3
            for index, mark_row in enumerate(_BRAND_MARK_COMPACT):
                label = self.gradient("DroneDream") if index == 3 else ""
                print(
                    "   "
                    + self._brand_mark(mark_row, "", span=mark_width)
                    + " " * max(0, mark_width - len(mark_row))
                    + label,
                    file=self.stream,
                )
        print(file=self.stream)

    def _brand_lockup(
        self,
        mark: tuple[str, ...],
        white: tuple[str, ...],
        wordmark: tuple[str, ...],
    ) -> None:
        wordmark_start = (len(mark) - len(wordmark)) // 2
        mark_width = max(max(map(len, mark)), max(map(len, white)))
        wordmark_width = max(map(len, wordmark))
        for index, mark_row in enumerate(mark):
            wordmark_index = index - wordmark_start
            wordmark_row = (
                wordmark[wordmark_index]
                if 0 <= wordmark_index < len(wordmark)
                else ""
            )
            print(
                "   "
                + self._brand_mark(mark_row, white[index], span=mark_width)
                + " " * (mark_width - max(len(mark_row), len(white[index])) + 3)
                + self.gradient(wordmark_row, span=wordmark_width),
                file=self.stream,
            )

    def _brand_mark(self, line: str, white_line: str, *, span: int | None = None) -> str:
        """Paint the filled brand mark and preserve its white internal strokes."""
        width = max(len(line), len(white_line))
        if not self.color:
            return line.ljust(width)
        overlays = {
            index: (char, WHITE)
            for index, char in enumerate(white_line.ljust(width))
            if char != " "
            and (
                (ord(char) - 0x2800).bit_count() >= 3
                or (index >= width * 2 // 3 and (ord(char) - 0x2800).bit_count() >= 2)
            )
        }
        return self.gradient(line.ljust(width), overlays=overlays, span=span)

    def rule(self, label: str = "", *, width: int | None = None) -> None:
        label_text = f" {label} " if label else ""
        line_width = min(self.width, width or self.width)
        remaining = max(1, line_width - len(label_text))
        left = remaining // 2
        right = remaining - left
        print(
            self.gradient("─" * left)
            + self.rgb(label_text, WHITE, bold=True)
            + self.gradient("─" * right, (BLUE, PURPLE, PINK)),
            file=self.stream,
        )

    def status(
        self,
        label: str,
        value: object,
        detail: str | None = None,
        *,
        blank_after: bool = False,
    ) -> None:
        normalized = str(value).lower()
        color = (
            GREEN
            if normalized
            in {
                "ready",
                "healthy",
                "verified",
                "completed",
                "qualified",
                "qualified_builtin",
                "true",
            }
            else RED
            if normalized in {"failed", "error", "blocked", "aborted", "false"}
            else YELLOW
            if normalized
            in {"warning", "awaiting_confirmation", "pending"}
            else BLUE
        )
        label_text = str(label)
        value_text = str(value)
        suffix = f"  {self.rgb(detail, MUTED)}" if detail else ""
        marker_text = (
            "✓"
            if color == GREEN
            else "✕"
            if color == RED
            else "!"
            if color == YELLOW
            else "◆"
        )
        marker = self.rgb(marker_text, color, bold=True)
        title = self.rgb(label_text.ljust(12), WHITE, bold=True)
        state = self.rgb(value_text.ljust(12), color, bold=True)
        print(f"  {marker} {title}  {state}{suffix}", file=self.stream)
        if blank_after:
            print(file=self.stream)

    def command_prompt(self, account_name: str | None = None) -> None:
        identity = f" ({account_name})" if account_name else ""
        print(
            self.rgb("  ◆ ", PINK, bold=True)
            + self.gradient("DroneDream", bold=True)
            + self.rgb(identity, PURPLE, bold=True)
            + self.rgb(" › ", BLUE, bold=True),
            end="",
            flush=True,
            file=self.stream,
        )

    def panel(
        self, title: str, lines: Iterable[str], *, accent: tuple[int, int, int] = PURPLE
    ) -> None:
        width = self.width
        inner = width - 4
        title_text = f" {title.strip()} "
        top_fill = max(0, width - 2 - len(title_text))
        print(
            self.rgb("╭─", accent)
            + self.rgb(title_text, WHITE, bold=True)
            + self.rgb("─" * max(0, top_fill - 1) + "╮", accent),
            file=self.stream,
        )
        wrote = False
        for raw in lines:
            raw_text = str(raw)
            if "\x1b[" in raw_text and _visible_len(raw_text) <= inner:
                wrapped = [raw_text]
            else:
                wrapped = textwrap.wrap(
                    raw_text, width=inner, break_long_words=False, break_on_hyphens=False
                ) or [""]
            for line in wrapped:
                padding = max(0, inner - _visible_len(line))
                print(
                    self.rgb("│", accent)
                    + " "
                    + line
                    + " " * padding
                    + " "
                    + self.rgb("│", accent),
                    file=self.stream,
                )
                wrote = True
        if not wrote:
            print(
                self.rgb("│", accent) + " ".ljust(inner + 1) + self.rgb("│", accent),
                file=self.stream,
            )
        print(self.rgb("╰" + "─" * (width - 2) + "╯", accent), file=self.stream)

    def choice(self, kind: str, labels: list[str]) -> None:
        lines = []
        for index, label in enumerate(labels, start=1):
            badge = self.style(f" {index:02d} ", foreground=WHITE, background=SURFACE, bold=True)
            lines.append(f"{badge}  {label}")
        self.panel(f"SELECT {kind.upper()}", lines, accent=BLUE if kind == "map" else PINK)

    def menu(self, title: str, choices: list[tuple[str, str, str]]) -> None:
        """Render a compact keyboard-driven menu with stable letter shortcuts."""
        lines = []
        colors = (PINK, PURPLE, BLUE, CYAN)
        for index, (key, label, detail) in enumerate(choices):
            color = colors[index % len(colors)]
            badge = self.style(
                f" {key.upper()} ", foreground=WHITE, background=SURFACE, bold=True
            )
            lines.append(
                f"{badge}  {self.rgb(label, color, bold=True)}"
                + (f"  {self.rgb(detail, MUTED)}" if detail else "")
            )
        self.panel(title, lines, accent=PURPLE)

    def prompt(self, label: str, hint: str | None = None) -> None:
        suffix = f"  {self.rgb(hint, MUTED)}" if hint else ""
        print(
            self.rgb("  › ", PINK, bold=True)
            + self.rgb(label, WHITE, bold=True)
            + suffix
            + " ",
            end="",
            flush=True,
            file=self.stream,
        )

    def confirm_plan(self, plan_revision_id: str) -> None:
        self.panel(
            "LAUNCH CONFIRMATION",
            [
                self.rgb("The reviewed route is ready for PX4/Gazebo.", YELLOW, bold=True),
                f"Reviewed plan: {plan_revision_id}",
                "Type FLY to launch. Any other response leaves the mission unlaunched.",
            ],
            accent=YELLOW,
        )
        self.prompt("Confirm", "fly")

    def error(self, code: str, detail: str | None = None) -> None:
        lines = [detail] if detail else ["The command stopped before changing mission state."]
        self.panel(f"ERROR · {code}", lines, accent=RED)

    def next_step(self, command: str) -> None:
        print(file=self.stream)
        print(
            self.rgb("  NEXT  ", WHITE, bold=True)
            + self.rgb(command, BLUE, bold=True),
            file=self.stream,
        )

    def table(
        self, headers: list[str], rows: list[list[object]], *, accent: tuple[int, int, int] = PURPLE
    ) -> None:
        if not rows:
            self.panel("No results", ["Nothing matched the current selection."], accent=accent)
            return
        values = [[str(item) for item in row] for row in rows]
        available = self.width - (3 * (len(headers) - 1))
        natural = [
            max(_visible_len(headers[i]), *(len(row[i]) for row in values))
            for i in range(len(headers))
        ]
        widths = natural[:]
        while sum(widths) > available:
            index = max(range(len(widths)), key=lambda item: widths[item])
            floor = 10 if index == 0 else 8
            if widths[index] <= floor:
                break
            widths[index] -= 1

        def fit(value: str, width: int) -> str:
            return value if len(value) <= width else value[: max(1, width - 1)] + "…"

        header = " │ ".join(
            self.rgb(fit(headers[i], widths[i]).ljust(widths[i]), WHITE, bold=True)
            for i in range(len(headers))
        )
        print(header, file=self.stream)
        print(self.rgb("─┼─".join("─" * width for width in widths), accent), file=self.stream)
        for row in values:
            cells = []
            for index in range(len(headers)):
                value = fit(row[index], widths[index]).ljust(widths[index])
                normalized = row[index].casefold()
                color = (
                    GREEN
                    if normalized in {"ready", "verified", "completed", "qualified_builtin", "yes"}
                    else RED
                    if normalized in {"failed", "blocked", "aborted", "no"}
                    else WHITE
                    if index == 0
                    else MUTED
                )
                cells.append(self.rgb(value, color, bold=index == 0))
            print(
                self.rgb(" │ ", accent).join(cells),
                file=self.stream,
            )

    def json(self, value: object) -> None:
        print(json.dumps(value, ensure_ascii=False, indent=2), file=self.stream)

    def render(self, value: dict, view: str, *, show_banner: bool = True) -> None:
        if show_banner:
            self.banner()
        renderer = getattr(self, f"_render_{view.replace('-', '_')}", None)
        if callable(renderer):
            renderer(value)
        else:
            self._render_generic(value)

    def _render_models(self, value: dict) -> None:
        rows = []
        for item in value.get("models", []):
            if isinstance(item, dict):
                rows.append(
                    [
                        item.get("id", ""),
                        item.get("provider", ""),
                        item.get("model", ""),
                        item.get("source", ""),
                    ]
                )
        self.table(["MODEL ID", "PROVIDER", "MODEL", "SOURCE"], rows, accent=PINK)

    def _render_maps(self, value: dict) -> None:
        rows = []
        for item in value.get("resources", []):
            if isinstance(item, dict):
                name = item.get("display_name", {})
                rows.append(
                    [
                        item.get("resource_id", ""),
                        _localized(name),
                        item.get("environment_class", ""),
                        item.get("catalog_status", item.get("status", "reviewed")),
                    ]
                )
        self.table(["MAP ID", "NAME", "ENVIRONMENT", "STATUS"], rows, accent=BLUE)

    def _render_aircraft(self, value: dict) -> None:
        rows = []
        for item in value.get("resources", []):
            if isinstance(item, dict):
                analysis = (
                    item.get("analysis", {}) if isinstance(item.get("analysis"), dict) else {}
                )
                rows.append(
                    [
                        item.get("resource_id", ""),
                        _localized(item.get("display_name", {})),
                        item.get("vehicle_class", ""),
                        ", ".join(
                            str(sensor)
                            for sensor in analysis.get(
                                "resolved_sensor_types",
                                analysis.get("sensors", item.get("sensors", [])),
                            )[:4]
                        ),
                    ]
                )
        self.table(["AIRCRAFT ID", "NAME", "CLASS", "KEY SENSORS"], rows, accent=PINK)

    def _render_pairs(self, value: dict) -> None:
        rows = []
        for item in value.get("pairs", []):
            if isinstance(item, dict):
                rows.append(
                    [
                        item.get("map_resource_id", ""),
                        item.get("vehicle_resource_id", ""),
                        item.get("status", ""),
                        item.get("qualification_id") or "—",
                    ]
                )
        self.table(["MAP", "AIRCRAFT", "QUALIFICATION", "RECEIPT"], rows, accent=PURPLE)
        counts = value.get("counts", {})
        if isinstance(counts, dict):
            self.status(
                "Ready now",
                counts.get("qualified_builtin", 0),
                "pair(s) with verified PX4/Gazebo evidence",
            )

    def _render_missions(self, value: dict) -> None:
        rows = []
        for item in value.get("threads", []):
            if isinstance(item, dict):
                rows.append(
                    [
                        item.get("thread_id", ""),
                        item.get("title", ""),
                        item.get("state", ""),
                        item.get("updated_at", ""),
                        "yes" if item.get("archived") else "no",
                    ]
                )
        self.table(["MISSION", "TITLE", "STATE", "UPDATED", "ARCHIVED"], rows, accent=PURPLE)

    def _render_runtime(self, value: dict) -> None:
        self.rule("RUNTIME")
        for key in (
            "status",
            "state",
            "runtime_available",
            "resources_ready",
            "provisioned",
            "runtime_ready",
            "services_ready",
            "engine_ready",
        ):
            if key in value:
                self.status(key.replace("_", " ").title(), value[key])
        self._render_key_values(
            value,
            excluded={
                "status",
                "state",
                "runtime_available",
                "resources_ready",
                "provisioned",
                "runtime_ready",
                "services_ready",
                "engine_ready",
            },
        )

    def _render_account(self, value: dict) -> None:
        account = value.get("account") if isinstance(value.get("account"), dict) else {}
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
        self.rule("ACCOUNT")
        self.status("Pilot", "signed in", str(account.get("display_name") or "Pilot"))
        if account.get("email"):
            self.status("Email", account["email"])
        self.status("Edition", account.get("edition", "—"))
        if usage:
            self.rule("COMPUTE")
            self._render_key_values(usage)

    def _render_usage(self, value: dict) -> None:
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else value
        self.rule("COMPUTE ALLOWANCE")
        self._render_key_values(usage)

    def _render_plan(self, value: dict) -> None:
        plan = value.get("mission_plan") if isinstance(value.get("mission_plan"), dict) else value
        plan_id = value.get("plan_revision_id") or plan.get("plan_revision_id") or "unknown"
        goal = plan.get("goal") or value.get("goal") or "Plan prepared"
        self.panel(
            "MODEL PLAN",
            [
                str(goal),
                f"Plan revision: {plan_id}",
                f"Model: {value.get('model_id', '—')} · Source: {value.get('model_source', '—')}",
            ],
            accent=PINK,
        )
        graph = plan.get("task_graph", {}) if isinstance(plan, dict) else {}
        nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        if isinstance(nodes, list) and nodes:
            rows = []
            for index, node in enumerate(nodes, start=1):
                if isinstance(node, dict):
                    rows.append(
                        [
                            index,
                            node.get("action") or node.get("label", ""),
                            node.get("target") or node.get("executor", ""),
                            ", ".join(
                                node.get("success_evidence") or node.get("completion_evidence", [])
                            ),
                        ]
                    )
            self.table(["#", "ACTION", "TARGET", "SUCCESS EVIDENCE"], rows, accent=BLUE)
        metrics = plan.get("metrics", {}) if isinstance(plan, dict) else {}
        if isinstance(metrics, dict) and metrics:
            self.rule("TOOL-PLANNED ROUTE")
            for label, key, suffix in (
                ("Route length", "route_length_m", " m"),
                ("Estimated duration", "estimated_duration_s", " s"),
                ("Minimum clearance", "minimum_clearance_m", " m"),
            ):
                if metrics.get(key) is not None:
                    self.status(label, f"{metrics[key]}{suffix}")
        proof = value.get("model_tool_proof", {})
        receipts = proof.get("tool_receipts", []) if isinstance(proof, dict) else []
        if isinstance(receipts, list) and receipts:
            rows = [
                [item.get("tool_id", ""), item.get("outcome", ""), item.get("call_id", "")]
                for item in receipts
                if isinstance(item, dict)
            ]
            self.rule("MODEL TOOL RECEIPTS")
            self.table(["TOOL", "OUTCOME", "CALL"], rows, accent=PURPLE)
        self.status("Model calls", value.get("model_calls", "—"))
        self.status("Planning rounds", value.get("planning_attempts", "—"))
        self.next_step(f"Review this plan, then confirm revision {plan_id}")

    def _render_mission(self, value: dict) -> None:
        self.panel(
            str(value.get("title", "Mission")),
            [
                f"ID: {value.get('thread_id', '—')}",
                f"Model: {value.get('selected_model', '—')}",
                f"Map: {value.get('selected_map_id') or 'not selected'}",
                f"Aircraft: {value.get('selected_vehicle_id') or 'not selected'}",
            ],
            accent=PURPLE,
        )
        self.status("Mission state", value.get("state", "unknown"))
        messages = value.get("messages", [])
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                self.panel("LATEST UPDATE", [str(last.get("content", ""))], accent=BLUE)

    def _render_telemetry(self, value: dict) -> None:
        self.rule("LIVE TELEMETRY")
        phase = value.get("phase", value.get("state", "unknown"))
        self.status("Flight phase", phase)
        self._render_key_values(value, excluded={"phase", "state"})

    def _render_evidence(self, value: dict) -> None:
        self.rule("EXECUTION EVIDENCE")
        self.status("Execution", value.get("state", "unknown"), str(value.get("execution_id", "")))
        result = value.get("result")
        if isinstance(result, dict):
            self.status(
                "Workflow", result.get("status", "unknown"), str(result.get("contract_id", ""))
            )
            gates = result.get("gates")
            if isinstance(gates, dict):
                required = (
                    "executor_completed",
                    "offboard_timing_complete",
                    "runtime_pose_samples_present",
                    "ros_observations_present",
                    "goal_observed",
                    "landing_confirmed",
                    "native_terminal_lifecycle_published",
                    "no_live_abort",
                    "px4_ulog_present",
                    "static_route_clearance_bound",
                )
                for key in required:
                    if key in gates:
                        self.status(key.replace("_", " ").title(), gates[key])
            measurements = result.get("measurements")
            if isinstance(measurements, dict):
                self._render_key_values(measurements)
            actions = result.get("runtime_action_receipts")
            if isinstance(actions, list) and actions:
                rows = [
                    [
                        item.get("step_id", ""),
                        item.get("action", ""),
                        item.get("status", ""),
                        item.get("attempts", ""),
                    ]
                    for item in actions
                    if isinstance(item, dict)
                ]
                self.rule("MISSION ACTIONS")
                self.table(["STEP", "ACTION", "STATUS", "TRIES"], rows, accent=PINK)
            checkpoint_calls = result.get("checkpoint_model_calls", [])
            if not isinstance(checkpoint_calls, list):
                checkpoint_calls = []
            calls = [result.get("completion_model_call"), *checkpoint_calls]
            calls = [item for item in calls if isinstance(item, dict)]
            if calls:
                rows = [
                    [
                        item.get("role", ""),
                        item.get("provider", ""),
                        item.get("model", ""),
                        item.get("response_id", ""),
                    ]
                    for item in calls
                ]
                self.rule("MODEL PARTICIPATION")
                self.table(["ROLE", "PROVIDER", "MODEL", "RESPONSE"], rows, accent=BLUE)

    def _render_fly(self, value: dict) -> None:
        self.rule("FLIGHT RESULT")
        self.status("Mission", value.get("status", "unknown"), str(value.get("thread_id", "")))
        self.status("Plan", value.get("plan_revision_id", "—"))
        if value.get("execution_id"):
            self.status("Execution", value["execution_id"])
        if "archived" in value:
            self.status("Archived", value["archived"])
        evidence = value.get("evidence")
        if isinstance(evidence, dict):
            self._render_evidence(evidence)

    def _render_doctor(self, value: dict) -> None:
        self.rule("SYSTEM CHECK")
        for item in value.get("checks", []):
            if isinstance(item, dict):
                self.status(
                    str(item.get("name", "check")),
                    item.get("status", "unknown"),
                    str(item.get("detail", "")),
                )

    def _render_generic(self, value: dict) -> None:
        self._render_key_values(value)

    def _render_key_values(self, value: dict, *, excluded: set[str] | None = None) -> None:
        excluded = excluded or set()
        rows: list[list[object]] = []
        for key, item in value.items():
            if key in excluded:
                continue
            if isinstance(item, (dict, list)):
                rendered = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            else:
                rendered = item
            rows.append([key.replace("_", " "), rendered])
        self.table(["FIELD", "VALUE"], rows, accent=BLUE)


def _localized(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("en-US") or value.get("zh-CN") or next(iter(value.values()), ""))
    return str(value or "")


__all__ = ["TerminalUI", "PINK", "PURPLE", "BLUE", "GREEN", "YELLOW", "RED"]
