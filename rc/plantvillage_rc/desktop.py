"""Simple Tkinter desktop demo for the layered hardware-RC pipeline.

The GUI is deliberately small: it replays the processing-layer output and the
measured-current response recorded for one of the 300 samples, then shows the
readout decision.  It adds no GUI dependency beyond Python's standard library
(Tkinter) and the project's existing Pillow dependency.

Run::

    PYTHONPATH=rc .venv/bin/python -m plantvillage_rc.desktop

This is a demo/verification surface, not the serving API.  The three layers
remain importable without importing this module.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, font as tkfont, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageTk

from .layers import (
    HardwareRCPipeline,
    LeafImageProcessor,
    ReadoutDecisionLayer,
    RecordedHardwareBackend,
    RecordedHardwareDataset,
)
from .layers.contracts import DRIVE_CHANNELS, RESPONSE_CHANNELS

ROOT = Path(__file__).resolve().parents[2]

# A deliberately small, neutral palette.  Tkinter is not a design tool; the
# point of this window is to make the layer contracts and real outputs legible.
BG = "#eef2f6"
CARD = "#ffffff"
INK = "#17212b"
MUTED = "#687586"
BORDER = "#d7e0e8"
ACCENT = "#0f766e"
ACCENT_DARK = "#115e59"
AMBER = "#b45309"
AMBER_BG = "#fff7ed"
RED = "#b91c1c"
GREEN = "#15803d"
GRID = "#e8eef3"

CLASS_ZH = {
    "Blueberry___healthy": "Blueberry / healthy",
    "Cherry_(including_sour)___healthy": "Cherry / healthy",
    "Corn_(maize)___Common_rust_": "Corn / common rust",
    "Corn_(maize)___healthy": "Corn / healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape / leaf blight",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange / huanglongbing",
    "Peach___healthy": "Peach / healthy",
    "Soybean___healthy": "Soybean / healthy",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato / leaf curl",
    "Tomato___healthy": "Tomato / healthy",
}


def class_display(name: str) -> str:
    return CLASS_ZH.get(str(name), str(name).replace("___", " / ").replace("_", " "))


def _default_model() -> Path | None:
    candidates = (
        ROOT / "artifacts_hardware300_v2/run/model.joblib",
        ROOT / "artifacts_hardware300/model.joblib",
    )
    return next((path for path in candidates if path.is_file()), None)


def _dataset_paths() -> tuple[Path, Path]:
    base = ROOT / "artifacts_hardware300/sources"
    return base / "responses.csv", base / "inputs.csv"


def _resolve_sample_image(sample_id: str) -> Path | None:
    """Resolve a PlantVillage sample path without allowing path traversal."""
    data_root = (ROOT / "data").resolve()
    candidate = (data_root / sample_id).resolve()
    try:
        candidate.relative_to(data_root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _drive_mosaic(drive: Any) -> Image.Image | None:
    """Render the 64 patch colours in snake order as an honest fallback preview."""
    patches = getattr(drive, "patches_rgb", None)
    if patches is None:
        return None
    tiles = np.asarray(patches, dtype=np.float32)[:, :3]
    if tiles.shape != (64, 3):
        return None
    grid = np.zeros((8, 8, 3), dtype=np.uint8)
    for step, colour in enumerate(tiles):
        row, col = divmod(step, 8)
        if row % 2:
            col = 7 - col
        grid[row, col] = np.clip(colour * 255.0, 0, 255).astype(np.uint8)
    # Each tile represents one 8x8 image patch.  Nearest-neighbour keeps the
    # 8x8 acquisition grid visible instead of pretending this is the original.
    return Image.fromarray(grid).resize((128, 128), Image.Resampling.NEAREST)


def _mix(first: str, second: str, amount: float) -> str:
    amount = max(0.0, min(1.0, float(amount)))
    a = tuple(int(first[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(second[i : i + 2], 16) for i in (1, 3, 5))
    value = tuple(round(x + (y - x) * amount) for x, y in zip(a, b))
    return "#" + "".join(f"{part:02x}" for part in value)


def _drive_colour(value: float, channel: int) -> str:
    if value <= 1e-7:
        return "#f2f6f9"
    palette = (
        "#dc2626",
        "#16a34a",
        "#2563eb",
        "#ea580c",
        "#ea580c",
        "#ea580c",
        "#7c3aed",
        "#7c3aed",
        "#7c3aed",
    )
    return _mix("#f2f6f9", palette[channel], 0.18 + 0.82 * math.sqrt(min(value, 1.0)))


def _current_colour(value: float, limit: float) -> str:
    zero = "#f7f9fb"
    if limit <= 0:
        return zero
    amount = min(abs(float(value)) / limit, 1.0)
    return _mix(zero, "#dc2626" if value >= 0 else "#2563eb", 0.15 + 0.85 * amount)


def _write_png_screenshot(root: tk.Misc, path: Path) -> bool:
    """Capture the real rendered Tk window into a PNG (X11/Windows/macOS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    root.update_idletasks()
    root.update()
    try:
        from PIL import ImageGrab

        x = root.winfo_rootx()
        y = root.winfo_rooty()
        image = ImageGrab.grab(
            bbox=(x, y, x + root.winfo_width(), y + root.winfo_height()),
            include_layered_windows=False,
        )
        image.save(path)
        return True
    except Exception:
        return False


class ImagePane(tk.Canvas):
    """An aspect-fit image preview with a small fallback-mosaic mode."""

    def __init__(self, master: tk.Misc, **kwargs: Any):
        super().__init__(
            master,
            bg="#f3f6f8",
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._mosaic = False
        self.bind("<Configure>", lambda _event: self._redraw())

    def set_image(self, image: Image.Image | None, *, mosaic: bool = False) -> None:
        self._image = image.convert("RGB") if image is not None else None
        self._mosaic = mosaic
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        width, height = max(self.winfo_width(), 2), max(self.winfo_height(), 2)
        if self._image is None:
            self.create_text(
                width / 2,
                height / 2,
                text="No image",
                fill=MUTED,
                font=("TkDefaultFont", 10),
            )
            return
        margin = 8
        scale = min(
            (width - 2 * margin) / self._image.width,
            (height - 2 * margin) / self._image.height,
        )
        size = (
            max(1, round(self._image.width * scale)),
            max(1, round(self._image.height * scale)),
        )
        resample = Image.Resampling.NEAREST if self._mosaic else Image.Resampling.LANCZOS
        rendered = self._image.resize(size, resample)
        self._photo = ImageTk.PhotoImage(rendered)
        x = (width - size[0]) / 2
        y = (height - size[1]) / 2
        self.create_image(x, y, image=self._photo, anchor="nw")
        self.create_rectangle(
            x,
            y,
            x + size[0],
            y + size[1],
            outline="#bfcbd6",
            width=1,
        )


class DriveCanvas(tk.Canvas):
    """Draw a compact [64, 9] frame matrix: rows are channels, columns are steps."""

    def __init__(self, master: tk.Misc, **kwargs: Any):
        super().__init__(master, bg=CARD, highlightthickness=0, bd=0, **kwargs)
        self._frames: np.ndarray | None = None
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_frames(self, frames: np.ndarray | None) -> None:
        self._frames = None if frames is None else np.asarray(frames, dtype=np.float32)
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width, height = max(self.winfo_width(), 2), max(self.winfo_height(), 2)
        if self._frames is None:
            self.create_text(
                width / 2,
                height / 2,
                text="Waiting for drive signals",
                fill=MUTED,
                font=("TkDefaultFont", 9),
            )
            return
        left, top, bottom = 40, 7, 19
        plot_width = max(width - left - 2, 10)
        plot_height = max(height - top - bottom, 10)
        row_height = plot_height / 9.0
        step_width = plot_width / 64.0
        labels = ("R", "G", "B", "ON R", "ON G", "ON B", "OFF R", "OFF G", "OFF B")
        for channel in range(9):
            y0 = top + channel * row_height
            self.create_text(
                3,
                y0 + row_height / 2,
                text=labels[channel],
                fill=MUTED,
                anchor="w",
                font=("TkFixedFont", 7),
            )
            for step in range(64):
                x0 = left + step * step_width
                self.create_rectangle(
                    x0,
                    y0,
                    left + (step + 1) * step_width,
                    y0 + row_height,
                    fill=_drive_colour(float(self._frames[step, channel]), channel),
                    outline="",
                )
        self.create_text(
            left,
            height - 4,
            text="step 1",
            fill=MUTED,
            anchor="sw",
            font=("TkFixedFont", 7),
        )
        self.create_text(
            left + plot_width,
            height - 4,
            text="64",
            fill=MUTED,
            anchor="se",
            font=("TkFixedFont", 7),
        )


class CurrentCanvas(tk.Canvas):
    """Draw [9,12,65] amperes as 9 x 65 row-averaged current traces."""

    def __init__(self, master: tk.Misc, **kwargs: Any):
        super().__init__(master, bg=CARD, highlightthickness=0, bd=0, **kwargs)
        self._currents: np.ndarray | None = None
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_currents(self, currents: np.ndarray | None) -> None:
        self._currents = None if currents is None else np.asarray(currents, dtype=np.float64)
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width, height = max(self.winfo_width(), 2), max(self.winfo_height(), 2)
        if self._currents is None:
            self.create_text(
                width / 2,
                height / 2,
                text="Waiting for hardware response",
                fill=MUTED,
                font=("TkDefaultFont", 9),
            )
            return
        mean = self._currents.mean(axis=1) * 1e6
        limit = float(np.max(np.abs(mean))) if mean.size else 0.0
        limit = max(limit, 1e-9)
        left, top, bottom = 40, 8, 35
        plot_width = max(width - left - 4, 10)
        plot_height = max(height - top - bottom, 10)
        row_height = plot_height / 9.0
        point_width = plot_width / 65.0
        labels = ("R", "G", "B", "ONr", "ONg", "ONb", "OFr", "OFg", "OFb")
        for channel in range(9):
            y0 = top + channel * row_height
            self.create_text(
                3,
                y0 + row_height / 2,
                text=labels[channel],
                fill=MUTED,
                anchor="w",
                font=("TkFixedFont", 7),
            )
            for point in range(65):
                x0 = left + point * point_width
                self.create_rectangle(
                    x0,
                    y0,
                    left + (point + 1) * point_width,
                    y0 + row_height,
                    fill=_current_colour(float(mean[channel, point]), limit),
                    outline="",
                )
        # The recorded response has 64 stimulation steps plus one tail point.
        tail_x = left + 64 * point_width
        self.create_line(
            tail_x,
            top,
            tail_x,
            top + plot_height,
            fill="#64748b",
            width=1,
        )
        self.create_text(
            tail_x - 3,
            height - 4,
            text="tail",
            fill="#64748b",
            anchor="se",
            font=("TkFixedFont", 7),
        )
        self.create_text(
            left,
            height - 4,
            text="step 1",
            fill=MUTED,
            anchor="sw",
            font=("TkFixedFont", 7),
        )
        self.create_text(
            left + 64 * point_width - 4,
            height - 4,
            text="64",
            fill=MUTED,
            anchor="se",
            font=("TkFixedFont", 7),
        )
        self.create_text(
            left,
            height - 18,
            text=f"-{limit:.2f} uA",
            fill="#2563eb",
            anchor="w",
            font=("TkFixedFont", 7),
        )
        self.create_text(
            width / 2,
            height - 18,
            text="0",
            fill=MUTED,
            anchor="center",
            font=("TkFixedFont", 7),
        )
        self.create_text(
            width - 4,
            height - 18,
            text=f"+{limit:.2f} uA",
            fill="#dc2626",
            anchor="e",
            font=("TkFixedFont", 7),
        )


class ProbabilityCanvas(tk.Canvas):
    """A simple top-five probability bar chart."""

    def __init__(self, master: tk.Misc, **kwargs: Any):
        super().__init__(master, bg=CARD, highlightthickness=0, bd=0, **kwargs)
        self._rows: list[tuple[str, float, bool]] = []
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_rows(self, rows: list[tuple[str, float, bool]]) -> None:
        self._rows = rows[:5]
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width, height = max(self.winfo_width(), 2), max(self.winfo_height(), 2)
        if not self._rows:
            self.create_text(
                width / 2,
                height / 2,
                text="Waiting for decision",
                fill=MUTED,
                font=("TkDefaultFont", 9),
            )
            return
        row_height = min(38.0, height / max(len(self._rows), 1))
        label_width = min(126, max(90, int(width * 0.42)))
        bar_left = label_width + 8
        bar_right = max(bar_left + 12, width - 55)
        for index, (name, probability, predicted) in enumerate(self._rows):
            centre_y = index * row_height + row_height / 2
            self.create_text(
                2,
                centre_y,
                text=class_display(name),
                fill=INK if predicted else MUTED,
                anchor="w",
                font=("TkDefaultFont", 8, "bold" if predicted else "normal"),
            )
            bar_y0 = centre_y - 7
            bar_y1 = centre_y + 7
            self.create_rectangle(
                bar_left,
                bar_y0,
                bar_right,
                bar_y1,
                fill=GRID,
                outline="",
            )
            filled = bar_left + (bar_right - bar_left) * max(0.0, min(probability, 1.0))
            self.create_rectangle(
                bar_left,
                bar_y0,
                filled,
                bar_y1,
                fill=ACCENT if predicted else "#94a3b8",
                outline="",
            )
            self.create_text(
                width - 2,
                centre_y,
                text=f"{probability * 100:.1f}%",
                fill=INK if predicted else MUTED,
                anchor="e",
                font=("TkFixedFont", 8, "bold" if predicted else "normal"),
            )


@dataclass
class GuiPaths:
    model: Path | None
    responses: Path
    inputs: Path


class DesktopApp:
    """Small Tk dashboard around the three pipeline layers."""

    def __init__(
        self,
        root: tk.Tk,
        paths: GuiPaths,
        *,
        initial_sample: int = 0,
        auto_run: bool = True,
    ):
        self.root = root
        self.paths = paths
        self.initial_sample = initial_sample
        self.auto_run = auto_run
        self.dataset: RecordedHardwareDataset | None = None
        self.dataset_error: str | None = None
        self.recorded: RecordedHardwareBackend | None = None
        self.processor = LeafImageProcessor()
        self.decision = ReadoutDecisionLayer(paths.model)
        self.current_result: Any | None = None
        self.current_index: int | None = None
        self._queue: queue.Queue[Any] = queue.Queue()
        self._busy = False
        self._loading = False
        self._font_family = "TkDefaultFont"
        self._mono_family = "TkFixedFont"
        self._setup_fonts()
        self._setup_style()
        self._build()
        self._load_dataset()
        if self.dataset is not None:
            index = max(0, min(initial_sample, len(self.dataset.metadata) - 1))
            self.sample_combo.current(index)
            self._on_sample_change()
            if auto_run:
                self.root.after(180, self.run_selected)

    # ------------------------------------------------------------------ setup
    def _setup_fonts(self) -> None:
        families = set(tkfont.families(self.root))
        for family in (
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "WenQuanYi Zen Hei",
            "DejaVu Sans",
            "TkDefaultFont",
        ):
            if family in families:
                self._font_family = family
                break
        for family in ("JetBrains Mono", "Noto Sans Mono", "DejaVu Sans Mono", "TkFixedFont"):
            if family in families:
                self._mono_family = family
                break
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            try:
                tkfont.nametofont(name).configure(family=self._font_family, size=10)
            except tk.TclError:
                pass
        tkfont.nametofont("TkFixedFont").configure(family=self._mono_family, size=9)

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            ".",
            font=(self._font_family, 10),
            background=BG,
            foreground=INK,
        )
        style.configure(
            "TCombobox",
            padding=6,
            fieldbackground=CARD,
            background=CARD,
            foreground=INK,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", CARD)],
            foreground=[("readonly", INK)],
        )
        style.configure(
            "Accent.TButton",
            padding=(10, 9),
            background=ACCENT,
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            font=(self._font_family, 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_DARK), ("disabled", "#9bb8b5")],
            foreground=[("disabled", "#f0f6f5")],
        )
        style.configure(
            "Quiet.TButton",
            padding=(8, 6),
            background="#edf2f6",
            foreground=INK,
            bordercolor=BORDER,
            relief="flat",
        )
        style.map("Quiet.TButton", background=[("active", "#dfe8ee")])
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#e2e8ee",
            background=ACCENT,
            bordercolor="#e2e8ee",
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )

    def _card(self, parent: tk.Misc) -> tuple[tk.Frame, tk.Frame]:
        border = tk.Frame(parent, bg=BORDER, bd=0, highlightthickness=0)
        inner = tk.Frame(border, bg=CARD, padx=14, pady=13, bd=0)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        return border, inner

    def _section_label(self, parent: tk.Misc, text: str) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            bg=parent.cget("bg"),
            fg=MUTED,
            anchor="w",
            font=(self._font_family, 8, "bold"),
        )

    def _stage_header(
        self, parent: tk.Misc, number: str, title: str, english: str, accent: str
    ) -> None:
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=(0, 9))
        tk.Label(
            row,
            text=number,
            bg=accent,
            fg="#ffffff",
            width=3,
            height=1,
            font=(self._mono_family, 9, "bold"),
        ).pack(side="left")
        text_frame = tk.Frame(row, bg=CARD)
        text_frame.pack(side="left", padx=(8, 0), fill="x", expand=True)
        tk.Label(
            text_frame,
            text=title,
            bg=CARD,
            fg=INK,
            anchor="w",
            font=(self._font_family, 12, "bold"),
        ).pack(fill="x")
        tk.Label(
            text_frame,
            text=english,
            bg=CARD,
            fg=MUTED,
            anchor="w",
            font=(self._mono_family, 7),
        ).pack(fill="x")
        tk.Label(
            row,
            text="READY",
            bg="#e7f5ef",
            fg=GREEN,
            padx=6,
            pady=3,
            font=(self._mono_family, 7, "bold"),
        ).pack(side="right")

    def _build(self) -> None:
        self.root.configure(bg=BG)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self._build_header()
        body = tk.Frame(self.root, bg=BG, padx=14, pady=12)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        self._build_sidebar(body)
        self._build_main(body)

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg="#ffffff", padx=18, pady=10, bd=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        tk.Label(
            header,
            text="RC",
            bg=ACCENT,
            fg="#ffffff",
            width=3,
            height=1,
            font=(self._mono_family, 12, "bold"),
        ).grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 10))
        tk.Label(
            header,
            text="Hardware RC | Layered Decision Console",
            bg="#ffffff",
            fg=INK,
            anchor="w",
            font=(self._font_family, 17, "bold"),
        ).grid(row=0, column=1, sticky="w")
        tk.Label(
            header,
            text="Processing layer  ->  Hardware response  ->  Readout decision",
            bg="#ffffff",
            fg=MUTED,
            anchor="w",
            font=(self._font_family, 9),
        ).grid(row=1, column=1, sticky="w")
        chips = tk.Frame(header, bg="#ffffff")
        chips.grid(row=0, column=2, rowspan=2, sticky="e")
        for text, fg, bg in (
            ("MEASURED REPLAY", ACCENT_DARK, "#e7f5ef"),
            ("300 SAMPLES", "#475569", "#eef2f6"),
            ("LOCAL ONLY", "#475569", "#eef2f6"),
        ):
            tk.Label(
                chips,
                text=text,
                bg=bg,
                fg=fg,
                padx=8,
                pady=4,
                font=(self._font_family, 8, "bold"),
            ).pack(side="left", padx=(6, 0))

    def _build_sidebar(self, parent: tk.Misc) -> None:
        card, inner = self._card(parent)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        card.grid_propagate(False)
        card.configure(width=296)
        inner.grid_columnconfigure(0, weight=1)

        self._section_label(inner, "SAMPLE SOURCE").grid(
            row=0, column=0, sticky="w"
        )
        self.dataset_value = tk.Label(
            inner,
            text="Loading recorded dataset...",
            bg=CARD,
            fg=INK,
            anchor="w",
            font=(self._font_family, 14, "bold"),
        )
        self.dataset_value.grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.dataset_detail = tk.Label(
            inner,
            text="artifacts_hardware300/sources",
            bg=CARD,
            fg=MUTED,
            anchor="w",
            font=(self._mono_family, 7),
        )
        self.dataset_detail.grid(row=2, column=0, sticky="w", pady=(2, 12))

        self.sample_combo = ttk.Combobox(inner, state="readonly", height=18)
        self.sample_combo.grid(row=3, column=0, sticky="ew")
        self.sample_combo.bind("<<ComboboxSelected>>", self._on_sample_change)

        navigation = tk.Frame(inner, bg=CARD)
        navigation.grid(row=4, column=0, sticky="ew", pady=(6, 8))
        navigation.grid_columnconfigure(1, weight=1)
        ttk.Button(
            navigation,
            text="<",
            width=3,
            style="Quiet.TButton",
            command=lambda: self._step_sample(-1),
        ).grid(row=0, column=0)
        ttk.Button(
            navigation,
            text=">",
            width=3,
            style="Quiet.TButton",
            command=lambda: self._step_sample(1),
        ).grid(row=0, column=2)
        self.run_button = ttk.Button(
            navigation,
            text="Run end-to-end pipeline",
            style="Accent.TButton",
            command=self.run_selected,
        )
        self.run_button.grid(row=0, column=1, sticky="ew", padx=6)
        self.progress = ttk.Progressbar(
            inner, mode="indeterminate", style="Horizontal.TProgressbar"
        )
        self.progress.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.progress.grid_remove()

        tk.Frame(inner, bg=GRID, height=1).grid(
            row=6, column=0, sticky="ew", pady=(2, 9)
        )
        self.sample_title = tk.Label(
            inner,
            text="No sample selected",
            bg=CARD,
            fg=MUTED,
            anchor="w",
            font=(self._font_family, 10, "bold"),
        )
        self.sample_title.grid(row=7, column=0, sticky="w")
        self.gold_label = tk.Label(
            inner,
            text="-",
            bg=AMBER_BG,
            fg=AMBER,
            anchor="w",
            justify="left",
            wraplength=255,
            padx=8,
            pady=7,
            font=(self._font_family, 10, "bold"),
        )
        self.gold_label.grid(row=8, column=0, sticky="ew", pady=(5, 3))
        self.label_caveat = tk.Label(
            inner,
            text="Demo label only; it is never passed to any layer.",
            bg=CARD,
            fg=MUTED,
            anchor="w",
            justify="left",
            wraplength=255,
            font=(self._font_family, 7),
        )
        self.label_caveat.grid(row=9, column=0, sticky="w")
        self.sample_detail = tk.Label(
            inner,
            text="",
            bg=CARD,
            fg=MUTED,
            anchor="w",
            justify="left",
            wraplength=255,
            font=(self._mono_family, 7),
        )
        self.sample_detail.grid(row=10, column=0, sticky="w", pady=(5, 8))

        tk.Frame(inner, bg=GRID, height=1).grid(
            row=11, column=0, sticky="ew", pady=(1, 9)
        )
        self._section_label(inner, "LAYER CONTRACTS").grid(
            row=12, column=0, sticky="w", pady=(0, 5)
        )
        self.layer_labels: list[tk.Label] = []
        for text in (
            "1  Processing    recorded drive  [64, 9]",
            "2  Hardware      [9, 12, 65] A",
            "3  Decision      10-class probabilities",
        ):
            label = tk.Label(
                inner,
                text=text,
                bg=CARD,
                fg=INK,
                anchor="w",
                font=(self._mono_family, 8),
            )
            label.grid(row=13 + len(self.layer_labels), column=0, sticky="w", pady=2)
            self.layer_labels.append(label)

        note = tk.Frame(inner, bg=AMBER_BG, padx=9, pady=8)
        note.grid(row=17, column=0, sticky="ew", pady=(12, 6))
        tk.Label(
            note,
            text="REPLAY NOTE",
            bg=AMBER_BG,
            fg=AMBER,
            anchor="w",
            font=(self._font_family, 8, "bold"),
        ).pack(fill="x")
        tk.Label(
            note,
            text=(
                "Drive frames are recorded in inputs.csv; currents come from "
                "responses.csv. No physical hardware is connected, and CSV "
                "labels are never used for inference."
            ),
            bg=AMBER_BG,
            fg="#92400e",
            anchor="w",
            justify="left",
            wraplength=248,
            font=(self._font_family, 8),
        ).pack(fill="x", pady=(3, 0))

        inner.grid_rowconfigure(18, weight=1)
        self.export_button = ttk.Button(
            inner,
            text="Export current result JSON",
            style="Quiet.TButton",
            command=self.export_result,
            state="disabled",
        )
        self.export_button.grid(row=19, column=0, sticky="ew", pady=(4, 0))
        self.sidebar_status = tk.Label(
            inner,
            text="Initializing...",
            bg=CARD,
            fg=MUTED,
            anchor="w",
            justify="left",
            wraplength=255,
            font=(self._font_family, 8),
        )
        self.sidebar_status.grid(row=20, column=0, sticky="ew", pady=(7, 0))

    def _build_main(self, parent: tk.Misc) -> None:
        main = tk.Frame(parent, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)
        stages = tk.Frame(main, bg=BG)
        stages.grid(row=0, column=0, sticky="nsew")
        stages.grid_rowconfigure(0, weight=1)
        for column in range(3):
            stages.grid_columnconfigure(column, weight=1, uniform="stage")
        self._build_processing_card(stages, 0)
        self._build_hardware_card(stages, 1)
        self._build_decision_card(stages, 2)
        self._build_footer(main)

    def _build_footer(self, parent: tk.Misc) -> None:
        footer = tk.Frame(parent, bg=BG)
        footer.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        footer.grid_columnconfigure(0, weight=1)
        self.status_label = tk.Label(
            footer,
            text="*  Initializing",
            bg=BG,
            fg=MUTED,
            anchor="w",
            font=(self._font_family, 8),
        )
        self.status_label.grid(row=0, column=0, sticky="w")
        self.timing_label = tk.Label(
            footer,
            text="processing -> hardware -> decision",
            bg=BG,
            fg=MUTED,
            anchor="e",
            font=(self._mono_family, 8),
        )
        self.timing_label.grid(row=0, column=1, sticky="e")

    def _build_processing_card(self, parent: tk.Misc, column: int) -> None:
        card, inner = self._card(parent)
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 6))
        self._stage_header(
            inner, "01", "Processing", "PROCESSING | IMAGE -> DRIVE", "#0f766e"
        )
        self.image_pane = ImagePane(inner, height=190)
        self.image_pane.pack(fill="x")
        self.image_caption = tk.Label(
            inner,
            text="Waiting for sample image",
            bg=CARD,
            fg=MUTED,
            anchor="w",
            justify="left",
            wraplength=280,
            font=(self._mono_family, 7),
        )
        self.image_caption.pack(fill="x", pady=(5, 9))
        row = tk.Frame(inner, bg=CARD)
        row.pack(fill="x", pady=(0, 4))
        tk.Label(
            row,
            text="DriveSignals",
            bg=CARD,
            fg=INK,
            anchor="w",
            font=(self._font_family, 9, "bold"),
        ).pack(side="left")
        self.drive_shape = tk.Label(
            row,
            text="[64, 9]",
            bg="#e7f5ef",
            fg=ACCENT_DARK,
            padx=6,
            pady=2,
            font=(self._mono_family, 8, "bold"),
        )
        self.drive_shape.pack(side="right")
        self.drive_canvas = DriveCanvas(inner, height=176)
        self.drive_canvas.pack(fill="both", expand=True, pady=(1, 4))
        self.drive_meta = tk.Label(
            inner,
            text="64 frames x 9 channels | snake scan | recorded input",
            bg=CARD,
            fg=MUTED,
            anchor="w",
            justify="left",
            wraplength=280,
            font=(self._font_family, 7),
        )
        self.drive_meta.pack(fill="x")
        tk.Label(
            inner,
            text="Processing does not classify; it only emits drive frames.",
            bg="#f8fafc",
            fg="#475569",
            anchor="w",
            justify="left",
            wraplength=280,
            padx=8,
            pady=6,
            font=(self._font_family, 8),
        ).pack(fill="x", pady=(8, 0))

    def _build_hardware_card(self, parent: tk.Misc, column: int) -> None:
        card, inner = self._card(parent)
        card.grid(row=0, column=column, sticky="nsew", padx=6)
        self._stage_header(
            inner, "02", "Hardware response", "HARDWARE | DRIVE -> CURRENT", "#b45309"
        )
        metric = tk.Frame(inner, bg=CARD)
        metric.pack(fill="x", pady=(0, 3))
        tk.Label(
            metric,
            text="mean",
            bg=CARD,
            fg=MUTED,
            anchor="w",
            font=(self._font_family, 8),
        ).pack(side="left", anchor="s", pady=(0, 3))
        self.current_metric = tk.Label(
            metric,
            text="-",
            bg=CARD,
            fg=AMBER,
            anchor="w",
            font=(self._mono_family, 20, "bold"),
        )
        self.current_metric.pack(side="left", padx=(5, 4), anchor="s")
        tk.Label(
            metric,
            text="uA  |I|",
            bg=CARD,
            fg=MUTED,
            anchor="s",
            font=(self._font_family, 8),
        ).pack(side="left", anchor="s", pady=(0, 4))
        self.response_shape = tk.Label(
            metric,
            text="[9, 12, 65] A",
            bg="#fff3e6",
            fg=AMBER,
            padx=6,
            pady=2,
            font=(self._mono_family, 8, "bold"),
        )
        self.response_shape.pack(side="right", anchor="s")
        tk.Label(
            inner,
            text="mean over 12 rows / 65 points",
            bg=CARD,
            fg=MUTED,
            anchor="w",
            font=(self._font_family, 7),
        ).pack(fill="x", pady=(0, 5))
        self.current_canvas = CurrentCanvas(inner, height=340)
        self.current_canvas.pack(fill="both", expand=True, pady=(0, 4))
        self.hardware_meta = tk.Label(
            inner,
            text="Waiting for measured / replayed currents",
            bg=CARD,
            fg=MUTED,
            anchor="w",
            justify="left",
            wraplength=280,
            font=(self._font_family, 7),
        )
        self.hardware_meta.pack(fill="x")
        tk.Label(
            inner,
            text="Hardware accepts only drive frames; it knows no image or label.",
            bg="#f8fafc",
            fg="#475569",
            anchor="w",
            justify="left",
            wraplength=280,
            padx=8,
            pady=6,
            font=(self._font_family, 8),
        ).pack(fill="x", pady=(8, 0))

    def _build_decision_card(self, parent: tk.Misc, column: int) -> None:
        card, inner = self._card(parent)
        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(6, 0 if column == 2 else 6),
        )
        self._stage_header(
            inner, "03", "Decision", "DECISION | CURRENT -> CLASS", "#7c3aed"
        )
        self.prediction_label = tk.Label(
            inner,
            text="Waiting for inference",
            bg=CARD,
            fg=INK,
            anchor="w",
            justify="left",
            wraplength=280,
            font=(self._font_family, 17, "bold"),
        )
        self.prediction_label.pack(fill="x", pady=(0, 1))
        confidence_row = tk.Frame(inner, bg=CARD)
        confidence_row.pack(fill="x", pady=(0, 2))
        self.confidence_label = tk.Label(
            confidence_row,
            text="-",
            bg=CARD,
            fg="#7c3aed",
            anchor="w",
            font=(self._mono_family, 22, "bold"),
        )
        self.confidence_label.pack(side="left")
        tk.Label(
            confidence_row,
            text="top-class probability",
            bg=CARD,
            fg=MUTED,
            anchor="s",
            font=(self._font_family, 8),
        ).pack(side="left", padx=(6, 0), pady=(0, 4))
        self.predicted_raw = tk.Label(
            inner,
            text="",
            bg=CARD,
            fg=MUTED,
            anchor="w",
            justify="left",
            wraplength=280,
            font=(self._mono_family, 7),
        )
        self.predicted_raw.pack(fill="x", pady=(0, 10))
        self._section_label(inner, "TOP-5 PROBABILITY").pack(fill="x", pady=(0, 2))
        self.probability_canvas = ProbabilityCanvas(inner, height=230)
        self.probability_canvas.pack(fill="both", expand=True)
        self.decision_meta = tk.Label(
            inner,
            text="Waiting for current input",
            bg=CARD,
            fg=MUTED,
            anchor="w",
            justify="left",
            wraplength=280,
            font=(self._font_family, 7),
        )
        self.decision_meta.pack(fill="x", pady=(5, 0))
        tk.Label(
            inner,
            text="Decision consumes only HardwareResponse.currents; no image or label.",
            bg="#f8fafc",
            fg="#475569",
            anchor="w",
            justify="left",
            wraplength=280,
            padx=8,
            pady=6,
            font=(self._font_family, 8),
        ).pack(fill="x", pady=(8, 0))

    # --------------------------------------------------------------- data state
    def _load_dataset(self) -> None:
        if not self.paths.responses.is_file() or not self.paths.inputs.is_file():
            self.dataset_error = "Recorded dataset was not found under artifacts_hardware300/sources"
            self.dataset_value.configure(text="Recorded dataset not installed", fg=RED)
            self.sidebar_status.configure(text=self.dataset_error, fg=RED)
            self.run_button.state(["disabled"])
            self.status_label.configure(text="*  Recorded dataset not installed", fg=RED)
            return
        try:
            self.dataset = RecordedHardwareDataset(
                self.paths.responses, self.paths.inputs
            )
            self.recorded = RecordedHardwareBackend(self.dataset)
        except Exception as error:
            self.dataset_error = str(error)
            self.dataset_value.configure(text="Failed to read recorded dataset", fg=RED)
            self.sidebar_status.configure(text=self.dataset_error, fg=RED)
            self.run_button.state(["disabled"])
            self.status_label.configure(text=f"*  {self.dataset_error}", fg=RED)
            return
        summaries = self.dataset.sample_summaries()
        values = [
            f"{row['index']:03d}  {class_display(row['class_name'])}"
            for row in summaries
        ]
        self.sample_combo.configure(values=values)
        self.dataset_value.configure(
            text=f"{len(summaries)} measured samples"
        )
        model_text = self.paths.model.name if self.paths.model else "model not found"
        self.sidebar_status.configure(
            text=f"Dataset ready | {model_text}",
            fg=GREEN if self.paths.model else RED,
        )
        self.status_label.configure(
            text="*  Dataset loaded. Click Run end-to-end pipeline.",
            fg=GREEN,
        )

    def _summary(self, index: int) -> dict[str, Any] | None:
        if self.dataset is None:
            return None
        summaries = self.dataset.sample_summaries()
        if not 0 <= index < len(summaries):
            return None
        return summaries[index]

    def _on_sample_change(self, _event: tk.Event | None = None) -> None:
        if self.dataset is None:
            return
        index = self.sample_combo.current()
        if index < 0:
            return
        summary = self._summary(index)
        if summary is None:
            return
        self._clear_outputs()
        self.sample_title.configure(
            text=f"Sample {index:03d} | source #{summary['source_sample_index']}"
        )
        self.gold_label.configure(text=class_display(summary["class_name"]))
        sample_id = str(summary["sample_id"])
        self.sample_detail.configure(
            text=f"{sample_id}\nclass_index={summary['label_index']}"
        )
        image_path = _resolve_sample_image(sample_id)
        if image_path is not None:
            try:
                with Image.open(image_path) as image:
                    self.image_pane.set_image(image.convert("RGB"))
                self.image_caption.configure(
                    text=f"Input image | {image_path.name}\n(preview only; drive is replayed from inputs.csv)"
                )
            except Exception as error:
                self.image_pane.set_image(None)
                self.image_caption.configure(text=f"Image read failed: {error}")
        else:
            drive = self.dataset.drive(index)
            self.image_pane.set_image(_drive_mosaic(drive), mosaic=True)
            self.image_caption.configure(
                text="8x8 patch-mean reconstruction (original image not installed)"
            )
        self.status_label.configure(
            text="*  Sample selected. Click Run end-to-end pipeline.",
            fg=MUTED,
        )
        self.timing_label.configure(text="processing -> hardware -> decision")

    def _step_sample(self, delta: int) -> None:
        if self.dataset is None or self._busy:
            return
        count = len(self.dataset.metadata)
        current = max(self.sample_combo.current(), 0)
        self.sample_combo.current((current + delta) % count)
        self._on_sample_change()

    def _clear_outputs(self) -> None:
        self.current_result = None
        self.current_index = None
        self.drive_canvas.set_frames(None)
        self.current_canvas.set_currents(None)
        self.probability_canvas.set_rows([])
        self.prediction_label.configure(text="Waiting for inference", fg=INK)
        self.confidence_label.configure(text="-")
        self.predicted_raw.configure(text="")
        self.current_metric.configure(text="-")
        self.hardware_meta.configure(text="Waiting for measured / replayed currents")
        self.drive_meta.configure(text="64 frames x 9 channels | snake scan | recorded input")
        self.decision_meta.configure(text="Waiting for current input")
        self.export_button.state(["disabled"])

    # ------------------------------------------------------------------- worker
    def _compute(self, index: int) -> tuple[Any, dict[str, Any], dict[str, float]]:
        if self.dataset is None or self.recorded is None:
            raise RuntimeError(self.dataset_error or "recorded dataset not installed")
        summary = self._summary(index)
        if summary is None:
            raise ValueError(f"sample index out of range: {index}")
        started = time.perf_counter()
        drive = self.dataset.drive(index)
        loaded = time.perf_counter()
        pipeline = HardwareRCPipeline(self.processor, self.recorded, self.decision)
        result = pipeline.run_drive(drive)
        finished = time.perf_counter()
        timings = {
            "recorded_drive_ms": round((loaded - started) * 1000, 3),
            "hardware_ms": float(result.timings_ms.get("hardware_ms", 0.0)),
            "decision_ms": float(result.timings_ms.get("decision_ms", 0.0)),
            "total_ms": round((finished - started) * 1000, 3),
        }
        return result, summary, timings

    def _worker(self, index: int) -> None:
        try:
            result, summary, timings = self._compute(index)
            self._queue.put(("ok", result, summary, timings))
        except Exception as error:
            self._queue.put(("error", str(error), traceback.format_exc()))

    def _poll_worker(self) -> None:
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            self.root.after(60, self._poll_worker)
            return
        self._set_busy(False)
        if item[0] == "ok":
            _, result, summary, timings = item
            self._apply_result(result, summary, timings)
        else:
            _, message, trace = item
            self.status_label.configure(text=f"*  Run failed: {message}", fg=RED)
            self.sidebar_status.configure(text=message, fg=RED)
            print(trace, flush=True)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self.run_button.state(["disabled"])
            self.progress.grid()
            self.progress.start(12)
            self.status_label.configure(text="*  Replaying measured response and running readout...", fg=AMBER)
            self.sidebar_status.configure(text="Loading model and scoring currents...", fg=AMBER)
        else:
            self.run_button.state(["!disabled"])
            self.progress.stop()
            self.progress.grid_remove()

    def run_selected(self) -> None:
        if self._busy or self.dataset is None or self.dataset_error:
            return
        index = self.sample_combo.current()
        if index < 0:
            return
        self._set_busy(True)
        threading.Thread(target=self._worker, args=(index,), daemon=True).start()
        self.root.after(60, self._poll_worker)

    def run_sync(self, index: int) -> tuple[Any, dict[str, Any], dict[str, float]]:
        """Synchronous entry point used by the screenshot/CI path."""
        result, summary, timings = self._compute(index)
        self.sample_combo.current(index)
        self._on_sample_change()
        self._apply_result(result, summary, timings)
        self.root.update_idletasks()
        self.root.update()
        return result, summary, timings

    def _apply_result(
        self, result: Any, summary: dict[str, Any], timings: dict[str, float]
    ) -> None:
        self.current_result = result
        self.current_index = int(summary["index"])
        self.drive_canvas.set_frames(result.drive.frames)
        self.current_canvas.set_currents(result.response.currents)
        current_uA = float(np.mean(np.abs(result.response.currents)) * 1e6)
        self.current_metric.configure(text=f"{current_uA:.3f}")
        self.hardware_meta.configure(
            text=(
                f"Backend: {result.response.backend}\n"
                "measured=True | amperes | display = mean over 12 rows"
            )
        )
        drive_mean = float(np.mean(result.drive.frames))
        active_events = int(np.count_nonzero(result.drive.frames[:, 3:]))
        self.drive_meta.configure(
            text=(
                "64 frames x 9 channels | snake scan | recorded input\n"
                f"mean {drive_mean:.4f} | non-zero event values {active_events}"
            )
        )
        decision = result.decision
        zh = class_display(decision.predicted_class)
        self.prediction_label.configure(text=zh, fg=INK)
        self.confidence_label.configure(
            text=f"{decision.probabilities[decision.predicted_index] * 100:.1f}%"
        )
        self.predicted_raw.configure(text=decision.predicted_class)
        rows = [
            (
                str(row["class_name"]),
                float(row["probability"]),
                int(row["index"]) == decision.predicted_index,
            )
            for row in decision.top_k[:5]
        ]
        self.probability_canvas.set_rows(rows)
        self.decision_meta.configure(
            text=(
                "HardwareResponse.currents [9, 12, 65] -> predict_proba\n"
                f"model: {decision.model}"
            )
        )
        self.status_label.configure(text="*  Complete | local measured-current replay", fg=GREEN)
        self.sidebar_status.configure(
            text=f"Done | sample {summary['index']:03d} | {class_display(summary['class_name'])}",
            fg=GREEN,
        )
        self.timing_label.configure(
            text=(
                f"recorded {timings['recorded_drive_ms']:.2f} ms  |  "
                f"hardware {timings['hardware_ms']:.2f} ms  |  "
                f"decision {timings['decision_ms']:.2f} ms  |  "
                f"total {timings['total_ms']:.2f} ms"
            )
        )
        self.export_button.state(["!disabled"])

    # ------------------------------------------------------------------ export
    def export_result(self) -> None:
        if self.current_result is None or self.current_index is None:
            return
        summary = self._summary(self.current_index)
        default_name = f"hardware_rc_sample_{self.current_index:03d}.json"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export current result",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        payload = {
            "sample": summary,
            "pipeline": self.current_result.to_dict(include_currents=False),
            "note": (
                "Recorded-drive and measured-current replay. The original CSV "
                "label is shown for comparison only and is not an input to any layer."
            ),
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.status_label.configure(text=f"*  Exported {Path(path).name}", fg=GREEN)


# --------------------------------------------------------------------- entrypoint
def _parser() -> argparse.ArgumentParser:
    responses, inputs = _dataset_paths()
    parser = argparse.ArgumentParser(
        description="Simple Tkinter demo for the layered hardware-RC pipeline."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=_default_model(),
        help="trusted local joblib readout model (default: auto-detect)",
    )
    parser.add_argument("--responses", type=Path, default=responses)
    parser.add_argument("--inputs", type=Path, default=inputs)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=None,
        help=(
            "render the real window offscreen and save demo screenshots, then exit"
        ),
    )
    parser.add_argument("--geometry", default="1380x860")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Tkinter only; Xvfb/ImageGrab captures this real Tk render when requested.
    root = tk.Tk()
    root.title("Hardware RC | Layered Decision Console")
    root.geometry(args.geometry)
    root.minsize(1200, 760)
    app = DesktopApp(
        root,
        GuiPaths(args.model, args.responses, args.inputs),
        initial_sample=args.sample_index,
        auto_run=args.capture_dir is None,
    )
    if args.capture_dir is not None:
        if app.dataset is None:
            root.destroy()
            print(f"cannot capture: {app.dataset_error}", flush=True)
            return 2
        capture_dir = args.capture_dir
        capture_dir.mkdir(parents=True, exist_ok=True)
        root.update_idletasks()
        root.update()
        samples = [(0, "gui-main.png"), (152, "gui-uncertain.png")]
        for index, filename in samples:
            if index >= len(app.dataset.metadata):
                continue
            app.run_sync(index)
            root.update_idletasks()
            root.update()
            time.sleep(0.15)
            target = capture_dir / filename
            if not _write_png_screenshot(root, target):
                print(f"failed to capture {target}", flush=True)
                root.destroy()
                return 3
            print(f"captured {target}", flush=True)
        root.destroy()
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
