#!/usr/bin/env python3
import csv
import fnmatch
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


ROOT = Path(__file__).resolve().parent
TRAINING_CSV = ROOT / "training.csv"
RUNS_DIR = ROOT / "runs_cubi"
IMAGES_DIR = ROOT / "plots"


class TrainingPlotHelper:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Training Plot Helper")
        self.root.geometry("1100x720")

        self.training_rows = self._load_training_rows()
        self.row_by_display: dict[str, dict] = {}
        self.loaded_data: dict[str, pd.DataFrame] = {}
        self.run_labels: dict[str, str] = {}
        self.current_scalar_columns: list[str] = []
        self._draw_after_id: str | None = None
        self._load_after_id: str | None = None
        self.ax_lr = None

        self._build_ui()

    def _load_training_rows(self) -> list[dict]:
        if not TRAINING_CSV.exists():
            raise FileNotFoundError(f"Missing file: {TRAINING_CSV}")

        with TRAINING_CSV.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        rows = [r for r in rows if r.get("name") and r.get("folder_name")]
        rows.sort(key=lambda r: r["name"])
        return rows

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        left_outer = ttk.Frame(container)
        left_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left_outer.columnconfigure(0, weight=1)
        left_outer.rowconfigure(0, weight=1)

        left_canvas = tk.Canvas(left_outer, highlightthickness=0, width=430)
        left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scroll = ttk.Scrollbar(
            left_outer, orient="vertical", command=left_canvas.yview
        )
        left_scroll.grid(row=0, column=1, sticky="ns")
        left_canvas.configure(yscrollcommand=left_scroll.set)

        left = ttk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.grid_columnconfigure(0, weight=1)

        left.bind(
            "<Configure>",
            lambda _e: left_canvas.configure(scrollregion=left_canvas.bbox("all")),
        )
        left_canvas.bind(
            "<Configure>",
            lambda e: left_canvas.itemconfigure(left_window, width=e.width),
        )

        filter_frame = ttk.LabelFrame(left, text="Filter filenames")
        filter_frame.pack(fill="x", pady=(0, 8))
        self.filter_var = tk.StringVar(value="")
        self.filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var)
        self.filter_entry.pack(fill="x", padx=4, pady=6)
        self.filter_entry.bind("<KeyRelease>", lambda _e: self._on_filter_change())

        ttk.Label(left, text="Trainings (column: name)").pack(anchor="w")
        names_frame = ttk.Frame(left)
        names_frame.pack(fill="x", expand=False, pady=(4, 8))
        self.name_list = tk.Listbox(
            names_frame,
            selectmode=tk.MULTIPLE,
            height=24,
            exportselection=False,
            width=48,
            font=("DejaVu Sans Mono", 10),
        )
        self.name_list.pack(side="left", fill="x", expand=True)
        names_scroll = ttk.Scrollbar(
            names_frame, orient="vertical", command=self.name_list.yview
        )
        names_scroll.pack(side="right", fill="y")
        self.name_list.configure(yscrollcommand=names_scroll.set)
        self.name_list.bind(
            "<<ListboxSelect>>", lambda _e: self._schedule_load_selected()
        )
        self._refresh_name_list()

        scalar_frame = ttk.LabelFrame(left, text="Variables (up to 2)")
        scalar_frame.pack(fill="x", pady=(4, 8))
        ttk.Label(scalar_frame, text="Var 1").grid(
            row=0, column=0, padx=4, pady=(6, 4), sticky="w"
        )
        ttk.Label(scalar_frame, text="Var 2").grid(
            row=1, column=0, padx=4, pady=(4, 6), sticky="w"
        )
        self.scalar_var_1 = tk.StringVar()
        self.scalar_var_2 = tk.StringVar()
        self.scalar_combo_1 = ttk.Combobox(
            scalar_frame, textvariable=self.scalar_var_1, state="readonly", width=30
        )
        self.scalar_combo_2 = ttk.Combobox(
            scalar_frame, textvariable=self.scalar_var_2, state="readonly", width=30
        )
        self.scalar_combo_1.grid(row=0, column=1, padx=4, pady=(6, 4), sticky="ew")
        self.scalar_combo_2.grid(row=1, column=1, padx=4, pady=(4, 6), sticky="ew")
        scalar_frame.columnconfigure(1, weight=1)
        self.scalar_combo_1.bind(
            "<<ComboboxSelected>>", lambda _e: self._schedule_draw()
        )
        self.scalar_combo_2.bind(
            "<<ComboboxSelected>>", lambda _e: self._schedule_draw()
        )

        title_frame = ttk.LabelFrame(left, text="Title")
        title_frame.pack(fill="x", pady=(0, 8))
        self.title_var = tk.StringVar(value="")
        self.title_entry = ttk.Entry(title_frame, textvariable=self.title_var)
        self.title_entry.pack(fill="x", padx=4, pady=6)
        self.title_entry.bind("<KeyRelease>", lambda _e: self._schedule_draw())
        self.title_entry.bind("<FocusOut>", lambda _e: self._schedule_draw())

        range_frame = ttk.LabelFrame(left, text="Step range")
        range_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(range_frame, text="From").grid(
            row=0, column=0, padx=4, pady=6, sticky="w"
        )
        ttk.Label(range_frame, text="To").grid(
            row=1, column=0, padx=4, pady=6, sticky="w"
        )
        self.x_from_var = tk.IntVar(value=0)
        self.x_to_var = tk.IntVar(value=400)
        self.x_from_spin = tk.Spinbox(
            range_frame,
            from_=0,
            to=200000,
            textvariable=self.x_from_var,
            width=10,
            command=self._schedule_draw,
        )
        self.x_to_spin = tk.Spinbox(
            range_frame,
            from_=0,
            to=200000,
            textvariable=self.x_to_var,
            width=10,
            command=self._schedule_draw,
        )
        self.x_from_spin.grid(row=0, column=1, padx=4, pady=6, sticky="w")
        self.x_to_spin.grid(row=1, column=1, padx=4, pady=6, sticky="w")
        self.x_from_spin.bind("<KeyRelease>", lambda _e: self._schedule_draw())
        self.x_to_spin.bind("<KeyRelease>", lambda _e: self._schedule_draw())
        self.x_from_spin.bind("<FocusOut>", lambda _e: self._schedule_draw())
        self.x_to_spin.bind("<FocusOut>", lambda _e: self._schedule_draw())

        smooth_frame = ttk.LabelFrame(left, text="Smoothing")
        smooth_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(smooth_frame, text="Weight (TensorBoard)").grid(
            row=0, column=0, padx=4, pady=6, sticky="w"
        )
        self.smooth_var = tk.DoubleVar(value=0.6)
        self.smooth_scale = ttk.Scale(
            smooth_frame,
            from_=0.0,
            to=0.99,
            variable=self.smooth_var,
            orient="horizontal",
            command=lambda _v: self._schedule_draw(),
        )
        self.smooth_scale.grid(row=1, column=0, padx=4, pady=(0, 6), sticky="ew")
        smooth_frame.columnconfigure(0, weight=1)
        self.smooth_label_var = tk.StringVar(value="0.60")
        ttk.Label(smooth_frame, textvariable=self.smooth_label_var).grid(
            row=1, column=1, padx=4, pady=(0, 6), sticky="e"
        )
        self.show_original_var = tk.BooleanVar(value=False)
        self.show_original_check = ttk.Checkbutton(
            smooth_frame,
            text="Show original curve",
            variable=self.show_original_var,
            command=self._schedule_draw,
        )
        self.show_original_check.grid(row=2, column=0, padx=4, pady=(0, 4), sticky="w")
        opacity_row = ttk.Frame(smooth_frame)
        opacity_row.grid(
            row=3, column=0, columnspan=2, padx=4, pady=(0, 6), sticky="ew"
        )
        opacity_row.columnconfigure(1, weight=1)
        ttk.Label(opacity_row, text="Original opacity").grid(
            row=0, column=0, sticky="w"
        )
        self.orig_alpha_var = tk.DoubleVar(value=0.30)
        self.orig_alpha_scale = ttk.Scale(
            opacity_row,
            from_=0.05,
            to=1.0,
            variable=self.orig_alpha_var,
            orient="horizontal",
            command=lambda _v: self._schedule_draw(),
        )
        self.orig_alpha_scale.grid(row=0, column=1, padx=(8, 6), sticky="ew")
        self.orig_alpha_label_var = tk.StringVar(value="0.30")
        ttk.Label(opacity_row, textvariable=self.orig_alpha_label_var).grid(
            row=0, column=2, sticky="e"
        )

        lr_frame = ttk.LabelFrame(left, text="Learning rate overlay (training/lr)")
        lr_frame.pack(fill="x", pady=(0, 8))
        self.show_lr_var = tk.BooleanVar(value=False)
        self.show_lr_check = ttk.Checkbutton(
            lr_frame,
            text="Show training/lr",
            variable=self.show_lr_var,
            command=self._schedule_draw,
        )
        self.show_lr_check.grid(row=0, column=0, padx=4, pady=(6, 4), sticky="w")
        ttk.Label(lr_frame, text="Scale").grid(row=1, column=0, padx=4, pady=(2, 6), sticky="w")
        self.lr_scale_var = tk.StringVar(value="real")
        self.lr_scale_combo = ttk.Combobox(
            lr_frame,
            textvariable=self.lr_scale_var,
            values=["real", "log"],
            state="readonly",
            width=10,
        )
        self.lr_scale_combo.grid(row=1, column=1, padx=4, pady=(2, 6), sticky="w")
        self.lr_scale_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_draw())

        labels_frame = ttk.LabelFrame(left, text="Legend labels (name = label)")
        labels_frame.pack(fill="x", pady=(0, 8))
        labels_edit_frame = ttk.Frame(left)
        labels_edit_frame.pack(fill="x", pady=(0, 8))
        self.labels_text = tk.Text(
            labels_edit_frame, height=6, wrap="none", font=("DejaVu Sans Mono", 9)
        )
        self.labels_text.pack(side="left", fill="x", expand=True)
        labels_scroll = ttk.Scrollbar(
            labels_edit_frame, orient="vertical", command=self.labels_text.yview
        )
        labels_scroll.pack(side="right", fill="y")
        self.labels_text.configure(yscrollcommand=labels_scroll.set)
        self.labels_text.bind("<KeyRelease>", lambda _e: self._schedule_draw())
        self.labels_text.bind("<FocusOut>", lambda _e: self._schedule_draw())

        save_frame = ttk.LabelFrame(left, text="Save image")
        save_frame.pack(fill="x", pady=(4, 8))
        ttk.Label(save_frame, text="Image name").grid(
            row=0, column=0, padx=4, pady=6, sticky="w"
        )
        self.image_name_var = tk.StringVar(value="plot.png")
        self.image_name_entry = ttk.Entry(
            save_frame, textvariable=self.image_name_var, width=24
        )
        self.image_name_entry.grid(row=1, column=0, padx=4, pady=(0, 8), sticky="ew")
        ttk.Button(save_frame, text="Save to plots/", command=self.save_plot).grid(
            row=2, column=0, padx=4, pady=(0, 8), sticky="ew"
        )

        self.status_var = tk.StringVar(value="Select trainings to load curves.")
        ttk.Label(
            left, textvariable=self.status_var, foreground="#555555", wraplength=280
        ).pack(anchor="w", pady=(8, 0))

        right = ttk.Frame(container)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(8.6, 6.2), dpi=100)
        self.ax.set_xlabel("step", fontsize=13)
        self.ax.set_ylabel("value", fontsize=13)
        self.ax.set_title("Training scalar plot", fontsize=15)
        self.ax.grid(True, alpha=0.25)
        self.ax.tick_params(axis="both", labelsize=11)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.draw_idle()

    def _selected_rows(self) -> list[dict]:
        displays = [self.name_list.get(i) for i in self.name_list.curselection()]
        return [self.row_by_display[d] for d in displays if d in self.row_by_display]

    def _scalars_path_from_row(self, row: dict) -> Path:
        return RUNS_DIR / row["folder_name"] / "scalars.csv"

    @staticmethod
    def _tensorboard_smooth(values: pd.Series, weight: float) -> pd.Series:
        """TensorBoard-style debiased exponential smoothing."""
        weight = min(max(float(weight), 0.0), 0.999)
        if weight <= 0.0:
            return values

        last = 0.0
        debias_weight = 1.0
        out = []
        for v in values.astype(float).tolist():
            last = last * weight + (1.0 - weight) * v
            debias_weight *= weight
            out.append(last / (1.0 - debias_weight))
        return pd.Series(out, index=values.index, dtype=float)

    def _schedule_draw(self) -> None:
        if self._draw_after_id is not None:
            self.root.after_cancel(self._draw_after_id)
        self._draw_after_id = self.root.after(120, lambda: self.draw_plot(silent=True))

    def _schedule_load_selected(self) -> None:
        if self._load_after_id is not None:
            self.root.after_cancel(self._load_after_id)
        self._load_after_id = self.root.after(
            120, lambda: self.load_selected(silent_no_selection=True)
        )

    def _refresh_name_list(self) -> None:
        selected_folders = {
            self.row_by_display[self.name_list.get(i)]["folder_name"]
            for i in self.name_list.curselection()
            if self.name_list.get(i) in self.row_by_display
        }
        needle = self.filter_var.get().strip().lower()
        if needle and "*" not in needle:
            needle = f"*{needle}*"

        self.name_list.delete(0, tk.END)
        self.row_by_display.clear()
        for row in self.training_rows:
            display = f"{row['name']} [{row['folder_name']}]"
            display_l = display.lower()
            name_l = row["name"].lower()
            folder_l = row["folder_name"].lower()
            if needle and not (
                fnmatch.fnmatch(display_l, needle)
                or fnmatch.fnmatch(name_l, needle)
                or fnmatch.fnmatch(folder_l, needle)
            ):
                continue
            self.row_by_display[display] = row
            self.name_list.insert(tk.END, display)
            if row["folder_name"] in selected_folders:
                self.name_list.selection_set(tk.END)

    def _on_filter_change(self) -> None:
        self._refresh_name_list()
        self._schedule_load_selected()

    def _parse_custom_labels(self) -> dict[str, str]:
        out: dict[str, str] = {}
        raw = self.labels_text.get("1.0", tk.END)
        for line in raw.splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if key and val:
                out[key] = val
        return out

    def _selected_scalars(self) -> list[str]:
        s1 = self.scalar_var_1.get().strip()
        s2 = self.scalar_var_2.get().strip()
        selected: list[str] = []
        if s1:
            selected.append(s1)
        if s2 and s2 != s1:
            selected.append(s2)
        return selected

    @staticmethod
    def _display_scalar_name(name: str) -> str:
        return name.replace("/", "-")

    def load_selected(self, silent_no_selection: bool = False) -> None:
        self._load_after_id = None
        rows = self._selected_rows()
        if not rows:
            self.loaded_data = {}
            self.run_labels = {}
            self.current_scalar_columns = []
            self.scalar_combo_1["values"] = ()
            self.scalar_combo_2["values"] = ("",)
            self.scalar_var_1.set("")
            self.scalar_var_2.set("")
            self.ax.clear()
            self.ax.set_xlabel("step", fontsize=13)
            self.ax.set_ylabel("value", fontsize=13)
            self.ax.set_title("Training scalar plot", fontsize=15)
            self.ax.grid(True, alpha=0.25)
            self.ax.tick_params(axis="both", labelsize=11)
            self.canvas.draw_idle()
            self.status_var.set("Select at least one training.")
            if not silent_no_selection:
                messagebox.showwarning(
                    "No selection", "Select at least one training name."
                )
            return

        loaded: dict[str, pd.DataFrame] = {}
        labels: dict[str, str] = {}
        missing = []
        for row in rows:
            name = row["name"]
            folder = row["folder_name"]
            path = self._scalars_path_from_row(row)
            if not path.exists():
                missing.append(f"{name} [{folder}] -> {path}")
                continue
            df = pd.read_csv(path)
            if "step" not in df.columns:
                missing.append(f"{name} [{folder}] -> missing step column")
                continue
            loaded[folder] = df
            labels[folder] = name

        if not loaded:
            self.status_var.set("Load skipped: no valid scalar files for current selection.")
            return

        common = set.intersection(*(set(df.columns) for df in loaded.values()))
        common.discard("step")
        scalar_cols = sorted(
            c
            for c in common
            if pd.api.types.is_numeric_dtype(loaded[next(iter(loaded))][c])
        )
        if not scalar_cols:
            self.status_var.set("Load skipped: no common numeric scalar columns.")
            return

        self.loaded_data = loaded
        self.run_labels = labels
        self.current_scalar_columns = scalar_cols
        selectable_scalar_cols = [c for c in scalar_cols if c != "training/lr"]
        self.scalar_combo_1["values"] = selectable_scalar_cols
        self.scalar_combo_2["values"] = ["", *selectable_scalar_cols]
        if selectable_scalar_cols:
            if self.scalar_var_1.get() not in selectable_scalar_cols:
                self.scalar_var_1.set(selectable_scalar_cols[0])
        else:
            self.scalar_var_1.set("")
        if self.scalar_var_2.get() not in selectable_scalar_cols:
            self.scalar_var_2.set("")

        note = f"Loaded {len(loaded)} runs."
        if missing:
            note += f" Skipped {len(missing)} run(s)."
        self.status_var.set(note)

        # Prefill editable legend labels keyed by run "name".
        existing_custom = self._parse_custom_labels()
        selected_names = sorted(set(labels.values()))
        self.labels_text.delete("1.0", tk.END)
        lines = [f"{n} = {existing_custom.get(n, n)}" for n in selected_names]
        self.labels_text.insert("1.0", "\n".join(lines))

        self.draw_plot(silent=True)

    def draw_plot(self, silent: bool = False) -> None:
        self._draw_after_id = None
        if self.ax_lr is not None:
            self.ax_lr.remove()
            self.ax_lr = None
        if not self.loaded_data:
            if not silent:
                messagebox.showwarning(
                    "Nothing loaded", "Select at least one training first."
                )
            return

        scalars = self._selected_scalars()
        show_lr = bool(self.show_lr_var.get())
        if not scalars and not show_lr:
            if not silent:
                messagebox.showwarning(
                    "No variable", "Choose at least one variable or enable training/lr."
                )
            return

        x_min = int(self.x_from_var.get())
        x_max = int(self.x_to_var.get())
        if x_min > x_max:
            x_min, x_max = x_max, x_min

        smooth_weight = min(max(float(self.smooth_var.get()), 0.0), 0.99)
        self.smooth_label_var.set(f"{smooth_weight:.2f}")
        show_original = bool(self.show_original_var.get())
        original_alpha = min(max(float(self.orig_alpha_var.get()), 0.05), 1.0)
        self.orig_alpha_label_var.set(f"{original_alpha:.2f}")
        self.ax.clear()
        self.ax.set_xlabel("step", fontsize=13)
        display_scalars = [self._display_scalar_name(s) for s in scalars]
        scalar_label = " / ".join(display_scalars) if display_scalars else "value"
        ylabel = (
            f"{scalar_label} (s={smooth_weight:.2f})"
            if smooth_weight > 0.0 and display_scalars
            else scalar_label
        )
        self.ax.set_ylabel(ylabel, fontsize=13)
        custom_title = self.title_var.get().strip()
        default_title = " / ".join(display_scalars) if display_scalars else self._display_scalar_name("training/lr")
        self.ax.set_title(custom_title if custom_title else default_title, fontsize=15)
        self.ax.grid(True, alpha=0.25)
        self.ax.tick_params(axis="both", labelsize=11)

        custom_labels = self._parse_custom_labels()
        plotted = 0
        multi_var = len(scalars) > 1
        legend_handles = []
        legend_labels = []
        for scalar in scalars:
            display_scalar = self._display_scalar_name(scalar)
            for run_key, df in self.loaded_data.items():
                if scalar not in df.columns:
                    continue
                sub = df[(df["step"] >= x_min) & (df["step"] <= x_max)]
                if sub.empty:
                    continue
                y_raw = sub[scalar].astype(float)
                y = self._tensorboard_smooth(y_raw, smooth_weight)
                base_name = self.run_labels.get(run_key, run_key)
                legend_label = custom_labels.get(base_name, base_name)
                if multi_var:
                    legend_label = f"{legend_label} | {display_scalar}"
                (line,) = self.ax.plot(
                    sub["step"], y, label=legend_label, linewidth=1.8, alpha=1.0
                )
                legend_handles.append(line)
                legend_labels.append(legend_label)
                if show_original and smooth_weight > 0.0:
                    self.ax.plot(
                        sub["step"],
                        y_raw,
                        color=line.get_color(),
                        linewidth=1.3,
                        alpha=original_alpha,
                    )
                plotted += 1

        lr_col = "training/lr"
        if show_lr:
            self.ax_lr = self.ax.twinx()
            ax_lr = self.ax_lr
            ax_lr.set_ylabel(self._display_scalar_name(lr_col), fontsize=13)
            ax_lr.tick_params(axis="y", labelsize=11)
            lr_scale = self.lr_scale_var.get().strip().lower()
            ax_lr.set_yscale("log" if lr_scale == "log" else "linear")
            lr_colors = plt.cm.Dark2.colors
            lr_idx = 0
            for run_key, df in self.loaded_data.items():
                if lr_col not in df.columns:
                    continue
                sub = df[(df["step"] >= x_min) & (df["step"] <= x_max)]
                if sub.empty:
                    continue
                y_lr = sub[lr_col].astype(float)
                if lr_scale == "log":
                    sub_plot = sub[y_lr > 0]
                    y_lr = y_lr[y_lr > 0]
                    if sub_plot.empty:
                        continue
                else:
                    sub_plot = sub
                base_name = self.run_labels.get(run_key, run_key)
                lr_label = f"{custom_labels.get(base_name, base_name)} | {self._display_scalar_name(lr_col)}"
                lr_color = lr_colors[lr_idx % len(lr_colors)]
                (line_lr,) = ax_lr.plot(
                    sub_plot["step"],
                    y_lr,
                    label=lr_label,
                    linewidth=1.6,
                    alpha=0.5,
                    color=lr_color,
                )
                lr_idx += 1
                legend_handles.append(line_lr)
                legend_labels.append(lr_label)
                plotted += 1

        if plotted == 0:
            self.canvas.draw_idle()
            if not silent:
                messagebox.showwarning("No data", "No points found in that step range.")
            return

        self.ax.legend(legend_handles, legend_labels, loc="best", fontsize=10)
        self.canvas.draw_idle()
        self.status_var.set(
            f"Drawn {plotted} curve(s), x range [{x_min}, {x_max}], TB smoothing={smooth_weight:.2f}, "
            f"show_original={'on' if show_original else 'off'}."
        )

    def save_plot(self) -> None:
        filename = self.image_name_var.get().strip()
        if not filename:
            messagebox.showwarning(
                "Missing name", "Enter an image filename (example: plot.png)."
            )
            return
        if "." not in filename:
            filename += ".png"

        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = IMAGES_DIR / filename
        self.fig.savefig(out_path, bbox_inches="tight", dpi=140)
        self.status_var.set(f"Saved: {out_path}")
        messagebox.showinfo("Saved", f"Image saved to:\n{out_path}")


def main() -> None:
    root = tk.Tk()
    app = TrainingPlotHelper(root)
    app.root.mainloop()


if __name__ == "__main__":
    main()
