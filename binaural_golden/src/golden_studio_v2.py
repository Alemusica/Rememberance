"""
Golden Studio v2 - Main Application

Swiss Typography Design with φ proportions.
Program-based workflow for creating healing sound journeys.
"""

import tkinter as tk
from tkinter import ttk
import json
from pathlib import Path

# UI Theme
from ui.golden_theme import (
    setup_golden_app, Colors, Spacing, FontSize, Typography, Radius,
    PHI, create_rounded_rectangle, GoldenCard
)

# Core
try:
    from core.audio_engine import AudioEngine
    from core.golden_math import PHI, PHI_CONJUGATE
except ImportError:
    from golden_constants import PHI, PHI_CONJUGATE
    AudioEngine = None

# Programs
try:
    from programs.program import Program
    from programs.step import Step, FrequencyConfig, PositionConfig, FadeCurve
except ImportError:
    Program = None
    Step = None


class GoldenStudioApp:
    """
    Main application window with φ-based layout.
    
    Layout (Golden Grid):
    ┌─────────────────────────────────────────────────────────────┐
    │  HEADER (φ⁻³ height)                                        │
    ├──────────────────┬──────────────────────────────────────────┤
    │                  │                                          │
    │   SIDEBAR        │         MAIN CONTENT                     │
    │   (φ⁻¹ width)    │         (1 - φ⁻¹ width)                  │
    │                  │                                          │
    │   • Programs     │         Tab content based on mode:       │
    │   • Steps        │         - Program Builder                │
    │   • Transport    │         - Live Control                   │
    │                  │         - Visualization                  │
    │                  │                                          │
    ├──────────────────┴──────────────────────────────────────────┤
    │  FOOTER / STATUS (φ⁻³ height)                               │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.style = setup_golden_app(self.root, "φ Golden Studio")
        
        # Larger window for program builder
        self._setup_geometry()
        
        # Audio engine
        self.audio = AudioEngine() if AudioEngine else None
        
        # Current program
        self.current_program = None
        self.current_step_index = 0
        self._is_playing = False
        
        # Build UI
        self._build_ui()
        
        # Load last program or create new
        self._load_or_create_program()
    
    def _setup_geometry(self):
        """Set window size using Fibonacci dimensions"""
        # 1597 × 987 (Fibonacci numbers, ratio ≈ φ)
        width = 1597
        height = 987
        
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(987, 610)  # Smaller Fibonacci pair
    
    def _build_ui(self):
        """Build the main UI structure"""
        # Main container
        self.main = ttk.Frame(self.root, style='TFrame')
        self.main.pack(fill='both', expand=True)
        
        # Header
        self._build_header()
        
        # Content area (sidebar + main)
        content = ttk.Frame(self.main)
        content.pack(fill='both', expand=True)
        
        # Sidebar (φ⁻¹ ≈ 38.2% width)
        self._build_sidebar(content)
        
        # Main content area
        self._build_main_content(content)
        
        # Footer/Status
        self._build_footer()
    
    def _build_header(self):
        """Build header with title and mode selector"""
        header = ttk.Frame(self.main, style='TFrame')
        header.pack(fill='x', padx=Spacing.XL, pady=Spacing.LG)
        
        # Logo/Title
        title_frame = ttk.Frame(header)
        title_frame.pack(side='left')
        
        ttk.Label(title_frame, text="φ", 
                 font=(Typography.FAMILY_SANS[0], FontSize.H2, 'bold'),
                 foreground=Colors.GOLD).pack(side='left')
        
        ttk.Label(title_frame, text=" Golden Studio",
                 font=(Typography.FAMILY_SANS[0], FontSize.H4),
                 foreground=Colors.TEXT_PRIMARY).pack(side='left', padx=(Spacing.SM, 0))
        
        # Mode tabs (right side)
        mode_frame = ttk.Frame(header)
        mode_frame.pack(side='right')
        
        self.mode_var = tk.StringVar(value="program")
        
        modes = [
            ("📋 Program", "program"),
            ("🎛️ Live", "live"),
            ("📊 Scope", "scope")
        ]
        
        for text, mode in modes:
            rb = ttk.Radiobutton(mode_frame, text=text, value=mode,
                                variable=self.mode_var,
                                command=self._on_mode_change)
            rb.pack(side='left', padx=Spacing.MD)
    
    def _build_sidebar(self, parent):
        """Build left sidebar with program structure"""
        # φ⁻¹ ≈ 0.382, so sidebar is ~38% width
        sidebar_width = int(1597 * PHI_CONJUGATE * PHI_CONJUGATE)  # ~385px
        
        sidebar = ttk.Frame(parent, width=sidebar_width, style='TFrame')
        sidebar.pack(side='left', fill='y', padx=(Spacing.XL, 0), pady=Spacing.MD)
        sidebar.pack_propagate(False)
        
        # ─────────────────────────────────────────────────────────────────
        # PROGRAM INFO
        # ─────────────────────────────────────────────────────────────────
        prog_card = GoldenCard(sidebar, title="PROGRAM")
        prog_card.pack(fill='x', pady=(0, Spacing.MD))
        
        # Program name
        name_row = ttk.Frame(prog_card.content)
        name_row.pack(fill='x', pady=Spacing.XS)
        
        ttk.Label(name_row, text="Name", style='Caption.TLabel').pack(anchor='w')
        
        self.program_name_var = tk.StringVar(value="Untitled Program")
        name_entry = ttk.Entry(name_row, textvariable=self.program_name_var,
                              font=(Typography.FAMILY_SANS[0], FontSize.BODY))
        name_entry.pack(fill='x', pady=Spacing.XS)
        
        # Duration display
        self.duration_label = ttk.Label(prog_card.content, text="Duration: 0:00",
                                        style='Mono.TLabel')
        self.duration_label.pack(anchor='w', pady=Spacing.XS)
        
        # ─────────────────────────────────────────────────────────────────
        # STEPS LIST
        # ─────────────────────────────────────────────────────────────────
        steps_card = GoldenCard(sidebar, title="STEPS")
        steps_card.pack(fill='both', expand=True, pady=(0, Spacing.MD))
        
        # Steps listbox with scrollbar
        list_frame = ttk.Frame(steps_card.content)
        list_frame.pack(fill='both', expand=True)
        
        self.steps_listbox = tk.Listbox(
            list_frame,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
            selectbackground=Colors.GOLD,
            selectforeground=Colors.TEXT_INVERSE,
            font=(Typography.FAMILY_MONO[0], FontSize.BODY_SM),
            borderwidth=0,
            highlightthickness=0,
            activestyle='none'
        )
        self.steps_listbox.pack(side='left', fill='both', expand=True)
        self.steps_listbox.bind('<<ListboxSelect>>', self._on_step_select)
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical',
                                  command=self.steps_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.steps_listbox.config(yscrollcommand=scrollbar.set)
        
        # Step buttons
        btn_row = ttk.Frame(steps_card.content)
        btn_row.pack(fill='x', pady=Spacing.SM)
        
        ttk.Button(btn_row, text="+ Add", width=8,
                  command=self._add_step).pack(side='left', padx=Spacing.XS)
        ttk.Button(btn_row, text="− Remove", width=8,
                  command=self._remove_step).pack(side='left', padx=Spacing.XS)
        ttk.Button(btn_row, text="↑↓ Move", width=8,
                  command=self._move_step).pack(side='left', padx=Spacing.XS)
        
        # ─────────────────────────────────────────────────────────────────
        # TRANSPORT
        # ─────────────────────────────────────────────────────────────────
        transport_card = GoldenCard(sidebar, title="TRANSPORT")
        transport_card.pack(fill='x')
        
        transport_row = ttk.Frame(transport_card.content)
        transport_row.pack(fill='x', pady=Spacing.SM)
        
        self.play_btn = ttk.Button(transport_row, text="▶ PLAY", 
                                   style='Primary.TButton',
                                   command=self._play_program)
        self.play_btn.pack(side='left', padx=Spacing.XS)
        
        self.stop_btn = ttk.Button(transport_row, text="⏹ STOP",
                                   command=self._stop_program, state='disabled')
        self.stop_btn.pack(side='left', padx=Spacing.XS)
        
        # Progress
        self.transport_progress = ttk.Progressbar(transport_card.content, 
                                                  mode='determinate')
        self.transport_progress.pack(fill='x', pady=Spacing.SM)
        
        self.transport_status = ttk.Label(transport_card.content, 
                                          text="Ready", style='Caption.TLabel')
        self.transport_status.pack(anchor='w')
    
    def _build_main_content(self, parent):
        """Build main content area with step editor"""
        main_area = ttk.Frame(parent, style='TFrame')
        main_area.pack(side='left', fill='both', expand=True, 
                      padx=Spacing.XL, pady=Spacing.MD)
        
        # Content container (changes based on mode)
        self.content_container = ttk.Frame(main_area)
        self.content_container.pack(fill='both', expand=True)
        
        # Build different views
        self._build_program_view()
    
    def _build_program_view(self):
        """Build the program/step editor view"""
        # Clear existing
        for widget in self.content_container.winfo_children():
            widget.destroy()
        
        # Two-column layout: Editor | Preview
        editor_frame = ttk.Frame(self.content_container)
        editor_frame.pack(side='left', fill='both', expand=True)
        
        preview_frame = ttk.Frame(self.content_container, width=300)
        preview_frame.pack(side='right', fill='y', padx=(Spacing.LG, 0))
        preview_frame.pack_propagate(False)
        
        # ─────────────────────────────────────────────────────────────────
        # STEP EDITOR
        # ─────────────────────────────────────────────────────────────────
        self._build_step_editor(editor_frame)
        
        # ─────────────────────────────────────────────────────────────────
        # PREVIEW / VISUALIZATION
        # ─────────────────────────────────────────────────────────────────
        self._build_preview(preview_frame)
    
    def _build_step_editor(self, parent):
        """Build step parameter editor"""
        # Step name/type header
        header = ttk.Frame(parent)
        header.pack(fill='x', pady=(0, Spacing.LG))
        
        self.step_name_var = tk.StringVar(value="New Step")
        ttk.Entry(header, textvariable=self.step_name_var,
                 font=(Typography.FAMILY_SANS[0], FontSize.H5, 'bold')).pack(fill='x')
        
        # Step type selector
        type_row = ttk.Frame(header)
        type_row.pack(fill='x', pady=Spacing.SM)
        
        ttk.Label(type_row, text="Type:", style='Caption.TLabel').pack(side='left')
        
        self.step_type_var = tk.StringVar(value="binaural")
        step_types = [
            ("Binaural", "binaural"),
            ("Spectral", "spectral"),
            ("Chakra Journey", "chakra_journey"),
            ("Sweep", "sweep"),
            ("Silence", "silence")
        ]
        
        for text, stype in step_types:
            ttk.Radiobutton(type_row, text=text, value=stype,
                           variable=self.step_type_var,
                           command=self._on_step_type_change).pack(side='left', padx=Spacing.MD)
        
        # Parameters in cards
        params_frame = ttk.Frame(parent)
        params_frame.pack(fill='both', expand=True)
        
        # Left column: Frequency & Duration
        left_col = ttk.Frame(params_frame)
        left_col.pack(side='left', fill='both', expand=True, padx=(0, Spacing.MD))
        
        # Right column: Position & Fade
        right_col = ttk.Frame(params_frame)
        right_col.pack(side='left', fill='both', expand=True)
        
        # ─────────────────────────────────────────────────────────────────
        # FREQUENCY CARD
        # ─────────────────────────────────────────────────────────────────
        freq_card = GoldenCard(left_col, title="FREQUENCY")
        freq_card.pack(fill='x', pady=(0, Spacing.MD))
        
        # Base frequency
        self._create_param_row(freq_card.content, "Base (Hz)", 
                              self._create_freq_var("base", 432.0),
                              20, 1000, "base_freq")
        
        # Beat frequency (for binaural)
        self._create_param_row(freq_card.content, "Beat (Hz)",
                              self._create_freq_var("beat", 7.83),
                              0.5, 40, "beat_freq")
        
        # Phase angle
        self._create_param_row(freq_card.content, "Phase (°)",
                              self._create_freq_var("phase", 26.26),
                              0, 360, "phase_angle")
        
        # Harmonics toggle
        harm_row = ttk.Frame(freq_card.content)
        harm_row.pack(fill='x', pady=Spacing.SM)
        
        self.harmonics_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(harm_row, text="Include φ harmonics",
                       variable=self.harmonics_var).pack(anchor='w')
        
        # ─────────────────────────────────────────────────────────────────
        # DURATION CARD
        # ─────────────────────────────────────────────────────────────────
        dur_card = GoldenCard(left_col, title="DURATION")
        dur_card.pack(fill='x', pady=(0, Spacing.MD))
        
        # Duration entry
        dur_row = ttk.Frame(dur_card.content)
        dur_row.pack(fill='x', pady=Spacing.SM)
        
        self.duration_min_var = tk.IntVar(value=5)
        self.duration_sec_var = tk.IntVar(value=0)
        
        ttk.Label(dur_row, text="Minutes:").pack(side='left')
        ttk.Entry(dur_row, textvariable=self.duration_min_var, 
                 width=4).pack(side='left', padx=Spacing.SM)
        
        ttk.Label(dur_row, text="Seconds:").pack(side='left', padx=(Spacing.MD, 0))
        ttk.Entry(dur_row, textvariable=self.duration_sec_var,
                 width=4).pack(side='left', padx=Spacing.SM)
        
        # Quick duration buttons (Fibonacci times)
        quick_row = ttk.Frame(dur_card.content)
        quick_row.pack(fill='x', pady=Spacing.SM)
        
        ttk.Label(quick_row, text="Quick:", style='Caption.TLabel').pack(side='left')
        
        for label, mins, secs in [("30s", 0, 30), ("1m", 1, 0), ("2m", 2, 0), 
                                   ("3m", 3, 0), ("5m", 5, 0), ("8m", 8, 0),
                                   ("13m", 13, 0), ("21m", 21, 0)]:
            ttk.Button(quick_row, text=label, width=4,
                      command=lambda m=mins, s=secs: self._set_duration(m, s)
                      ).pack(side='left', padx=2)
        
        # ─────────────────────────────────────────────────────────────────
        # POSITION CARD (Body mapping)
        # ─────────────────────────────────────────────────────────────────
        pos_card = GoldenCard(right_col, title="BODY POSITION")
        pos_card.pack(fill='x', pady=(0, Spacing.MD))
        
        # Position presets
        pos_row = ttk.Frame(pos_card.content)
        pos_row.pack(fill='x', pady=Spacing.SM)
        
        self.position_var = tk.StringVar(value="SOLAR_PLEXUS")
        positions = ["HEAD", "THROAT", "HEART", "SOLAR_PLEXUS", "SACRAL", "ROOT", "FEET"]
        
        for pos in positions:
            ttk.Radiobutton(pos_row, text=pos.replace("_", " ").title(),
                           value=pos, variable=self.position_var,
                           command=self._on_position_change).pack(anchor='w')
        
        # Custom pan value
        pan_row = ttk.Frame(pos_card.content)
        pan_row.pack(fill='x', pady=Spacing.SM)
        
        ttk.Label(pan_row, text="Pan:", style='Caption.TLabel').pack(side='left')
        
        self.pan_var = tk.DoubleVar(value=-0.385)
        ttk.Scale(pan_row, from_=-1, to=1, variable=self.pan_var,
                 orient='horizontal').pack(side='left', fill='x', expand=True, padx=Spacing.SM)
        
        self.pan_label = ttk.Label(pan_row, text="-0.39", style='Mono.TLabel', width=6)
        self.pan_label.pack(side='left')
        
        self.pan_var.trace_add('write', lambda *a: self.pan_label.config(
            text=f"{self.pan_var.get():+.2f}"))
        
        # ─────────────────────────────────────────────────────────────────
        # FADE CARD
        # ─────────────────────────────────────────────────────────────────
        fade_card = GoldenCard(right_col, title="FADE CURVES")
        fade_card.pack(fill='x', pady=(0, Spacing.MD))
        
        # Fade in
        fade_in_row = ttk.Frame(fade_card.content)
        fade_in_row.pack(fill='x', pady=Spacing.SM)
        
        ttk.Label(fade_in_row, text="Fade In:", style='Caption.TLabel', width=10).pack(side='left')
        
        self.fade_in_var = tk.StringVar(value="GOLDEN")
        for curve in ["NONE", "LINEAR", "GOLDEN", "EXPONENTIAL"]:
            ttk.Radiobutton(fade_in_row, text=curve.title(), value=curve,
                           variable=self.fade_in_var).pack(side='left', padx=Spacing.XS)
        
        self.fade_in_duration = tk.DoubleVar(value=5.0)
        ttk.Label(fade_in_row, text="Duration:").pack(side='left', padx=(Spacing.MD, Spacing.SM))
        ttk.Entry(fade_in_row, textvariable=self.fade_in_duration, width=5).pack(side='left')
        ttk.Label(fade_in_row, text="s").pack(side='left')
        
        # Fade out
        fade_out_row = ttk.Frame(fade_card.content)
        fade_out_row.pack(fill='x', pady=Spacing.SM)
        
        ttk.Label(fade_out_row, text="Fade Out:", style='Caption.TLabel', width=10).pack(side='left')
        
        self.fade_out_var = tk.StringVar(value="GOLDEN")
        for curve in ["NONE", "LINEAR", "GOLDEN", "EXPONENTIAL"]:
            ttk.Radiobutton(fade_out_row, text=curve.title(), value=curve,
                           variable=self.fade_out_var).pack(side='left', padx=Spacing.XS)
        
        self.fade_out_duration = tk.DoubleVar(value=5.0)
        ttk.Label(fade_out_row, text="Duration:").pack(side='left', padx=(Spacing.MD, Spacing.SM))
        ttk.Entry(fade_out_row, textvariable=self.fade_out_duration, width=5).pack(side='left')
        ttk.Label(fade_out_row, text="s").pack(side='left')
        
        # ─────────────────────────────────────────────────────────────────
        # PRESETS
        # ─────────────────────────────────────────────────────────────────
        presets_card = GoldenCard(right_col, title="QUICK PRESETS")
        presets_card.pack(fill='x')
        
        presets = [
            ("🌅 Chakra Sunrise", self._preset_chakra_sunrise),
            ("🧘 Deep Theta 7.83Hz", self._preset_theta),
            ("💫 Golden 54Hz", self._preset_golden_54),
            ("🌊 Delta Sleep", self._preset_delta),
            ("⚡ Gamma Focus", self._preset_gamma),
        ]
        
        for text, command in presets:
            ttk.Button(presets_card.content, text=text,
                      command=command).pack(fill='x', pady=Spacing.XS)
    
    def _build_preview(self, parent):
        """Build preview/visualization panel"""
        preview_card = GoldenCard(parent, title="PREVIEW")
        preview_card.pack(fill='both', expand=True)
        
        # Canvas for visualization
        self.preview_canvas = tk.Canvas(
            preview_card.content,
            bg=Colors.BG_DEEPEST,
            highlightthickness=0,
            width=280,
            height=400
        )
        self.preview_canvas.pack(fill='both', expand=True, pady=Spacing.SM)
        
        # Draw initial state
        self._draw_preview()
        
        # Preview play button
        ttk.Button(preview_card.content, text="▶ Preview Step",
                  command=self._preview_step).pack(fill='x', pady=Spacing.SM)
    
    def _build_footer(self):
        """Build footer with status and file operations"""
        footer = ttk.Frame(self.main, style='TFrame')
        footer.pack(fill='x', side='bottom', padx=Spacing.XL, pady=Spacing.MD)
        
        # Left: File operations
        file_frame = ttk.Frame(footer)
        file_frame.pack(side='left')
        
        ttk.Button(file_frame, text="New", command=self._new_program).pack(side='left', padx=Spacing.XS)
        ttk.Button(file_frame, text="Open", command=self._open_program).pack(side='left', padx=Spacing.XS)
        ttk.Button(file_frame, text="Save", command=self._save_program).pack(side='left', padx=Spacing.XS)
        ttk.Button(file_frame, text="Export WAV", command=self._export_wav).pack(side='left', padx=Spacing.XS)
        
        # Right: Status
        self.status_label = ttk.Label(footer, text="Ready", style='Caption.TLabel')
        self.status_label.pack(side='right')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _create_freq_var(self, name: str, default: float) -> tk.DoubleVar:
        """Create a tracked frequency variable"""
        var = tk.DoubleVar(value=default)
        setattr(self, f"{name}_var", var)
        return var
    
    def _create_param_row(self, parent, label: str, var: tk.DoubleVar,
                         min_val: float, max_val: float, name: str):
        """Create a labeled parameter row with slider and entry"""
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=Spacing.SM)
        
        ttk.Label(row, text=label, style='Caption.TLabel', width=12).pack(side='left')
        
        scale = ttk.Scale(row, from_=min_val, to=max_val, variable=var,
                         orient='horizontal')
        scale.pack(side='left', fill='x', expand=True, padx=Spacing.SM)
        
        entry = ttk.Entry(row, textvariable=var, width=8)
        entry.pack(side='left')
        
        setattr(self, f"{name}_scale", scale)
        setattr(self, f"{name}_entry", entry)
    
    def _set_duration(self, mins: int, secs: int):
        """Set duration from quick button"""
        self.duration_min_var.set(mins)
        self.duration_sec_var.set(secs)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _on_mode_change(self):
        """Handle mode tab change"""
        mode = self.mode_var.get()
        if mode == "program":
            self._build_program_view()
        elif mode == "live":
            self._build_live_view()
        elif mode == "scope":
            self._build_scope_view()
    
    def _on_step_select(self, event):
        """Handle step selection in list"""
        selection = self.steps_listbox.curselection()
        if selection:
            self.current_step_index = selection[0]
            self._load_step_to_editor(self.current_step_index)
    
    def _on_step_type_change(self):
        """Handle step type change"""
        # Update UI based on step type
        step_type = self.step_type_var.get()
        # Show/hide relevant controls
        self._draw_preview()
    
    def _on_position_change(self):
        """Handle body position preset change"""
        pos = self.position_var.get()
        # Map position to pan value
        position_pans = {
            "HEAD": -1.0,
            "THROAT": -0.7,
            "HEART": -0.5,
            "SOLAR_PLEXUS": -0.385,
            "SACRAL": -0.18,
            "ROOT": 0.3,
            "FEET": 0.795
        }
        self.pan_var.set(position_pans.get(pos, 0.0))
        self._draw_preview()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _add_step(self):
        """Add a new step to the program"""
        name = f"Step {self.steps_listbox.size() + 1}"
        self.steps_listbox.insert('end', name)
        self.steps_listbox.select_clear(0, 'end')
        self.steps_listbox.select_set('end')
        self._on_step_select(None)
        self._update_duration()
    
    def _remove_step(self):
        """Remove selected step"""
        selection = self.steps_listbox.curselection()
        if selection:
            self.steps_listbox.delete(selection[0])
            self._update_duration()
    
    def _move_step(self):
        """Move step up/down (cycles through)"""
        selection = self.steps_listbox.curselection()
        if selection:
            idx = selection[0]
            if idx > 0:
                # Swap with previous
                text = self.steps_listbox.get(idx)
                self.steps_listbox.delete(idx)
                self.steps_listbox.insert(idx - 1, text)
                self.steps_listbox.select_set(idx - 1)
    
    def _load_step_to_editor(self, index: int):
        """Load step data into editor"""
        # This will be implemented with actual Step objects
        self.step_name_var.set(self.steps_listbox.get(index))
        self._draw_preview()
    
    def _update_duration(self):
        """Update total program duration"""
        # Sum all step durations
        total_secs = self.steps_listbox.size() * 300  # Placeholder: 5 min each
        mins = total_secs // 60
        secs = total_secs % 60
        self.duration_label.config(text=f"Duration: {mins}:{secs:02d}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRESETS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _preset_chakra_sunrise(self):
        """Load Chakra Sunrise preset"""
        self.step_type_var.set("chakra_journey")
        self.base_var.set(128.0)
        self.position_var.set("SOLAR_PLEXUS")
        self._set_duration(5, 0)
        self.fade_in_var.set("GOLDEN")
        self.fade_out_var.set("GOLDEN")
        self.step_name_var.set("Chakra Sunrise Journey")
        self._draw_preview()
    
    def _preset_theta(self):
        """Load Deep Theta preset"""
        self.step_type_var.set("binaural")
        self.base_var.set(432.0)
        self.beat_var.set(7.83)  # Schumann
        self.phase_var.set(26.26)
        self._set_duration(8, 0)
        self.step_name_var.set("Deep Theta 7.83Hz")
        self._draw_preview()
    
    def _preset_golden_54(self):
        """Load Golden 54Hz preset"""
        self.step_type_var.set("spectral")
        self.base_var.set(54.0)  # 432/8 = 54
        self.phase_var.set(26.26)
        self.harmonics_var.set(True)
        self._set_duration(5, 0)
        self.step_name_var.set("Golden 54Hz + φ Harmonics")
        self._draw_preview()
    
    def _preset_delta(self):
        """Load Delta Sleep preset"""
        self.step_type_var.set("binaural")
        self.base_var.set(108.0)
        self.beat_var.set(2.0)
        self._set_duration(21, 0)  # Fibonacci
        self.step_name_var.set("Delta Sleep 2Hz")
        self._draw_preview()
    
    def _preset_gamma(self):
        """Load Gamma Focus preset"""
        self.step_type_var.set("binaural")
        self.base_var.set(432.0)
        self.beat_var.set(40.0)
        self._set_duration(13, 0)  # Fibonacci
        self.step_name_var.set("Gamma Focus 40Hz")
        self._draw_preview()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _draw_preview(self):
        """Draw step preview visualization"""
        if not hasattr(self, 'preview_canvas'):
            return
        
        canvas = self.preview_canvas
        canvas.delete('all')
        
        w = canvas.winfo_width() or 280
        h = canvas.winfo_height() or 400
        
        # Draw body silhouette
        cx = w // 2
        
        # Head
        canvas.create_oval(cx - 20, 30, cx + 20, 70, 
                          outline=Colors.BORDER_DEFAULT, width=2)
        
        # Torso
        canvas.create_rectangle(cx - 25, 80, cx + 25, 200,
                               outline=Colors.BORDER_DEFAULT, width=2)
        
        # Legs
        canvas.create_rectangle(cx - 25, 210, cx - 5, 350,
                               outline=Colors.BORDER_DEFAULT, width=2)
        canvas.create_rectangle(cx + 5, 210, cx + 25, 350,
                               outline=Colors.BORDER_DEFAULT, width=2)
        
        # Highlight current position
        pos = self.position_var.get() if hasattr(self, 'position_var') else "SOLAR_PLEXUS"
        
        position_y = {
            "HEAD": 50,
            "THROAT": 75,
            "HEART": 110,
            "SOLAR_PLEXUS": 150,
            "SACRAL": 180,
            "ROOT": 200,
            "FEET": 330
        }
        
        y = position_y.get(pos, 150)
        
        # Draw golden circle at position
        r = 15
        canvas.create_oval(cx - r, y - r, cx + r, y + r,
                          fill=Colors.GOLD, outline=Colors.GOLD_LIGHT, width=2)
        
        # Draw sound waves
        for i in range(1, 4):
            wave_r = r + i * 15
            alpha_hex = format(int(255 * (1 - i/4)), '02x')
            canvas.create_oval(cx - wave_r, y - wave_r//2, 
                              cx + wave_r, y + wave_r//2,
                              outline=f"#{alpha_hex}{alpha_hex}50", width=1)
        
        # Frequency info
        if hasattr(self, 'base_var'):
            freq_text = f"{self.base_var.get():.1f} Hz"
            canvas.create_text(cx, h - 30, text=freq_text,
                              fill=Colors.GOLD, font=(Typography.FAMILY_MONO[0], FontSize.BODY))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PLAYBACK
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _play_program(self):
        """Start program playback"""
        if not self.audio:
            self.status_label.config(text="Audio engine not available")
            return
        
        self._is_playing = True
        self.play_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.transport_status.config(text="Playing...")
        # TODO: Implement actual playback
    
    def _stop_program(self):
        """Stop program playback"""
        self._is_playing = False
        if self.audio:
            self.audio.stop()
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.transport_status.config(text="Stopped")
        self.transport_progress['value'] = 0
    
    def _preview_step(self):
        """Preview current step"""
        if not self.audio:
            return
        
        # Quick 10-second preview
        self.status_label.config(text="Previewing step...")
        # TODO: Implement step preview
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FILE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _new_program(self):
        """Create new program"""
        self.program_name_var.set("Untitled Program")
        self.steps_listbox.delete(0, 'end')
        self._update_duration()
        self.status_label.config(text="New program created")
    
    def _open_program(self):
        """Open program from file"""
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=Path(__file__).parent / "programs" / "presets"
        )
        if filepath:
            self._load_program(filepath)
    
    def _save_program(self):
        """Save program to file"""
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=self.program_name_var.get().replace(" ", "_").lower()
        )
        if filepath:
            self._save_program_to_file(filepath)
    
    def _export_wav(self):
        """Export program as WAV file"""
        self.status_label.config(text="WAV export not yet implemented")
    
    def _load_program(self, filepath: str):
        """Load program from JSON file"""
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            self.program_name_var.set(data.get('name', 'Loaded Program'))
            
            # Load steps
            self.steps_listbox.delete(0, 'end')
            for step in data.get('steps', []):
                self.steps_listbox.insert('end', step.get('name', 'Unnamed Step'))
            
            self._update_duration()
            self.status_label.config(text=f"Loaded: {Path(filepath).name}")
        except Exception as e:
            self.status_label.config(text=f"Error loading: {e}")
    
    def _save_program_to_file(self, filepath: str):
        """Save program to JSON file"""
        # TODO: Build proper program data structure
        data = {
            'name': self.program_name_var.get(),
            'steps': []
        }
        
        for i in range(self.steps_listbox.size()):
            data['steps'].append({
                'name': self.steps_listbox.get(i)
            })
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.status_label.config(text=f"Saved: {Path(filepath).name}")
        except Exception as e:
            self.status_label.config(text=f"Error saving: {e}")
    
    def _load_or_create_program(self):
        """Load last program or create new"""
        # For now, create a demo program
        self.program_name_var.set("Golden Session Demo")
        
        demo_steps = [
            "1. Theta Grounding (7.83Hz)",
            "2. Chakra Sunrise Journey",
            "3. Golden 54Hz Meditation",
            "4. Integration Silence"
        ]
        
        for step in demo_steps:
            self.steps_listbox.insert('end', step)
        
        self._update_duration()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PLACEHOLDER VIEWS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _build_live_view(self):
        """Build live control view"""
        for widget in self.content_container.winfo_children():
            widget.destroy()
        
        ttk.Label(self.content_container, text="🎛️ Live Control Mode",
                 style='Heading.TLabel').pack(pady=Spacing.XXL)
        
        ttk.Label(self.content_container, 
                 text="Real-time parameter control coming soon...",
                 style='Caption.TLabel').pack()
    
    def _build_scope_view(self):
        """Build oscilloscope view"""
        for widget in self.content_container.winfo_children():
            widget.destroy()
        
        ttk.Label(self.content_container, text="📊 Oscilloscope Mode",
                 style='Heading.TLabel').pack(pady=Spacing.XXL)
        
        ttk.Label(self.content_container,
                 text="Waveform visualization coming soon...",
                 style='Caption.TLabel').pack()
    
    def run(self):
        """Start the application"""
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = GoldenStudioApp()
    app.run()


if __name__ == "__main__":
    main()
