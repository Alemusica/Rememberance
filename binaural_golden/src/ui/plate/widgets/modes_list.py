"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MODES LIST PANEL - Mode Selection Widget                  ║
║                                                                              ║
║   Displays list of computed modes with:                                      ║
║   • Frequency                                                                 ║
║   • Mode numbers (m, n)                                                       ║
║   • Play button                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List

from ..viewmodel import PlateState


class ModesListPanel(ttk.Frame):
    """
    Panel showing list of computed modes.
    """
    
    def __init__(
        self,
        parent,
        on_mode_selected: Callable[[int], None] = None,
        on_play_clicked: Callable[[int], None] = None,
    ):
        super().__init__(parent)
        
        # Callbacks
        self._on_mode_selected = on_mode_selected
        self._on_play_clicked = on_play_clicked
        
        # Create widgets
        self._create_widgets()
    
    def _create_widgets(self):
        """Create mode list widgets."""
        
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X)
        
        ttk.Label(
            header_frame,
            text="🎵 Modal Frequencies",
            font=("SF Pro", 11, "bold")
        ).pack(side=tk.LEFT, padx=5)
        
        # Treeview
        columns = ("mode", "freq", "m_n")
        
        self._tree = ttk.Treeview(
            self,
            columns=columns,
            show="headings",
            height=6,
            selectmode="browse"
        )
        
        self._tree.heading("mode", text="#")
        self._tree.heading("freq", text="Frequency")
        self._tree.heading("m_n", text="(m,n)")
        
        self._tree.column("mode", width=40, anchor="center")
        self._tree.column("freq", width=100, anchor="center")
        self._tree.column("m_n", width=60, anchor="center")
        
        self._tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)
        
        # Bind selection
        self._tree.bind("<<TreeviewSelect>>", self._on_select)
        self._tree.bind("<Double-1>", self._on_double_click)
    
    def update_state(self, state: PlateState):
        """Update mode list from state."""
        # Clear existing
        for item in self._tree.get_children():
            self._tree.delete(item)
        
        # Add modes
        if state.fem_result:
            for i, mode in enumerate(state.fem_result.modes):
                freq_str = f"{mode.frequency:.1f} Hz"
                m_n_str = f"({mode.m},{mode.n})" if mode.m > 0 else "--"
                
                item_id = self._tree.insert(
                    "",
                    "end",
                    values=(i + 1, freq_str, m_n_str),
                    tags=("selected" if i == state.selected_mode_idx else "",)
                )
                
                # Highlight selected
                if i == state.selected_mode_idx:
                    self._tree.selection_set(item_id)
    
    def _on_select(self, event):
        """Mode selected."""
        selection = self._tree.selection()
        if selection:
            item = selection[0]
            index = self._tree.index(item)
            
            if self._on_mode_selected:
                self._on_mode_selected(index)
    
    def _on_double_click(self, event):
        """Double-click to play."""
        selection = self._tree.selection()
        if selection:
            item = selection[0]
            index = self._tree.index(item)
            
            if self._on_play_clicked:
                self._on_play_clicked(index)
