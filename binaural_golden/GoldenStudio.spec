# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Golden Studio macOS app
"""

import sys
import os

block_cipher = None

# Path to source files
src_path = os.path.join(os.path.dirname(os.path.abspath(SPEC)), 'src')

a = Analysis(
    ['src/golden_studio.py'],
    pathex=[src_path],
    binaries=[],
    datas=[
        ('src/presets', 'presets'),
        ('src/programs/presets', 'programs/presets'),
        ('src/core', 'core'),
        ('src/programs', 'programs'),
        ('src/ui', 'ui'),
        ('src/golden_constants.py', '.'),
        ('src/spectral_sound.py', '.'),
        ('src/molecular_sound.py', '.'),
    ],
    hiddenimports=[
        'numpy',
        'pyaudio',
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'golden_constants',
        'spectral_sound',
        'molecular_sound',
        'core',
        'core.audio_engine',
        'core.golden_math',
        'programs',
        'programs.program',
        'programs.step',
        'ui',
        'ui.golden_theme',
        'ui.binaural_tab',
        'ui.spectral_tab',
        'ui.molecular_tab',
        'ui.harmonic_tree_tab',
        'ui.vibroacoustic_tab',
        'ui.session_builder_tab',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'PIL', 'pandas', 'IPython', 'jupyter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Golden Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Golden Studio',
)

app = BUNDLE(
    coll,
    name='Golden Studio.app',
    icon=None,  # Add GoldenStudio.icns here if you have an icon
    bundle_identifier='com.rememberance.goldenstudio',
    info_plist={
        'CFBundleName': 'Golden Studio',
        'CFBundleDisplayName': 'Golden Studio',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0',
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'Golden Studio needs microphone access for audio features.',
        'LSMinimumSystemVersion': '10.15',
    },
)
