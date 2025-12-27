"""
Setup script per creare Golden Studio.app per macOS
Usa: python setup_app.py py2app
"""

from setuptools import setup

APP = ['src/golden_studio.py']
DATA_FILES = [
    ('presets', ['src/presets/Punto987.json']),
    ('programs/presets', ['src/programs/presets/chakra_sunrise.json']),
]

OPTIONS = {
    'argv_emulation': False,
    # 'iconfile': 'GoldenStudio.icns',  # Aggiungi icona custom se vuoi
    'plist': {
        'CFBundleName': 'Golden Studio',
        'CFBundleDisplayName': 'Golden Studio',
        'CFBundleIdentifier': 'com.rememberance.goldenstudio',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'Golden Studio needs microphone access for audio input.',
        'LSMinimumSystemVersion': '10.15',
    },
    'packages': [
        'numpy',
        'tkinter',
    ],
    'includes': [
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
    'excludes': ['matplotlib', 'scipy', 'PIL', 'pandas'],
    'resources': [],
}

setup(
    app=APP,
    name='Golden Studio',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
