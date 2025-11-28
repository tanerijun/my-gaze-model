# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

# Aggressively exclude to speed up startup
sys.modules['matplotlib'] = None
sys.modules['mpl_toolkits'] = None

# Get the absolute path to the project root (where this spec file is)
project_root = os.path.abspath(os.getcwd())

a = Analysis(
    [os.path.join('data_collector', 'main.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        ('weights', 'weights'),
        ('mediapipe_models', 'mediapipe_models'),
        (os.path.join('data_collector', 'ui', 'assets'), os.path.join('data_collector', 'ui', 'assets')),
        ('.env', '.')
    ],
    hiddenimports=[
        'src',
        'src.inference',
        'data_collector',
        'data_collector.core',
        'data_collector.ui',
        'data_collector.utils',
        'dotenv',
        'boto3',
        'botocore',
        'pynput',
        'pynput.mouse',
        'pynput.keyboard',
        'cv2',
        'torch',
        'mediapipe',
        'numpy',
        'psutil',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludedimports=[
        'tkinter',
        'matplotlib',
        'mpl_toolkits',
        'matplotlib.pyplot',
        'seaborn',
        'pandas',
        'fastapi',
        'aiohttp',
        'websockets',
        'mss',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher, optimize=1)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GazeDataCollector',
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
    name='GazeDataCollector',
)
