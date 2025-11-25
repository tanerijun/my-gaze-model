# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Aggressively exclude to speed up startup
import sys
sys.modules['matplotlib'] = None
sys.modules['mpl_toolkits'] = None

a = Analysis(
    ['data_collector/main.py'],
    pathex=['/Users/tanerijun/projects/master-research/my-gaze-model/inference'],
    binaries=[],
    datas=[
        ('weights', 'weights'),
        ('mediapipe_models', 'mediapipe_models'),
        ('data_collector/ui/assets', 'data_collector/ui/assets'),
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

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='GazeDataCollector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
