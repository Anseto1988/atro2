# -*- mode: python ; coding: utf-8 -*-
"""AstroAI Suite — PyInstaller spec for single-folder bundle."""
import sys
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "astroai" / "ui" / "main" / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "astroai" / "ui" / "resources"), "astroai/ui/resources"),
    ],
    hiddenimports=[
        "astroai.core.io.fits_io",
        "astroai.core.io.raw_io",
        "astroai.core.io.xisf_io",
        "astroai.core.pipeline.base",
        "astroai.core.calibration.matcher",
        "astroai.core.calibration.calibrate",
        "astroai.engine.registration.aligner",
        "astroai.engine.stacking.stacker",
        "astroai.inference.backends.gpu",
        "astroai.inference.models.registry",
        "astroai.inference.scoring.frame_scorer",
        "astroai.processing.background",
        "astroai.processing.background.extractor",
        "astroai.processing.background.gradient_remover",
        "astroai.processing.background.pipeline_step",
        "astroai.processing.channels",
        "astroai.processing.channels.combiner",
        "astroai.processing.channels.narrowband_mapper",
        "astroai.processing.channels.pipeline_step",
        "astroai.processing.deconvolution",
        "astroai.processing.deconvolution.deconvolver",
        "astroai.processing.deconvolution.pipeline_step",
        "astroai.processing.denoise",
        "astroai.processing.denoise.denoiser",
        "astroai.processing.denoise.pipeline_step",
        "astroai.processing.stretch",
        "astroai.processing.stretch.stretcher",
        "astroai.processing.stretch.pipeline_step",
        "astroai.processing.stars",
        "astroai.processing.stars.star_manager",
        "astroai.processing.stars.pipeline_step",
        "astroai.ui.main.loader",
        "astroai.ui.widgets.image_viewer",
        "astroai.ui.widgets.histogram_widget",
        "astroai.ui.widgets.workflow_graph",
        "astroai.ui.widgets.progress_widget",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "numpy",
        "scipy",
        "astropy.io.fits",
        "rawpy",
        "PIL",
        "tqdm",
        "onnxruntime",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "pytest",
        "mypy",
        "ruff",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AstroAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=sys.platform == "darwin",
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AstroAI",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="AstroAI.app",
        icon=None,
        bundle_identifier="com.astroai.suite",
        info_plist={"CFBundleShortVersionString": "0.1.0-alpha"},
    )
