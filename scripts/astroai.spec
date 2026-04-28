# -*- mode: python ; coding: utf-8 -*-
"""AstroAI Suite — PyInstaller spec for single-folder bundle (Windows/Linux/macOS)."""
import sys
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH).parent
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"

_hidden_imports = [
    "astroai.core.io.fits_io",
    "astroai.core.io.raw_io",
    "astroai.core.io.xisf_io",
    "astroai.core.pipeline.base",
    "astroai.core.calibration.matcher",
    "astroai.core.calibration.calibrate",
    "astroai.engine.registration.aligner",
    "astroai.engine.stacking.stacker",
    "astroai.astrometry",
    "astroai.astrometry.catalog",
    "astroai.astrometry.solver",
    "astroai.astrometry.pipeline_step",
    "astroai.core.pipeline.platesolving_step",
    "astroai.core.calibration.gpu_engine",
    "astroai.engine.platesolving.astap_binary",
    "astroai.engine.platesolving.solver",
    "astroai.engine.platesolving.wcs_writer",
    "astroai.engine.platesolving.annotation",
    "astroai.engine.drizzle",
    "astroai.engine.drizzle.engine",
    "astroai.engine.drizzle.pipeline_step",
    "astroai.engine.mosaic",
    "astroai.engine.mosaic.engine",
    "astroai.engine.mosaic.pipeline_step",
    "astroai.processing.color",
    "astroai.processing.color.calibrator",
    "astroai.processing.color.pipeline_step",
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
    "astroai.ui.widgets.calibration_benchmark",
    "astroai.ui.widgets.image_viewer",
    "astroai.ui.widgets.histogram_widget",
    "astroai.ui.widgets.workflow_graph",
    "astroai.ui.widgets.progress_widget",
    "astroai.ui.widgets.drizzle_panel",
    "astroai.ui.widgets.mosaic_panel",
    "astroai.ui.widgets.color_calibration_panel",
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
    "reproject",
    "astroquery",
    "astroquery.gaia",
    "astroquery.vizier",
]

if IS_WIN:
    try:
        import pywintypes  # noqa: F401
        _hidden_imports += ["win32crypt", "win32api", "pywintypes"]
    except ImportError:
        pass

a = Analysis(
    [str(ROOT / "astroai" / "ui" / "main" / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "astroai" / "ui" / "resources"), "astroai/ui/resources"),
        (str(ROOT / "astroai" / "engine" / "platesolving" / "bin"), "astroai/engine/platesolving/bin"),
    ],
    hiddenimports=_hidden_imports,
    hookspath=[str(ROOT / "scripts" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "astropy.visualization",
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

_entitlements = str(ROOT / "scripts" / "entitlements.plist") if IS_MAC else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AstroAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=not IS_MAC,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=IS_MAC,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=_entitlements,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=not IS_MAC,
    upx_exclude=[],
    name="AstroAI",
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name="AstroAI.app",
        icon=None,
        bundle_identifier="com.astroai.suite",
        info_plist={
            "CFBundleShortVersionString": "2.1.0-alpha",
            "CFBundleDisplayName": "AstroAI Suite",
            "CFBundleName": "AstroAI",
            "CFBundlePackageType": "APPL",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
            "NSRequiresAquaSystemAppearance": False,
        },
    )
