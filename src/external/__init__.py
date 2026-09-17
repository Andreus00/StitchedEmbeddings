"""Bundled third-party code. GarmentCode's own modules import each other as top-level packages
(`pygarment`, `assets`), and the Hunyuan3D files as `hunyuan_model`, so both folders are put on sys.path."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _sub in ('GarmentCode', 'Hunyuan3D'):
    _p = os.path.join(_here, _sub)
    if _p not in sys.path:
        sys.path.append(_p)
