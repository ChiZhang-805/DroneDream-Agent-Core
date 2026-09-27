"""Optional CPU-only extension; pure Python remains available on unsupported platforms."""

import os
from setuptools import Extension, setup

setup(ext_modules=[Extension('_dronedream_metric_scan', ['native/metric_scan/metric_scan.cpp'],
    language='c++', optional=True,
    extra_compile_args=['/std:c++17', '/O2', '/fp:strict', '/utf-8'] if os.name == 'nt'
    else ['-std=c++17', '-O3', '-fno-fast-math', '-ffp-contract=off'])])
