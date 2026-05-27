# -*- coding: utf-8 -*-
"""
Script de instalación del paquete.
Permite instalar el proyecto en modo editable para resolver imports.
"""

from setuptools import setup, find_packages

# CONFIGURACION DEL PAQUETE
setup(
    name="biomecanica_tfm",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "pydantic",
        "ahrs",
        "scikit-learn",
        "influxdb-client"
    ]
)