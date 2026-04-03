"""
Mini ICU Physiology Engine

A lightweight biomedical simulation library for generating realistic vital signs data.
"""

from .physiology_simulator import PhysiologySimulator, quick_simulate

__version__ = "1.0.0"
__author__ = "Mini ICU Team"
__email__ = "team@minicu.org"

__all__ = ['PhysiologySimulator', 'quick_simulate']
