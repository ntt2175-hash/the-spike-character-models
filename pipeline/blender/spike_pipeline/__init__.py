"""The Spike character pipeline for Blender 4.2 LTS - 5.2.

Modules
    common        scene conventions (collections, names) and small helpers
    rig           builds the standard armature for a character from the specs
    rom           generates the volleyball range-of-motion test action
    outline       bakes tangent-space smoothed normals for the inverted-hull outline
    validate      checks a character scene against the specs, stage by stage
    presentation  renders the QC set (silhouette, clay, turnaround, close-ups, poses)
    export        exports the game collection to glTF for Godot

Run everything through pipeline/blender/run.py (see its docstring), or import
the modules from Blender's Python console after adding pipeline/blender to
sys.path.
"""
import sys
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

__version__ = "1.0.0"
