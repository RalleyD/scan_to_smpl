"""Clean SMPL .pkl files of Chumpy objects without installing chumpy.

Chumpy is a Python 2 autodiff library that SMPL model files were originally
serialized with. Modern Python 3 environments can't install it cleanly.
This script fakes the chumpy module hierarchy so pickle can reconstruct
the arrays, then re-saves them as plain numpy.

Usage:
    python -m scantosmpl.utils.clean_smpl models/smpl/ --output models/smpl/
"""

import copyreg
import pickle
import sys
import types
from pathlib import Path

import numpy as np


def _patch_chumpy():
    """Inject fake chumpy modules so pickle can resolve chumpy class references."""

    orig_reconstructor = copyreg._reconstructor

    def _patched_reconstructor(cls, base, state):
        if issubclass(cls, np.ndarray) and cls is not np.ndarray:
            return np.ndarray.__new__(cls, shape=(0,))
        return orig_reconstructor(cls, base, state)

    copyreg._reconstructor = _patched_reconstructor

    class FakeCh(np.ndarray):
        def __array_finalize__(self, obj):
            pass

        def __setstate__(self, state):
            if isinstance(state, tuple) and len(state) >= 5:
                super().__setstate__(state)
            elif isinstance(state, dict) and "x" in state:
                arr = np.asarray(state["x"])
                super().__setstate__(
                    (1, arr.shape, arr.dtype, arr.flags["F_CONTIGUOUS"], arr.tobytes())
                )

    for mod_name in [
        "chumpy",
        "chumpy.ch",
        "chumpy.utils",
        "chumpy.logic",
        "chumpy.reordering",
    ]:
        m = types.ModuleType(mod_name)
        for attr in ["Ch", "array", "Array", "row", "col", "vstack", "hstack"]:
            setattr(m, attr, FakeCh)
        sys.modules[mod_name] = m

    return orig_reconstructor


def _unpatch_chumpy(orig_reconstructor):
    """Remove fake chumpy modules and restore copyreg."""
    copyreg._reconstructor = orig_reconstructor
    for m in [k for k in sys.modules if k.startswith("chumpy")]:
        del sys.modules[m]


def clean_smpl_pkl(input_path: Path, output_path: Path) -> dict:
    """Load a SMPL .pkl containing chumpy objects and re-save as pure numpy.

    Returns the cleaned data dict.
    """
    orig = _patch_chumpy()
    try:
        with open(input_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        clean = {
            k: np.asarray(v).copy() if isinstance(v, np.ndarray) else v
            for k, v in data.items()
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(clean, f)

        return clean
    finally:
        _unpatch_chumpy(orig)


# Standard SMPL v1.1.0 filenames -> canonical names
SMPL_NAME_MAP = {
    "basicmodel_f_lbs_10_207_0_v1.1.0.pkl": "SMPL_FEMALE.pkl",
    "basicmodel_m_lbs_10_207_0_v1.1.0.pkl": "SMPL_MALE.pkl",
    "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl": "SMPL_NEUTRAL.pkl",
}


def clean_directory(input_dir: Path, output_dir: Path | None = None):
    """Clean all SMPL .pkl files in a directory.

    Detects original SMPL filenames and renames to canonical form
    (SMPL_NEUTRAL.pkl, etc). Files already in canonical form are
    cleaned in place.
    """
    output_dir = output_dir or input_dir
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    pkl_files = list(input_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {input_dir}")
        return

    for src in pkl_files:
        out_name = SMPL_NAME_MAP.get(src.name, src.name)
        dst = output_dir / out_name
        print(f"Cleaning {src.name}...")
        data = clean_smpl_pkl(src, dst)

        shapedirs = data.get("shapedirs")
        v_template = data.get("v_template")
        sd_shape = shapedirs.shape if isinstance(shapedirs, np.ndarray) else "missing"
        vt_shape = v_template.shape if isinstance(v_template, np.ndarray) else "missing"
        print(f"  -> {out_name} ({len(data)} keys, shapedirs={sd_shape}, v_template={vt_shape})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean SMPL .pkl files of Chumpy objects")
    parser.add_argument("input_dir", type=Path, help="Directory containing SMPL .pkl files")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    clean_directory(args.input_dir, args.output)
