# Repo Spec — <feature-name> (ScanToSMPL)

Applies the master spec to this repo's conventions and lists the exact skills the loop runs.

## Subpackages touched

- `scantosmpl/<subpkg>/` — <what changes>
- `scantosmpl/types.py` — <new fields / new dataclass>
- `scantosmpl/config.py` — <new config entries>
- `tests/test_<x>.py` — new unit tests
- `tests/integration/fixtures/<slug>/` — new fixture (if any)

## Coordinate frames + units

For any tensor this feature produces or consumes, name the frame and units. Silent mismatches are the P0 bug class.

| Tensor | Shape | Dtype | Frame | Units |
|--------|-------|-------|-------|-------|
| `…` | `(B, N, 3)` | `float32` | camera | metres |

## Determinism

List every RANSAC / stochastic step this feature introduces or touches. For each, name the seed source (config field, function kwarg) and confirm reproducibility.

## Verification

Skills the specialist(s) run per step (in order):

- `py-lint` — after any code change.
- `py-typecheck` — after any change to `scantosmpl/types.py` or a strict-annotated module.
- `py-test` — after each behaviour change (`pytest tests/test_<x>.py -v` per step, full suite at the end).
- `pipeline-smoke` — at the end of any component that could affect end-to-end behaviour.

## Definition of done

- All acceptance criteria in master §10 pass on the merged tree.
- `py-lint`, `py-typecheck`, `py-test` all green.
- `pipeline-smoke` on `tests/integration/fixtures/mini/` exits 0 with the expected artefacts.
- Any new adversarial fixtures pass (or gracefully degrade with a clear diagnostic).
- No P0 or P1 findings from the loop's `reviewer` remain open.
