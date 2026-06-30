# Agent Notes

- Keep the Python package name `qbdp`.
- Keep default runs CPU-compatible with PyTorch and synthetic data only.
- Keep PennyLane, Minari, Gymnasium, MuJoCo, RoboMimic, ManiSkill, and RLBench
  optional.
- RLBench uses the GitHub package plus CoppeliaSim/PyRep runtime environment
  variables; keep that path outside the default CPU synthetic workflow.
- Keep `scripts/rlbench_imitation.py` as an optional diagnostic; its current
  low-dimensional BC setup is a smoke experiment, not a solved RLBench method.
- Keep Gymnasium MuJoCo locomotion experiments behind the optional `mujoco`
  extra; do not make them part of the default CPU synthetic path.
- Prefer normal script entry points such as `python scripts/train.py` in docs
  and examples; keep `qbdp` as the internal package.
- After meaningful changes, run `python scripts/update_status.py --message "<change summary>"`.
- Verify with `python -m compileall qbdp tests` and `python -m pytest -q`.
- Add explicit tensor shape comments in model forward methods.
- Update `README.md`, `docs/REPORT.md`, `PROJECT_STATUS.md`, `AGENTS.md`, and `prompts/CODEX_PROMPT.md` as the project evolves.
