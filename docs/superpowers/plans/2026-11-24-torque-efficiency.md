# Torque Efficiency Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task-by-task with tests before implementation.

**Goal:** Provide a verifiable torque PPO experiment measuring motion efficiency.
**Architecture:** Separate Gymnasium torque environment reuses legacy model lookup/rendering, with a motor MJCF, substep passage detection, and independent physical cost accounting. Dedicated PPO training/evaluation CLI persists exact config and paired episode seeds.
**Tech Stack:** MuJoCo, Gymnasium, NumPy, Stable-Baselines3, pytest.
**Spec:** docs/superpowers/specs/2026-11-24-torque-efficiency.md

## Global Constraints
Feature branch only. Preserve legacy tasks. No large training artifacts committed. Success must mean whole-ball passage. PPO; two torque actions plus autonomous release.

## Review Focus
- Fixed timer or reattachment accidentally survives: test hold > 0.5s, release, reset.
- Plane-only/below-hoop false success: test complete, upward, outside and partial passages.
- Non-finite inputs and impossible target settings: fail before physics.
- Cost depends on frame count: integrate SI quantities at physics dt; test known torque.
- Evaluation selects easy successes: include per-target failures and nullable success-only statistics.

### Task 1: Torque environment
Files: envs/basketball_torque_env.py, assets/planar_torque.xml, tests/test_torque_env.py.
Interface: TorqueConfig; BasketballTorqueEnv(config, render_mode); reset(options={'target': xyz}); step(Box(3)) with episode metrics in info.
- [x] Write and run failing physical and scoring tests.
- [x] Implement motor actuation, release, seeded goals, substep whole-ball passage and costs.
- [x] Pass pytest and Gymnasium/SB3 checkers.

### Task 2: Reproducible experiments
Files: scripts/torque_experiment.py, tests/test_torque_experiment.py, docs/torque_efficiency.md, README.md.
Interface: train/evaluate CLI; baseline/time/efficient presets; exact config + checkpoints + evaluation JSON/CSV.
- [x] Test aggregation and real short training/save/reload/evaluate contract.
- [x] Implement PPO training and paired evaluation with independent metrics.
- [x] Document equations, units, commands, limitations and paper distinction.

### Task 3: Verification and PR
- [x] Run repository tests, headless learning smoke and rendering check.
- [x] Independent whole-branch review; fix important findings.
- [ ] Commit/push feature branch to fork and open PR against upstream main. Never merge.
