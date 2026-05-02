"""Validates IrisMA6TestEnv.get_aux_supervision() (ticket 031, slice 2).

The method is a pure READ on env state cached during _get_rewards. We
construct a small env, overwrite the cached state with known synthetic
values, then call get_aux_supervision() and assert correctness.

This avoids stepping the env (expensive + nondeterministic) while still
exercising the actual env class and its method.
"""
import traceback

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Triangulation result is a NamedTuple from the env's triangulation module.
from isaaclab_tasks.direct.iris_ma6.triangulation import TriangulationResult


def _build_env(num_envs=2):
    task = "Isaac-Iris-MA6-Direct-Test-v0"
    env_cfg = parse_env_cfg(task, device="cuda:0" if torch.cuda.is_available() else "cpu",
                             num_envs=num_envs)
    env = gym.make(task, cfg=env_cfg)
    return env


def _make_synthetic_tri_result(num_envs, position_per_env, is_valid_per_env, device):
    """Build a TriangulationResult with shape (N, 1, ...) — single target."""
    N = num_envs
    pos = torch.zeros(N, 1, 3, device=device)
    for i, p in enumerate(position_per_env):
        pos[i, 0, :] = torch.tensor(p, device=device)
    is_valid = torch.tensor(is_valid_per_env, dtype=torch.bool, device=device).unsqueeze(-1)  # (N, 1)
    return TriangulationResult(
        position=pos,
        covariance=torch.eye(3, device=device).expand(N, 1, 3, 3).contiguous(),
        quality_metric=torch.zeros(N, 1, device=device),
        is_valid=is_valid,
        condition_number=torch.zeros(N, 1, device=device),
        num_valid_cameras=torch.zeros(N, 1, device=device),
    )


def _compose_caches(num_envs, num_agents, gt_positions, scene_valid_per_env,
                    per_agent_bbox_nonempty, device):
    """Replicates the cache-population logic in _get_rewards. Used by the test
    to construct synthetic supervision state.

    `gt_positions` and `scene_valid_per_env` may be shorter than num_envs; the
    remainder is padded with zero positions and True scene-valid (so the
    test's assertions only need to inspect the explicitly-specified leading
    envs).
    """
    pos = torch.zeros(num_envs, 3, device=device)
    for i, p in enumerate(gt_positions[:num_envs]):
        pos[i, :] = torch.tensor(p, device=device)
    pos_cache = pos.unsqueeze(1).expand(-1, num_agents, -1).contiguous().to(dtype=torch.float32)

    # Pad scene_valid_per_env up to num_envs with True so unspecified trailing
    # envs don't add noise to the validity-composition assertions.
    full_scene_valid = list(scene_valid_per_env[:num_envs])
    full_scene_valid.extend([True] * (num_envs - len(full_scene_valid)))
    scene_valid = torch.tensor(full_scene_valid, dtype=torch.bool, device=device)      # (num_envs,)
    valid_cache = per_agent_bbox_nonempty & scene_valid.unsqueeze(-1)                  # (N, A)
    return pos_cache, valid_cache


def run_supervision_tests(results, device, verbose=False, existing_env=None):
    """Run the supervision contract tests.

    Args:
        existing_env: Optional pre-built env. When provided (e.g. by the
            combined integration runner), this function does NOT call
            gym.make() or env.close() — it reuses the caller's env. This
            sidesteps Isaac Sim USD-cache deadlocks that can occur when a
            second env is created in the same process under heavy contention.
    """
    env = None
    own_env = existing_env is None
    try:
        if own_env:
            env = _build_env(num_envs=2)
            unwrapped = env.unwrapped
        else:
            env = existing_env
            # The caller may pass either a wrapped env or the unwrapped
            # IrisMA6TestEnv directly. Resolve via the standard gymnasium
            # `.unwrapped` chain (idempotent on already-unwrapped envs).
            unwrapped = env
            while hasattr(unwrapped, "unwrapped") and unwrapped.unwrapped is not unwrapped:
                unwrapped = unwrapped.unwrapped
        N = unwrapped.num_envs
        A = unwrapped.cfg.num_agents
        env_device = unwrapped.device

        # ---- Case A: scene valid, per-agent bbox-nonempty mixed ----
        # Test data for envs 0 and 1 only; trailing envs (when N > 2 because we
        # share an env with the smoke pipeline) are padded with neutral
        # defaults: zero position, scene_valid=True, bbox_nonempty=False.
        gt_pos_w = [(10.0, 5.0, 2.0), (-3.0, 1.0, 4.0)]
        per_agent = torch.zeros(N, A, dtype=torch.bool, device=env_device)
        per_agent[0, 0] = True   # env 0, agent 0
        per_agent[1, 1] = True   # env 1, agent 1
        pos_cache, valid_cache = _compose_caches(
            N, A, gt_pos_w, [True] * N, per_agent, env_device,
        )
        unwrapped._aux_target_position_w_cache = pos_cache
        unwrapped._aux_target_valid_cache = valid_cache

        sup = unwrapped.get_aux_supervision()
        pos = sup["tri_target_position_w"]
        valid = sup["tri_target_valid"]

        assert tuple(pos.shape) == (N, A, 3), \
            f"position shape: expected ({N}, {A}, 3), got {tuple(pos.shape)}"
        assert tuple(valid.shape) == (N, A), \
            f"valid shape: expected ({N}, {A}), got {tuple(valid.shape)}"
        assert valid.dtype == torch.bool

        # Position broadcast: every agent in env 0 sees (10, 5, 2); env 1 sees (-3, 1, 4).
        for env_idx, expected in enumerate(gt_pos_w):
            for agent_idx in range(A):
                got = pos[env_idx, agent_idx, :].cpu().tolist()
                assert all(abs(g - e) < 1e-6 for g, e in zip(got, expected)), \
                    f"position broadcast wrong at env={env_idx} agent={agent_idx}: got={got}, expected={expected}"

        # Validity composes correctly: scene_valid (T) AND per_agent_nonempty.
        assert valid[0, 0].item() is True
        assert valid[0, 1].item() is False
        assert valid[1, 0].item() is False
        assert valid[1, 1].item() is True
        results.add_pass("Position broadcast + validity composition (case A)")

        # ---- Case B: scene invalid -> all agents masked regardless of bbox ----
        all_nonempty = torch.ones(N, A, dtype=torch.bool, device=env_device)
        pos_cache, valid_cache = _compose_caches(
            N, A, gt_pos_w, [False] * N, all_nonempty, env_device,
        )
        unwrapped._aux_target_position_w_cache = pos_cache
        unwrapped._aux_target_valid_cache = valid_cache

        sup = unwrapped.get_aux_supervision()
        valid = sup["tri_target_valid"]
        assert not valid.any().item(), \
            f"scene-invalid should zero out all per-agent valid; got {valid}"
        results.add_pass("Scene-invalid masks all agents (case B)")

        # ---- Case C: scene valid, all agents see -> all True ----
        pos_cache, valid_cache = _compose_caches(
            N, A, gt_pos_w, [True] * N, all_nonempty, env_device,
        )
        unwrapped._aux_target_position_w_cache = pos_cache
        unwrapped._aux_target_valid_cache = valid_cache

        sup = unwrapped.get_aux_supervision()
        valid = sup["tri_target_valid"]
        assert valid.all().item(), \
            f"all-true case should give all True valid; got {valid}"
        results.add_pass("All-valid: every agent supervised (case C)")

        # ---- Case D: pre-step state (caches still None) -> zero-fill, all-False ----
        unwrapped._aux_target_position_w_cache = None
        unwrapped._aux_target_valid_cache = None
        sup = unwrapped.get_aux_supervision()
        assert sup["tri_target_position_w"].abs().sum().item() == 0.0
        assert not sup["tri_target_valid"].any().item()
        results.add_pass("Pre-step state returns zero-filled / all-False (case D)")

    except Exception:
        results.add_fail("World-frame supervision suite", traceback.format_exc())
    finally:
        if env is not None and own_env:
            try:
                env.close()
            except Exception:
                pass
