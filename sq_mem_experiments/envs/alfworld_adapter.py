"""ALFWorld environment adapter (TextWorld variant).

Requires:  pip install 'sq_mem_experiments[alfworld]'
           + ALFWorld data downloaded to $ALFWORLD_DATA
             (see https://github.com/alfworld/alfworld for setup).

This adapter wraps `alfworld.agents.environment.AlfredTWEnv` (text-only).
ALFWorld differs from ScienceWorld in three ways that matter for SQ-Mem:

1. **No `get_gold_action_sequence()` on the live env.**
   Gold paths live in per-task `traj_data.json` files next to each game
   file. We parse the `plan.low_actions` field to reconstruct an expert
   action sequence for memory building. If the file is missing the
   adapter falls back to no gold path for that task (random-only memory).

2. **Action space is per-state and called `admissible_commands`.**
   Returned in `info['admissible_commands']` after each reset/step. This
   is the analogue of ScienceWorld's `get_valid_action_object_combinations()`.

3. **Reward is binary at done.**
   `info['won']` is the success bit; intermediate steps usually have
   reward=0. We surface the same `step_reward` mechanism so the rest of
   the pipeline (return-to-go, calibration, intervention audit) works
   identically.

The adapter exposes the same surface as `ScienceWorldAdapter`:
`reset`, `step`, `get_valid_actions`, `close`, `task_goal`,
`gold_sequence`, and a `make_tasks` staticmethod.
"""
import glob
import importlib.util
import json
import os
from typing import Any

from sq_mem_experiments.envs.base import BaseEnv
from sq_mem_experiments.schema import TaskSpec

_alfworld_available: bool = importlib.util.find_spec("alfworld") is not None


def _load_alfworld_config() -> Any:
    """Load ALFWorld's default config dict, allowing override via $ALFWORLD_CONFIG.

    ALFWorld's helper for loading config has moved across versions
    (alfworld.agents.utils.misc.load_config in some, alfworld.utils.misc
    in others, removed entirely in newer releases). To stay version-agnostic
    we just read the YAML directly with PyYAML.
    """
    import yaml  # type: ignore[import]
    import alfworld  # type: ignore[import]

    # 1. Explicit override
    cfg_path = os.environ.get("ALFWORLD_CONFIG")
    candidates: list[str] = []
    if cfg_path:
        candidates.append(cfg_path)

    # 2. Common locations relative to the installed alfworld package
    alfworld_dir = os.path.dirname(alfworld.__file__)
    candidates.extend([
        os.path.join(alfworld_dir, "configs", "base_config.yaml"),
        os.path.join(alfworld_dir, "agents", "configs", "base_config.yaml"),
        os.path.join(alfworld_dir, "..", "configs", "base_config.yaml"),
    ])

    # 3. ALFWORLD_DATA/configs (sometimes shipped with the data download)
    data_root = os.environ.get("ALFWORLD_DATA")
    if data_root:
        candidates.append(os.path.join(data_root, "configs", "base_config.yaml"))

    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        "Could not find ALFWorld base_config.yaml. Tried:\n  "
        + "\n  ".join(os.path.abspath(c) for c in candidates)
        + "\nSet ALFWORLD_CONFIG=/path/to/base_config.yaml to override."
    )


def _unwrap_batch_list(value: Any) -> list[str]:
    """ALFWorld returns admissible_commands as either a flat list of strings or
    a list-of-lists (one inner list per batch slot). We always run batch_size=1,
    so the inner list is what we want."""
    if value is None:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return [str(x) for x in value]


def _unwrap_scalar(value: Any) -> Any:
    """Pull batch slot 0 from a possibly-batched scalar field."""
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return value[0]
    return value


def _resolve_alfred_tw_env_cls() -> Any:
    """Locate the AlfredTWEnv class across ALFWorld package layouts.

    The class has moved at least three times:
      - older releases: alfworld.agents.environment.AlfredTWEnv
      - some releases: alfworld.agents.environment.alfred_tw_env.AlfredTWEnv
      - newer releases: alfworld.env.alfred_tw_env.AlfredTWEnv
    Try each import path; raise a clear error listing what was tried.
    """
    import importlib
    candidates = [
        ("alfworld.env.alfred_tw_env", "AlfredTWEnv"),
        ("alfworld.agents.environment.alfred_tw_env", "AlfredTWEnv"),
        ("alfworld.agents.environment", "AlfredTWEnv"),
    ]
    tried: list[str] = []
    for module_path, cls_name in candidates:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            tried.append(f"{module_path} (import failed)")
            continue
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            return cls
        tried.append(f"{module_path}.{cls_name} (attribute missing)")
    raise ImportError(
        "Could not locate AlfredTWEnv in the installed alfworld package. Tried:\n  "
        + "\n  ".join(tried)
        + "\nYour alfworld version may have moved the class. "
        "Check `python -c 'import alfworld; print(alfworld.__version__)'` "
        "and the package's __init__.py for the correct path."
    )


def _extract_gold_actions(gamefile_path: str) -> list[str]:
    """Read traj_data.json next to the gamefile and return the expert action list.

    ALFWorld's traj_data.json stores the high-level plan under `plan.high_pddl`
    and the executed low-level actions under `plan.low_actions`. The text-game
    accepts commands resembling the low-level actions, e.g.::

        "go to drawer 1", "open drawer 1", "take book 1 from drawer 1"

    These are reconstructed from `plan.low_actions[*].api_action`. If the file
    is missing or malformed we return an empty list and the caller falls back
    to random memory building for that task.
    """
    traj_path = os.path.join(os.path.dirname(gamefile_path), "traj_data.json")
    if not os.path.exists(traj_path):
        return []
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    low_actions = data.get("plan", {}).get("low_actions", [])
    out: list[str] = []
    for la in low_actions:
        api = la.get("api_action") or {}
        action = api.get("action")
        if not action:
            continue
        # Map ALFWorld's PDDL-flavoured api_action to natural-language commands.
        # The text-game accepts many surface forms; we use the simplest
        # consistent mapping.
        obj_id = api.get("objectId", "")
        recv_id = api.get("receptacleObjectId", "")
        if action == "GotoLocation":
            out.append(f"go to {recv_id or obj_id}")
        elif action == "PickupObject":
            out.append(f"take {obj_id} from {recv_id}" if recv_id else f"take {obj_id}")
        elif action == "PutObject":
            out.append(f"put {obj_id} in {recv_id}")
        elif action == "OpenObject":
            out.append(f"open {obj_id}")
        elif action == "CloseObject":
            out.append(f"close {obj_id}")
        elif action == "ToggleObjectOn":
            out.append(f"turn on {obj_id}")
        elif action == "ToggleObjectOff":
            out.append(f"turn off {obj_id}")
        elif action == "SliceObject":
            out.append(f"slice {obj_id}")
        elif action == "HeatObject":
            out.append(f"heat {obj_id}")
        elif action == "CoolObject":
            out.append(f"cool {obj_id}")
        elif action == "CleanObject":
            out.append(f"clean {obj_id}")
        elif action == "ExamineObject":
            out.append(f"examine {obj_id}")
        else:
            # Unknown action verb — record verb+object so retrieval can still match
            out.append(f"{action.lower()} {obj_id}".strip())
    return out


def _list_split_gamefiles(data_root: str, split: str) -> list[str]:
    """Enumerate ALFWorld gamefiles under a split directory deterministically.

    `split` is one of "train", "valid_seen", "valid_unseen". The returned
    list is sorted by path so the same code on the same data produces the
    same task ordering across runs.
    """
    pattern = os.path.join(data_root, "json_2.1.1", split, "**", "game.tw-pddl")
    return sorted(glob.glob(pattern, recursive=True))


class ALFWorldAdapter(BaseEnv):
    """Adapter for ALFWorld text games (AlfredTWEnv)."""

    def __init__(
        self,
        reward_mode: str = "score_delta",
        generate_gold_path: bool = False,
    ) -> None:
        if not _alfworld_available:
            raise ImportError(
                "alfworld is not installed. "
                "Run: pip install 'sq_mem_experiments[alfworld]'"
            )
        from typing import cast

        cfg = cast(Any, _load_alfworld_config())
        AlfredTWEnv = _resolve_alfred_tw_env_cls()
        # train_eval controls a single thing in AlfredTWEnv: whether to attach
        # the AlfredExpert wrapper (alfred_tw_env.py:262 — `expert_plan =
        # (train_eval == "train")`). The expert runs ALFWorld's handcoded
        # expert on every step via _gather_infos(), which contains an upstream
        # bug (handcoded_expert.py:268-270: substring-vs-class-match
        # disagreement → random.choice([])). Under self_rollout we ignore
        # info["extra.expert_plan"] entirely, so we should not pay for that
        # buggy path. Only request it when we actually need gold actions for
        # the gold-path memory builder.
        #
        # collect_game_files is tri-state: "train", "eval_in_distribution",
        # or "eval_out_of_distribution" (alfred_tw_env.py:137-142). Anything
        # else leaves `data_path` unbound. We use OOD here because the
        # gamefile is overridden per-task via self._env.gamefiles = [gamefile]
        # anyway — collect_game_files only needs to find *some* path to walk.
        if generate_gold_path:
            train_eval_mode = "train"
        else:
            train_eval_mode = "eval_out_of_distribution"
            # Ensure the OOD/ID dataset paths exist in the loaded config;
            # fall back to data_path if the config didn't ship with them.
            ds = cfg.setdefault("dataset", {})
            ds.setdefault("eval_ood_data_path", ds.get("data_path", ""))
            ds.setdefault("eval_id_data_path", ds.get("data_path", ""))
        tw_env = AlfredTWEnv(cfg, train_eval=train_eval_mode)
        self._env: Any = tw_env.init_env(batch_size=1)
        self.reward_mode = reward_mode
        self._generate_gold_path = generate_gold_path
        self._last_score: float = 0.0
        self._goal: str = ""
        self._valid_actions: list[str] = []
        self._gold_sequence: list[str] = []
        self._current_gamefile: str = ""
        self._last_info: dict[str, Any] = {}

    def reset(self, task_spec: TaskSpec) -> str:
        """Reset to the gamefile encoded in `task_spec.metadata["gamefile"]`."""
        self._last_score = 0.0
        self._valid_actions = []
        self._gold_sequence = []
        gamefile = str(task_spec.metadata.get("gamefile", ""))
        if not gamefile:
            raise ValueError(
                f"ALFWorld task {task_spec.task_id!r} has no 'gamefile' in metadata; "
                "use ALFWorldAdapter.make_tasks() to construct TaskSpecs."
            )
        self._current_gamefile = gamefile
        # Force the env to load THIS specific gamefile. AlfredTWEnv's init_env
        # exposes the underlying batched textworld env; we point its game-file
        # list at our single gamefile so reset() returns this task.
        try:
            self._env.gamefiles = [gamefile]  # type: ignore[attr-defined]
        except AttributeError:
            pass
        obs_batch, info_batch = self._env.reset()
        obs = obs_batch[0] if isinstance(obs_batch, (list, tuple)) else obs_batch
        # info can be EITHER a list-of-dicts (one per batch slot) OR a dict whose
        # *values* are batched lists. Handle both shapes.
        if isinstance(info_batch, (list, tuple)):
            info: dict[str, Any] = dict(info_batch[0])
        else:
            info = dict(info_batch)
        self._goal = str(info.get("task_desc", "")) or self._extract_goal(str(obs))
        self._valid_actions = _unwrap_batch_list(info.get("admissible_commands"))
        self._last_info = info
        # ALFWorld's expert_plan is *streamed* — info["extra.expert_plan"] gives
        # only the next gold action, not the full sequence. So we accumulate it
        # via next_gold_action() as the rollout proceeds. We seed _gold_sequence
        # with the first action so callers that just read env.gold_sequence
        # after reset() see something non-empty.
        if self._generate_gold_path:
            first = self._peek_expert_action()
            self._gold_sequence = [first] if first else []
        return str(obs)

    @staticmethod
    def _extract_goal(obs: str) -> str:
        # Observations end with "Your task is to: <goal>"
        marker = "Your task is to:"
        if marker in obs:
            return obs.split(marker, 1)[1].strip()
        return ""

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        obs_batch, reward_batch, done_batch, info_batch = self._env.step([action])
        obs = _unwrap_scalar(obs_batch)
        reward = _unwrap_scalar(reward_batch)
        done = _unwrap_scalar(done_batch)
        if isinstance(info_batch, (list, tuple)):
            info: dict[str, Any] = dict(info_batch[0])
        else:
            info = dict(info_batch)
        # ALFWorld text env normally gives reward=0 except at done; surface a
        # 0–1 score so the rest of the pipeline (return-to-go, calibration,
        # intervention audit) can treat it uniformly with ScienceWorld.
        won = bool(_unwrap_scalar(info.get("won", False)))
        score = 1.0 if won else 0.0
        if self.reward_mode == "score_delta":
            step_reward = score - self._last_score
        else:
            step_reward = float(reward)
        self._last_score = score
        info["score"] = score
        info["normalized_score"] = score
        info["won"] = won
        self._valid_actions = _unwrap_batch_list(info.get("admissible_commands"))
        self._last_info = info
        return str(obs), step_reward, bool(done), info

    def _peek_expert_action(self) -> str | None:
        """Return ALFWorld's next gold action from the most recent info dict.

        ALFWorld surfaces `info["extra.expert_plan"]` as a batched list whose
        inner list contains only the next 1 (sometimes few) gold actions.
        We take the first one.
        """
        for key in ("expert_plan", "extra.expert_plan"):
            plan = self._last_info.get(key)
            if plan is None:
                continue
            # Outer level: per-batch list. Take batch slot 0.
            if isinstance(plan, list) and plan and isinstance(plan[0], list):
                inner = plan[0]
            else:
                inner = plan if isinstance(plan, list) else [plan]
            if inner:
                return str(inner[0])
        return None

    def next_gold_action(self) -> str | None:
        """Streamed gold-action interface used by `rollout_gold_episode`.

        Each call returns ALFWorld's current next-gold-action from
        `info["extra.expert_plan"]` and appends it to `gold_sequence` for
        debugging/decision-log purposes. Returns None when no further gold
        action is available (typically because the episode is done).
        """
        action = self._peek_expert_action()
        if action is None:
            return None
        self._gold_sequence.append(action)
        return action

    def get_valid_actions(self) -> list[str]:
        return list(self._valid_actions)

    def close(self) -> None:
        try:
            self._env.close()
        except AttributeError:
            pass

    @property
    def task_goal(self) -> str:
        return self._goal

    @property
    def gold_sequence(self) -> list[str]:
        return list(self._gold_sequence)

    @property
    def current_score(self) -> float:
        return self._last_score

    @staticmethod
    def make_tasks(
        data_root: str,
        train_split: str = "train",
        test_split: str = "valid_unseen",
        n_train: int = 15,
        n_test: int = 15,
        task_types: list[str] | None = None,
    ) -> tuple[list[TaskSpec], list[TaskSpec]]:
        """Enumerate gamefiles under each split and build TaskSpecs.

        `data_root` is `$ALFWORLD_DATA` (or wherever the `json_2.1.1` tree lives).
        `task_types` optionally restricts to a subset of ALFWorld task types
        (e.g. ["look_at_obj_in_light", "pick_and_place_simple"]). The task
        type is read from the gamefile path: `…/<split>/<task_type>/<trial>/`.
        """
        train_files = _list_split_gamefiles(data_root, train_split)
        test_files = _list_split_gamefiles(data_root, test_split)
        if not train_files:
            raise FileNotFoundError(
                f"No ALFWorld gamefiles under {data_root}/json_2.1.1/{train_split}"
            )
        if not test_files:
            raise FileNotFoundError(
                f"No ALFWorld gamefiles under {data_root}/json_2.1.1/{test_split}"
            )

        if task_types:
            allowed = set(task_types)
            def _passes(g: str) -> bool:
                # gamefile path: …/json_2.1.1/<split>/<task_type>-...-<...>/<trial>/game.tw-pddl
                # ALFWorld task_type dirs prefix with one of TASK_TYPES values
                # then optional "-<obj>-<recep>-..." suffixes.
                parts = g.split(os.sep)
                try:
                    idx = parts.index("json_2.1.1")
                    type_dir = parts[idx + 2]
                except (ValueError, IndexError):
                    return False
                return any(type_dir.startswith(t) for t in allowed)
            train_files = [g for g in train_files if _passes(g)]
            test_files = [g for g in test_files if _passes(g)]
            if not train_files:
                raise FileNotFoundError(
                    f"No ALFWorld train games match task_types={list(allowed)} "
                    f"under {data_root}/json_2.1.1/{train_split}"
                )
            if not test_files:
                raise FileNotFoundError(
                    f"No ALFWorld test games match task_types={list(allowed)} "
                    f"under {data_root}/json_2.1.1/{test_split}"
                )

        def _task_id(gamefile: str, split: str) -> str:
            # gamefile path: …/json_2.1.1/<split>/<task_type>/<trial>/game.tw-pddl
            parts = gamefile.split(os.sep)
            try:
                idx = parts.index("json_2.1.1")
                trial = parts[idx + 3]  # split, task_type, trial
                task_type = parts[idx + 2]
            except (ValueError, IndexError):
                trial = os.path.basename(os.path.dirname(gamefile))
                task_type = "unknown"
            return f"{split}_{task_type}_{trial}"

        train_tasks = [
            TaskSpec(
                task_id=_task_id(g, "train"),
                env_name="alfworld",
                task_name=os.path.basename(os.path.dirname(os.path.dirname(g))),
                variation_id=i,
                split="train",
                metadata={"gamefile": g},
            )
            for i, g in enumerate(train_files[:n_train])
        ]
        test_tasks = [
            TaskSpec(
                task_id=_task_id(g, "test"),
                env_name="alfworld",
                task_name=os.path.basename(os.path.dirname(os.path.dirname(g))),
                variation_id=i,
                split="test",
                metadata={"gamefile": g},
            )
            for i, g in enumerate(test_files[:n_test])
        ]
        return train_tasks, test_tasks
