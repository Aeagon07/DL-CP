"""
Phase 4 — Action Space
=======================
Encodes/decodes the RL agent's staffing reallocation decisions.

Action space: MultiDiscrete([5, 5, 5, 5, 5, 5])
  - One delta per queue (6 queues)
  - Delta levels: [-2, -1, 0, +1, +2] agents

Budget constraint: total net change across all queues <= MAX_TOTAL_DELTA (3).
This prevents the agent from trivially overstaffing every queue.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

QUEUE_NAMES: List[str] = [
    "Document Review",
    "Compliance Check",
    "Payment Processing",
    "Customer Onboarding",
    "Risk Assessment",
    "Audit Preparation",
]

# Per-queue agent bounds (from Phase 3 DEFAULT_QUEUE_CONFIGS)
QUEUE_MIN_AGENTS: Dict[str, int] = {
    "Document Review":     3,
    "Compliance Check":    2,
    "Payment Processing":  4,
    "Customer Onboarding": 2,
    "Risk Assessment":     2,
    "Audit Preparation":   1,
}
QUEUE_MAX_AGENTS: Dict[str, int] = {
    "Document Review":     12,
    "Compliance Check":    8,
    "Payment Processing":  15,
    "Customer Onboarding": 10,
    "Risk Assessment":     8,
    "Audit Preparation":   6,
}
QUEUE_DEFAULT_AGENTS: Dict[str, int] = {
    "Document Review":     7,
    "Compliance Check":    5,
    "Payment Processing":  9,
    "Customer Onboarding": 6,
    "Risk Assessment":     5,
    "Audit Preparation":   3,
}

DELTAS: List[int] = [-2, -1, 0, +1, +2]
N_QUEUES: int = len(QUEUE_NAMES)
N_DELTA_LEVELS: int = len(DELTAS)
MAX_TOTAL_DELTA: int = 3   # net agent change budget per step


class ActionSpace:
    """
    Manages encoding, decoding, validation, and masking of RL actions.

    Action array shape: (N_QUEUES,) — each element is an index into DELTAS.
    E.g. [2, 2, 4, 2, 2, 2] → [0, 0, +2, 0, 0, 0] = add 2 agents to Payment Processing.
    """

    QUEUE_NAMES      = QUEUE_NAMES
    QUEUE_MIN_AGENTS = QUEUE_MIN_AGENTS
    QUEUE_MAX_AGENTS = QUEUE_MAX_AGENTS
    QUEUE_DEFAULT_AGENTS = QUEUE_DEFAULT_AGENTS
    DELTAS           = DELTAS
    N_QUEUES         = N_QUEUES
    N_DELTA_LEVELS   = N_DELTA_LEVELS
    MAX_TOTAL_DELTA  = MAX_TOTAL_DELTA

    # gymnasium MultiDiscrete nvec
    NVEC = np.array([N_DELTA_LEVELS] * N_QUEUES, dtype=np.int64)

    # ------------------------------------------------------------------
    @staticmethod
    def decode(action: np.ndarray) -> Dict[str, int]:
        """
        Map a MultiDiscrete action vector → {queue_name: agent_delta}.

        Args:
            action: np.ndarray of shape (N_QUEUES,), values in [0, N_DELTA_LEVELS)

        Returns:
            dict mapping queue name → delta (int in DELTAS)
        """
        return {
            QUEUE_NAMES[i]: DELTAS[int(action[i])]
            for i in range(N_QUEUES)
        }

    # ------------------------------------------------------------------
    @staticmethod
    def apply(
        deltas: Dict[str, int],
        current_agents: Dict[str, int],
    ) -> Dict[str, int]:
        """
        Apply agent deltas to current allocations, clamped to [min, max].

        Args:
            deltas:          {queue_name: delta}
            current_agents:  {queue_name: current_agent_count}

        Returns:
            {queue_name: new_agent_count}
        """
        new_agents = {}
        for q in QUEUE_NAMES:
            delta = deltas.get(q, 0)
            current = current_agents.get(q, QUEUE_DEFAULT_AGENTS[q])
            new_val = current + delta
            new_val = max(QUEUE_MIN_AGENTS[q], min(QUEUE_MAX_AGENTS[q], new_val))
            new_agents[q] = new_val
        return new_agents

    # ------------------------------------------------------------------
    @staticmethod
    def is_valid(
        action: np.ndarray,
        current_agents: Dict[str, int],
    ) -> bool:
        """
        Check if action is valid:
          1. Decoded deltas don't push any queue outside [min, max]
          2. Net total delta <= MAX_TOTAL_DELTA
        """
        deltas = ActionSpace.decode(action)
        net_delta = 0
        for q in QUEUE_NAMES:
            d = deltas.get(q, 0)
            new_val = current_agents.get(q, QUEUE_DEFAULT_AGENTS[q]) + d
            if new_val < QUEUE_MIN_AGENTS[q] or new_val > QUEUE_MAX_AGENTS[q]:
                return False
            net_delta += abs(d)
        return net_delta <= MAX_TOTAL_DELTA

    # ------------------------------------------------------------------
    @staticmethod
    def get_action_mask(current_agents: Dict[str, int]) -> np.ndarray:
        """
        Returns a boolean mask of shape (N_QUEUES, N_DELTA_LEVELS).
        True = action is valid for that (queue, delta_level) pair.
        Used for action masking in PPO to avoid illegal moves.
        """
        mask = np.ones((N_QUEUES, N_DELTA_LEVELS), dtype=bool)
        for qi, q in enumerate(QUEUE_NAMES):
            curr = current_agents.get(q, QUEUE_DEFAULT_AGENTS[q])
            for di, delta in enumerate(DELTAS):
                new_val = curr + delta
                if new_val < QUEUE_MIN_AGENTS[q] or new_val > QUEUE_MAX_AGENTS[q]:
                    mask[qi, di] = False
        return mask

    # ------------------------------------------------------------------
    @staticmethod
    def no_op_action() -> np.ndarray:
        """Returns the action that changes nothing (delta=0 for all queues)."""
        neutral_idx = DELTAS.index(0)
        return np.full(N_QUEUES, neutral_idx, dtype=np.int64)

    # ------------------------------------------------------------------
    @staticmethod
    def action_to_str(action: np.ndarray) -> str:
        """Human-readable description of an action."""
        deltas = ActionSpace.decode(action)
        parts = []
        for q, d in deltas.items():
            if d > 0:
                parts.append(f"+{d} → {q}")
            elif d < 0:
                parts.append(f"{d} → {q}")
        return ", ".join(parts) if parts else "No change"

    # ------------------------------------------------------------------
    @staticmethod
    def default_agents() -> Dict[str, int]:
        """Return baseline agent allocation dict."""
        return dict(QUEUE_DEFAULT_AGENTS)
