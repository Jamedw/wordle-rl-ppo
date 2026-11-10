"""
Wordle as a Gymnasium-style RL environment.

MDP Formulation
===============
State:  Derived knowledge about the puzzle, not raw guess history.
        - letter_position_mask (5×26): binary — is letter c still possible at position p?
        - letter_status (26×3): one-hot — [unknown, confirmed_in_word, eliminated]
        - turn_number (1,): normalized scalar 0..1
        Total: 5*26 + 26*3 + 1 = 209 dimensions (flattened).

Action: Index into the word list (discrete, |V| choices).

Reward: Shaped reward encouraging fewer guesses:
        - Solve: +10 minus turns used (so turn-1 solve = +9, turn-6 solve = +4)
        - Per guess that doesn't solve: -1
        - Fail after 6 guesses: -10

Transition: Deterministic given hidden target word.
            Stochasticity comes from the target being sampled at episode start.

Why derived-knowledge state instead of raw history?
    The raw 6×5×29 history encodes the same information but forces the network
    to re-derive what's possible every forward pass.  The knowledge state is a
    sufficient statistic (it captures everything the agent needs to decide)
    and is much smaller, which helps learning.
"""

import numpy as np
from word_list import get_word_list


class WordleEnv:
    def __init__(self, word_list: list[str] | None = None):
        self.word_list = word_list or get_word_list()
        self.word_to_idx = {w: i for i, w in enumerate(self.word_list)}
        self.max_turns = 6

        self.state_dim = 5 * 26 + 26 * 3 + 1  # 209
        self.action_dim = len(self.word_list)

        self._target: str = ""
        self._turn: int = 0
        self._done: bool = False
        self._letter_pos_mask = np.zeros((5, 26), dtype=np.float32)
        self._letter_status = np.zeros((26, 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, target: str | None = None) -> np.ndarray:
        """Start a new episode. Optionally fix the target word for evaluation."""
        self._target = target or self.word_list[np.random.randint(len(self.word_list))]
        self._turn = 0
        self._done = False
        self._letter_pos_mask = np.ones((5, 26), dtype=np.float32)
        self._letter_status = np.zeros((26, 3), dtype=np.float32)
        self._letter_status[:, 0] = 1.0  # all letters start as "unknown"
        return self._observe()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert not self._done, "Episode is over — call reset()"
        guess = self.word_list[action]
        feedback = self._compute_feedback(guess)
        self._update_knowledge(guess, feedback)
        self._turn += 1

        solved = all(f == 2 for f in feedback)
        failed = self._turn >= self.max_turns and not solved
        self._done = solved or failed

        if solved:
            reward = 10.0 - self._turn
        elif failed:
            reward = -10.0
        else:
            reward = -1.0

        info = {
            "solved": solved,
            "turn": self._turn,
            "guess": guess,
            "target": self._target if self._done else "?????",
            "feedback": feedback,
        }
        return self._observe(), reward, self._done, info

    def get_valid_action_mask(self) -> np.ndarray:
        """
        Returns a binary mask over the action space.

        A word is 'valid' if none of its letters violate known constraints.
        Using this as an action mask dramatically shrinks the effective action
        space and accelerates learning.
        """
        mask = np.ones(self.action_dim, dtype=np.float32)
        for idx, word in enumerate(self.word_list):
            for pos, ch in enumerate(word):
                c = ord(ch) - ord("a")
                if self._letter_pos_mask[pos, c] == 0:
                    mask[idx] = 0
                    break
        return mask

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        return np.concatenate([
            self._letter_pos_mask.flatten(),
            self._letter_status.flatten(),
            np.array([self._turn / self.max_turns], dtype=np.float32),
        ])

    @staticmethod
    def _compute_feedback(guess: str, target: str | None = None) -> list[int]:
        """
        Wordle feedback: 0 = gray, 1 = yellow, 2 = green.

        Two-pass algorithm handles duplicate letters correctly:
        pass 1 marks greens and removes matched target chars,
        pass 2 marks yellows from the remaining pool.
        """
        # Allow static usage for testing
        if target is None:
            raise ValueError("target is required")
        feedback = [0] * 5
        remaining = list(target)

        for i in range(5):
            if guess[i] == target[i]:
                feedback[i] = 2
                remaining[i] = None

        for i in range(5):
            if feedback[i] != 0:
                continue
            if guess[i] in remaining:
                feedback[i] = 1
                remaining[remaining.index(guess[i])] = None

        return feedback

    # Override for instance use (target stored internally)
    def _compute_feedback(self, guess: str) -> list[int]:
        feedback = [0] * 5
        remaining = list(self._target)

        for i in range(5):
            if guess[i] == self._target[i]:
                feedback[i] = 2
                remaining[i] = None

        for i in range(5):
            if feedback[i] != 0:
                continue
            if guess[i] in remaining:
                feedback[i] = 1
                remaining[remaining.index(guess[i])] = None

        return feedback

    def _update_knowledge(self, guess: str, feedback: list[int]) -> None:
        """
        Update the belief state with new feedback.

        Handling duplicate letters: if a letter appears twice in a guess and one
        instance is gray, we can only eliminate it from that position (not globally)
        if the other instance was yellow or green.
        """
        for i, (ch, fb) in enumerate(zip(guess, feedback)):
            c = ord(ch) - ord("a")

            if fb == 2:  # GREEN — correct letter, correct position
                self._letter_pos_mask[i, :] = 0
                self._letter_pos_mask[i, c] = 1
                self._letter_status[c] = [0, 1, 0]

            elif fb == 1:  # YELLOW — correct letter, wrong position
                self._letter_pos_mask[i, c] = 0
                self._letter_status[c] = [0, 1, 0]

            else:  # GRAY — letter not in word (or excess duplicate)
                letter_confirmed = self._letter_status[c, 1] == 1
                letter_in_this_guess = any(
                    guess[j] == ch and feedback[j] in (1, 2)
                    for j in range(5)
                    if j != i
                )
                if letter_confirmed or letter_in_this_guess:
                    # Duplicate case: only remove from this position
                    self._letter_pos_mask[i, c] = 0
                else:
                    # Fully eliminate letter from all positions
                    self._letter_pos_mask[:, c] = 0
                    self._letter_status[c] = [0, 0, 1]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def render(self, guess: str, feedback: list[int]) -> str:
        COLORS = {0: "\033[90m", 1: "\033[93m", 2: "\033[92m"}
        RESET = "\033[0m"
        return "".join(f"{COLORS[fb]}{ch.upper()}{RESET}" for ch, fb in zip(guess, feedback))
