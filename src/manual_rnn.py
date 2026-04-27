from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ManualRNNConfig:
    input_size: int
    hidden_size: int = 32
    output_size: int = 1
    learning_rate: float = 1e-3
    epochs: int = 25
    random_seed: int = 42


class ManualRNN:
    """Scaffold for a manually implemented many-to-one RNN regressor."""

    def __init__(self, config: ManualRNNConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.random_seed)

        self.W_xh = rng.normal(0.0, 0.05, size=(config.input_size, config.hidden_size))
        self.W_hh = rng.normal(0.0, 0.05, size=(config.hidden_size, config.hidden_size))
        self.b_h = np.zeros(config.hidden_size, dtype=float)

        self.W_hy = rng.normal(0.0, 0.05, size=(config.hidden_size, config.output_size))
        self.b_y = np.zeros(config.output_size, dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
        raise NotImplementedError(
            "Implement the manual RNN training loop here: forward pass, loss, "
            "backpropagation through time, and parameter updates."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Implement sequence inference using the final hidden state for regression."
        )

    def _forward_sequence(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Return hidden states and output for a single input sequence."
        )

    def _backward_sequence(
        self, sequence: np.ndarray, target: float, hidden_states: np.ndarray
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError(
            "Compute gradients for a single training example using BPTT."
        )

