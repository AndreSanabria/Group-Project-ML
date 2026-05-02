from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ManualRNNConfig:
    input_size: int
    hidden_size: int = 16
    output_size: int = 1
    learning_rate: float = 1e-3
    epochs: int = 10
    random_seed: int = 42
    gradient_clip: float = 1.0
    max_train_samples: int | None = 20_000


class ManualRNN:
    """Manually implemented many-to-one tanh RNN regressor."""

    def __init__(self, config: ManualRNNConfig) -> None:
        if config.input_size <= 0 or config.hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive integers.")
        if config.output_size != 1:
            raise ValueError("ManualRNN currently supports one scalar regression output.")
        if config.learning_rate <= 0 or config.epochs <= 0:
            raise ValueError("learning_rate and epochs must be positive.")
        if config.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive.")
        if config.max_train_samples is not None and config.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive when provided.")
        self.config = config
        rng = np.random.default_rng(config.random_seed)
        self.W_xh = rng.normal(0.0, 0.05, size=(config.input_size, config.hidden_size))
        self.W_hh = rng.normal(0.0, 0.05, size=(config.hidden_size, config.hidden_size))
        self.b_h = np.zeros(config.hidden_size, dtype=float)
        self.W_hy = rng.normal(0.0, 0.05, size=(config.hidden_size, config.output_size))
        self.b_y = np.zeros(config.output_size, dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
        X, y = self._validate_training_data(X, y)
        rng = np.random.default_rng(self.config.random_seed)
        history: dict[str, list[float]] = {"loss": []}

        for _ in range(self.config.epochs):
            sample_order = rng.permutation(X.shape[0])
            if self.config.max_train_samples is not None:
                sample_order = sample_order[: self.config.max_train_samples]
            squared_errors: list[float] = []
            for sample_idx in sample_order:
                hidden_states, output = self._forward_sequence(X[sample_idx])
                error = float(output[0] - y[sample_idx])
                squared_errors.append(error**2)
                gradients = self._backward_sequence(
                    X[sample_idx], float(y[sample_idx]), hidden_states
                )
                self._clip_gradients(gradients)
                self._apply_gradients(gradients)

            history["loss"].append(float(np.mean(squared_errors)))

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_feature_tensor(X)
        predictions = np.zeros(X.shape[0], dtype=float)

        for sample_idx, sequence in enumerate(X):
            _, output = self._forward_sequence(sequence)
            predictions[sample_idx] = output[0]

        return predictions

    def _forward_sequence(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sequence = self._validate_sequence(sequence)
        time_steps = sequence.shape[0]
        hidden_states = np.zeros((time_steps + 1, self.config.hidden_size), dtype=float)

        for time_idx in range(time_steps):
            hidden_input = (
                sequence[time_idx] @ self.W_xh
                + hidden_states[time_idx] @ self.W_hh
                + self.b_h
            )
            hidden_states[time_idx + 1] = np.tanh(hidden_input)

        output = hidden_states[-1] @ self.W_hy + self.b_y
        return hidden_states, output

    def _backward_sequence(
        self, sequence: np.ndarray, target: float, hidden_states: np.ndarray
    ) -> dict[str, np.ndarray]:
        sequence = self._validate_sequence(sequence)
        if hidden_states.shape != (sequence.shape[0] + 1, self.config.hidden_size):
            raise ValueError(
                "hidden_states must include the initial state and one state per timestep."
            )

        gradients = {
            "W_xh": np.zeros_like(self.W_xh),
            "W_hh": np.zeros_like(self.W_hh),
            "b_h": np.zeros_like(self.b_h),
            "W_hy": np.zeros_like(self.W_hy),
            "b_y": np.zeros_like(self.b_y),
        }

        prediction = hidden_states[-1] @ self.W_hy + self.b_y
        d_output = prediction - float(target)
        gradients["W_hy"] += np.outer(hidden_states[-1], d_output)
        gradients["b_y"] += d_output
        d_hidden_next = self.W_hy @ d_output
        for time_idx in range(sequence.shape[0] - 1, -1, -1):
            hidden_state = hidden_states[time_idx + 1]
            previous_hidden_state = hidden_states[time_idx]
            d_raw_hidden = d_hidden_next * (1.0 - hidden_state**2)
            gradients["W_xh"] += np.outer(sequence[time_idx], d_raw_hidden)
            gradients["W_hh"] += np.outer(previous_hidden_state, d_raw_hidden)
            gradients["b_h"] += d_raw_hidden
            d_hidden_next = self.W_hh @ d_raw_hidden

        return gradients

    def _apply_gradients(self, gradients: dict[str, np.ndarray]) -> None:
        learning_rate = self.config.learning_rate
        self.W_xh -= learning_rate * gradients["W_xh"]
        self.W_hh -= learning_rate * gradients["W_hh"]
        self.b_h -= learning_rate * gradients["b_h"]
        self.W_hy -= learning_rate * gradients["W_hy"]
        self.b_y -= learning_rate * gradients["b_y"]

    def _clip_gradients(self, gradients: dict[str, np.ndarray]) -> None:
        total_norm = float(
            np.sqrt(sum(np.sum(gradient**2) for gradient in gradients.values()))
        )
        if total_norm <= self.config.gradient_clip:
            return
        scale = self.config.gradient_clip / (total_norm + 1e-12)
        for gradient in gradients.values():
            gradient *= scale

    def _validate_training_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X = self._validate_feature_tensor(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("Training data must contain at least one sample.")
        return X, y

    def _validate_feature_tensor(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("X must have shape (samples, timesteps, features).")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one timestep.")
        if X.shape[2] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} input features, found {X.shape[2]}."
            )
        return X

    def _validate_sequence(self, sequence: np.ndarray) -> np.ndarray:
        sequence = np.asarray(sequence, dtype=float)
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape (timesteps, features).")
        if sequence.shape[0] == 0:
            raise ValueError("sequence must contain at least one timestep.")
        if sequence.shape[1] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} input features, "
                f"found {sequence.shape[1]}."
            )
        return sequence

