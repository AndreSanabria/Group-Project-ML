from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ManualLSTMConfig:
    input_size: int
    hidden_size: int = 32
    output_size: int = 1
    learning_rate: float = 1e-3
    epochs: int = 25
    random_seed: int = 42
    gradient_clip_value: float = 5.0
    max_train_samples: int | None = 20_000


class ManualLSTM:
    """Manually implemented many-to-one LSTM regressor."""

    def __init__(self, config: ManualLSTMConfig) -> None:
        if config.output_size != 1:
            raise ValueError("ManualLSTM currently supports only a single regression output.")
        if config.learning_rate <= 0 or config.epochs <= 0:
            raise ValueError("learning_rate and epochs must both be positive.")
        if config.max_train_samples is not None and config.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive when provided.")
        self.config = config
        rng = np.random.default_rng(config.random_seed)
        concat_size = config.input_size + config.hidden_size
        self._rng = rng
        self.W_f = rng.normal(0.0, 0.05, size=(concat_size, config.hidden_size))
        self.b_f = np.zeros(config.hidden_size, dtype=float)
        self.W_i = rng.normal(0.0, 0.05, size=(concat_size, config.hidden_size))
        self.b_i = np.zeros(config.hidden_size, dtype=float)
        self.W_c = rng.normal(0.0, 0.05, size=(concat_size, config.hidden_size))
        self.b_c = np.zeros(config.hidden_size, dtype=float)
        self.W_o = rng.normal(0.0, 0.05, size=(concat_size, config.hidden_size))
        self.b_o = np.zeros(config.hidden_size, dtype=float)
        self.W_hy = rng.normal(0.0, 0.05, size=(config.hidden_size, config.output_size))
        self.b_y = np.zeros(config.output_size, dtype=float)
        self._last_forward_cache: dict[str, np.ndarray] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
        features, targets = self._validate_training_inputs(X, y)
        history = {"train_loss": []}

        for _ in range(self.config.epochs):
            sample_indices = self._rng.permutation(features.shape[0])
            if self.config.max_train_samples is not None:
                sample_indices = sample_indices[: self.config.max_train_samples]
            epoch_loss = 0.0

            for sample_index in sample_indices:
                sequence = features[sample_index]
                target_value = float(targets[sample_index])
                hidden_states, cell_states, prediction = self._forward_sequence(sequence)
                prediction_value = float(prediction[0])
                error = prediction_value - target_value
                epoch_loss += 0.5 * (error**2)
                gradients = self._backward_sequence(
                    sequence, target_value, hidden_states, cell_states
                )
                self._apply_gradients(gradients)

            history["train_loss"].append(epoch_loss / sample_indices.shape[0])

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        features = self._validate_feature_batch(X)
        predictions = np.zeros(features.shape[0], dtype=float)

        for sample_index, sequence in enumerate(features):
            _, _, output = self._forward_sequence(sequence)
            predictions[sample_index] = float(output[0])

        return predictions

    def _forward_sequence(
        self, sequence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequence_array = self._validate_single_sequence(sequence)
        time_steps = sequence_array.shape[0]
        hidden_size = self.config.hidden_size
        concat_size = self.config.input_size + hidden_size

        hidden_states = np.zeros((time_steps, hidden_size), dtype=float)
        cell_states = np.zeros((time_steps, hidden_size), dtype=float)
        forget_gates = np.zeros((time_steps, hidden_size), dtype=float)
        input_gates = np.zeros((time_steps, hidden_size), dtype=float)
        candidate_states = np.zeros((time_steps, hidden_size), dtype=float)
        output_gates = np.zeros((time_steps, hidden_size), dtype=float)
        concat_inputs = np.zeros((time_steps, concat_size), dtype=float)

        hidden_prev = np.zeros(hidden_size, dtype=float)
        cell_prev = np.zeros(hidden_size, dtype=float)

        for time_index in range(time_steps):
            x_t = sequence_array[time_index]
            concat_vector = np.concatenate((x_t, hidden_prev))

            forget_gate = _sigmoid(concat_vector @ self.W_f + self.b_f)
            input_gate = _sigmoid(concat_vector @ self.W_i + self.b_i)
            candidate_state = np.tanh(concat_vector @ self.W_c + self.b_c)
            output_gate = _sigmoid(concat_vector @ self.W_o + self.b_o)

            cell_state = forget_gate * cell_prev + input_gate * candidate_state
            hidden_state = output_gate * np.tanh(cell_state)

            concat_inputs[time_index] = concat_vector
            forget_gates[time_index] = forget_gate
            input_gates[time_index] = input_gate
            candidate_states[time_index] = candidate_state
            output_gates[time_index] = output_gate
            cell_states[time_index] = cell_state
            hidden_states[time_index] = hidden_state

            hidden_prev = hidden_state
            cell_prev = cell_state

        output = hidden_states[-1] @ self.W_hy + self.b_y
        self._last_forward_cache = {
            "sequence": sequence_array,
            "concat_inputs": concat_inputs,
            "forget_gates": forget_gates,
            "input_gates": input_gates,
            "candidate_states": candidate_states,
            "output_gates": output_gates,
            "prediction": output.copy(),
        }

        return hidden_states, cell_states, output.copy()

    def _backward_sequence(
        self,
        sequence: np.ndarray,
        target: float,
        hidden_states: np.ndarray,
        cell_states: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if self._last_forward_cache is None:
            raise RuntimeError("No forward-pass cache is available for backpropagation.")

        sequence_array = self._validate_single_sequence(sequence)
        cache = self._last_forward_cache
        if cache["sequence"].shape != sequence_array.shape:
            raise ValueError("The backward sequence does not match the cached forward pass.")

        gradients = self._zero_gradients()
        prediction = cache["prediction"]
        target_vector = np.array([float(target)], dtype=float)

        output_gradient = prediction - target_vector
        gradients["W_hy"] += np.outer(hidden_states[-1], output_gradient)
        gradients["b_y"] += output_gradient
        hidden_gradient_next = self.W_hy @ output_gradient
        cell_gradient_next = np.zeros(self.config.hidden_size, dtype=float)

        for time_index in range(sequence_array.shape[0] - 1, -1, -1):
            cell_state = cell_states[time_index]
            cell_prev = (
                cell_states[time_index - 1]
                if time_index > 0
                else np.zeros(self.config.hidden_size, dtype=float)
            )
            forget_gate = cache["forget_gates"][time_index]
            input_gate = cache["input_gates"][time_index]
            candidate_state = cache["candidate_states"][time_index]
            output_gate = cache["output_gates"][time_index]
            concat_vector = cache["concat_inputs"][time_index]

            tanh_cell_state = np.tanh(cell_state)
            output_gate_gradient = hidden_gradient_next * tanh_cell_state
            output_pre_activation_gradient = (
                output_gate_gradient * output_gate * (1.0 - output_gate)
            )

            cell_gradient = (
                cell_gradient_next
                + hidden_gradient_next * output_gate * (1.0 - tanh_cell_state**2)
            )
            forget_gate_gradient = cell_gradient * cell_prev
            forget_pre_activation_gradient = (
                forget_gate_gradient * forget_gate * (1.0 - forget_gate)
            )

            input_gate_gradient = cell_gradient * candidate_state
            input_pre_activation_gradient = (
                input_gate_gradient * input_gate * (1.0 - input_gate)
            )

            candidate_state_gradient = cell_gradient * input_gate
            candidate_pre_activation_gradient = (
                candidate_state_gradient * (1.0 - candidate_state**2)
            )

            gradients["W_f"] += np.outer(concat_vector, forget_pre_activation_gradient)
            gradients["b_f"] += forget_pre_activation_gradient
            gradients["W_i"] += np.outer(concat_vector, input_pre_activation_gradient)
            gradients["b_i"] += input_pre_activation_gradient
            gradients["W_c"] += np.outer(concat_vector, candidate_pre_activation_gradient)
            gradients["b_c"] += candidate_pre_activation_gradient
            gradients["W_o"] += np.outer(concat_vector, output_pre_activation_gradient)
            gradients["b_o"] += output_pre_activation_gradient
            concat_gradient = (
                self.W_f @ forget_pre_activation_gradient
                + self.W_i @ input_pre_activation_gradient
                + self.W_c @ candidate_pre_activation_gradient
                + self.W_o @ output_pre_activation_gradient
            )

            hidden_gradient_next = concat_gradient[self.config.input_size :]
            cell_gradient_next = cell_gradient * forget_gate

        self._clip_gradients(gradients)
        return gradients

    def _zero_gradients(self) -> dict[str, np.ndarray]:
        return {
            "W_f": np.zeros_like(self.W_f),
            "b_f": np.zeros_like(self.b_f),
            "W_i": np.zeros_like(self.W_i),
            "b_i": np.zeros_like(self.b_i),
            "W_c": np.zeros_like(self.W_c),
            "b_c": np.zeros_like(self.b_c),
            "W_o": np.zeros_like(self.W_o),
            "b_o": np.zeros_like(self.b_o),
            "W_hy": np.zeros_like(self.W_hy),
            "b_y": np.zeros_like(self.b_y),
        }

    def _apply_gradients(self, gradients: dict[str, np.ndarray]) -> None:
        learning_rate = self.config.learning_rate
        self.W_f -= learning_rate * gradients["W_f"]
        self.b_f -= learning_rate * gradients["b_f"]
        self.W_i -= learning_rate * gradients["W_i"]
        self.b_i -= learning_rate * gradients["b_i"]
        self.W_c -= learning_rate * gradients["W_c"]
        self.b_c -= learning_rate * gradients["b_c"]
        self.W_o -= learning_rate * gradients["W_o"]
        self.b_o -= learning_rate * gradients["b_o"]
        self.W_hy -= learning_rate * gradients["W_hy"]
        self.b_y -= learning_rate * gradients["b_y"]

    def _clip_gradients(self, gradients: dict[str, np.ndarray]) -> None:
        clip_value = self.config.gradient_clip_value
        if clip_value <= 0:
            return
        for gradient in gradients.values():
            np.clip(gradient, -clip_value, clip_value, out=gradient)

    def _validate_training_inputs(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        features = self._validate_feature_batch(X)
        targets = np.asarray(y, dtype=float).reshape(-1)
        if targets.shape[0] != features.shape[0]:
            raise ValueError(
                "The number of target values must match the number of input sequences. "
                f"Got {targets.shape[0]} targets for {features.shape[0]} sequences."
            )
        if not np.isfinite(targets).all():
            raise ValueError("Target values must all be finite.")
        return features, targets

    def _validate_feature_batch(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        if features.ndim != 3:
            raise ValueError(
                "Expected X to have shape (samples, time_steps, input_size). "
                f"Got array with shape {features.shape}."
            )
        if features.shape[2] != self.config.input_size:
            raise ValueError(
                "Input feature size does not match the model configuration. "
                f"Expected {self.config.input_size}, got {features.shape[2]}."
            )
        if features.shape[0] == 0:
            raise ValueError("At least one training sequence is required.")
        if not np.isfinite(features).all():
            raise ValueError("Input features must all be finite.")
        return features

    def _validate_single_sequence(self, sequence: np.ndarray) -> np.ndarray:
        sequence_array = np.asarray(sequence, dtype=float)
        if sequence_array.ndim != 2:
            raise ValueError(
                "Each input sequence must have shape (time_steps, input_size). "
                f"Got {sequence_array.shape}."
            )
        if sequence_array.shape[1] != self.config.input_size:
            raise ValueError(
                "Sequence feature size does not match the model configuration. "
                f"Expected {self.config.input_size}, got {sequence_array.shape[1]}."
            )
        if sequence_array.shape[0] == 0:
            raise ValueError("Sequences must contain at least one time step.")
        if not np.isfinite(sequence_array).all():
            raise ValueError("Sequence values must all be finite.")
        return sequence_array


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped_values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped_values))
