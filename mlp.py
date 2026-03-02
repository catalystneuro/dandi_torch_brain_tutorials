import numpy as np
import torch
import torch.nn as nn

from torch_brain.registry import ModalitySpec


class MLPNeuralDecoder(nn.Module):
    """MLP decoder from binned spike counts to a continuous behavioral signal.

    This model treats the full binned spike population as a flat feature vector 
    and learns a direct linear mapping through hidden layers.
    It is a useful baseline to compare against more expressive architectures, such as POYO.

    Args:
        num_units: Number of recorded units (input channels).
        bin_size: Spike bin width in seconds (input temporal resolution).
        sequence_length: Duration of each analysis window in seconds.
        readout_spec: :class:`torch_brain.registry.ModalitySpec` that specifies
            the output dimension (``readout_spec.dim``) and the dot-separated
            paths used to extract target values and their timestamps
            (``readout_spec.value_key`` and ``readout_spec.timestamp_key``).
        readout_sampling_rate: Native sampling rate of the target signal in Hz.
            Determines ``num_output_steps = int(sequence_length * readout_sampling_rate)``.
        hidden_dim: Width of the two hidden linear layers.
    """

    def __init__(
        self,
        num_units: int,
        bin_size: float,
        sequence_length: float,
        readout_spec: ModalitySpec,
        readout_sampling_rate: float,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.bin_size = bin_size
        self.readout_spec = readout_spec
        self.output_dim = readout_spec.dim
        self.num_input_bins = int(sequence_length / bin_size)
        self.num_output_steps = int(sequence_length * readout_sampling_rate)

        self.net = nn.Sequential(
            nn.Linear(self.num_input_bins * num_units, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_output_steps * self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batched binned spike matrix to behavioral predictions.

        Args:
            x: Tensor of shape ``(batch, num_input_bins, num_units)``.

        Returns:
            Tensor of shape ``(batch, num_output_steps, output_dim)``.
        """
        x = x.flatten(1)                                                     # (B, T_in*N)
        x = self.net(x)                                                      # (B, T_out*D)
        return x.reshape(-1, self.num_output_steps, self.output_dim)         # (B, T_out, D)

    def _bin_spikes(self, spikes, num_units: int) -> np.ndarray:
        """Bin spike timestamps into a ``(num_units, num_input_bins)`` array.

        Timestamps are made relative to the domain start so that the bin
        indices always fall within ``[0, num_input_bins)``.
        """
        rate = 1.0 / self.bin_size
        binned = np.zeros((num_units, self.num_input_bins), dtype=np.float32)
        # Make timestamps relative to the window start
        t_rel = spikes.timestamps - spikes.domain.start[0]
        bin_idx = np.floor(t_rel * rate).astype(int)
        # Clamp to valid range (handles floating-point edge cases at window boundary)
        bin_idx = np.clip(bin_idx, 0, self.num_input_bins - 1)
        np.add.at(binned, (spikes.unit_index, bin_idx), 1)
        return binned

    def tokenize(self, data) -> dict:
        """Convert a :class:`temporaldata.Data` slice into a model-ready dict.

        This method can be used directly as a dataset transform::

            dataset.transform = model.tokenize

        This code runs on CPU. Do not access GPU tensors inside this function.

        Returns:
            A dict with keys:

            - ``"model_inputs"`` → ``{"x": Tensor(num_input_bins, num_units)}``
            - ``"target_values"`` → ``Tensor(num_output_steps,)`` or
              ``Tensor(num_output_steps, output_dim)``
        """
        # Input: binned spikes (num_input_bins, num_units)
        x = self._bin_spikes(data.spikes, num_units=len(data.units)).T
        
        # Target: e.g. data.wheel_velocity.values
        # shape: (num_output_steps,) for 1-D signals, (num_output_steps, D) otherwise
        target = data
        for attr in self.readout_spec.value_key.split("."):
            target = getattr(target, attr)

        y = np.asarray(target, dtype=np.float32)

        # Guard against ±1 sample boundary mismatch from RegularTimeSeries slicing.
        # The sliced window may contain -1 or +1 sample instead of exactly num_output_steps
        y = y[:self.num_output_steps]                                        # truncate if too long
        if len(y) < self.num_output_steps:                                   # pad if too short
            y = np.pad(y, (0, self.num_output_steps - len(y)), mode='edge')  # repeat last value

        return {
            "model_inputs": {
                "x": torch.tensor(x, dtype=torch.float32),
            },
            "target_values": torch.tensor(y, dtype=torch.float32),
        }
