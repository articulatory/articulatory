# -*- coding: utf-8 -*-

import logging
from re import L

import numpy as np
import torch

from articulatory.layers import GBlock
from articulatory.layers import PastFCEncoder
from articulatory.utils import read_hdf5


class GBlockGenerator(torch.nn.Module):
    """Generator module based on GAN-TTS vocoder."""

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        g_scales=(8, 8, 2, 2),
        g_kernel_sizes=(16, 16, 4, 4),
        use_weight_norm=True,
        use_ar=False, ar_input=512, ar_hidden=256, ar_output=128,
        use_tanh=True,
        use_spk_id=False,
        num_spk=None,
        spk_emb_size=32,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            resample_scales (list): List of upsampling scales.
            g_kernel_sizes (list): List of kernel sizes for upsampling layers.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        self.use_ar = use_ar
        self.use_spk_id = use_spk_id

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(g_scales) == len(g_kernel_sizes)

        # define modules
        self.num_resamples = len(g_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(
            in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        g_in_channels = [channels, channels, channels, channels//2, channels//2, channels//2, channels//2, channels//4, channels//4, channels//8]
        g_out_channels = [channels, channels, channels//2, channels//2, channels//2, channels//2, channels//4, channels//4, channels//8, channels//8]
        self.resamples = torch.nn.ModuleList()
        for i in range(len(g_kernel_sizes)):
            self.resamples += [
                GBlock(g_in_channels[i], g_out_channels[i], upsample=g_scales[i], kernel_size=g_kernel_sizes[i], norm=False)
            ]

        if use_tanh:
            self.output_conv = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    channels // 8,
                    out_channels,
                    kernel_size,
                    1,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.Tanh(),
            )
        else:
            self.output_conv = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    channels // 8,
                    out_channels,
                    kernel_size,
                    1,
                    padding=(kernel_size - 1) // 2,
                ),
            )

        if use_ar:
            self.ar_model = PastFCEncoder(input_len=ar_input, hidden_dim=ar_hidden, output_dim=ar_output)
        if use_spk_id:
            assert num_spk is not None
            self.spk_emb_mat = torch.nn.Embedding(num_spk, spk_emb_size)
            self.spk_fc = torch.nn.Linear(spk_emb_size, in_channels)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c, spk_id=None, ar=None):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, input_dim, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        if self.use_ar:
            ar_feats = self.ar_model(ar)  # (batchsize, ar_output)
            ar_feats = ar_feats.unsqueeze(2).repeat(1, 1, c.shape[2])  # (batchsize, ar_output, length)
            c = torch.cat((c, ar_feats), dim=1)
        if self.use_spk_id:
            spk_emb = self.spk_emb_mat(spk_id)  # (batchsize, spk_emb_size)
            spk_emb = self.spk_fc(spk_emb)  # (batchsize, in_channels)
            spk_emb = spk_emb.unsqueeze(2).repeat(1, 1, c.shape[2])  # (batchsize, in_channels, length)
            c = c + spk_emb
        c = self.input_conv(c)
        for i in range(self.num_resamples):
            c = self.resamples[i](c)
        c = self.output_conv(c)  # (batch_size, 1, input_len*final_scale)
        return c

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def inference(self, c, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(resample_scales), out_channels).

        """
        c = c.unsqueeze(1)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.forward(c.transpose(1, 0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)

    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        if stats.endswith(".h5"):
            mean = read_hdf5(stats, "mean").reshape(-1)
            scale = read_hdf5(stats, "scale").reshape(-1)
        else:
            mean = np.load(stats)[0].reshape(-1)
            scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        logging.info("Successfully registered stats as buffer.")
