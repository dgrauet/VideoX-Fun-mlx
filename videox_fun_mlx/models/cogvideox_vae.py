# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VAE foundation layers for CogVideoX-Fun, ported to MLX.

All tensors use channels-last layout: (B, D, H, W, C).
Original PyTorch code uses channels-first: (B, C, D, H, W).
"""

from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_ops.spatial import upsample_nearest


class CogVideoXCausalConv3d(nn.Module):
    """A 3D causal convolution layer that pads the input tensor to ensure causality.

    Args:
        in_channels: Number of channels in the input tensor.
        out_channels: Number of output channels produced by the convolution.
        kernel_size: Kernel size of the convolutional kernel.
        stride: Stride of the convolution.
        dilation: Dilation rate of the convolution.
        pad_mode: Padding mode ("constant" or "replicate").
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.pad_mode = pad_mode
        self.height_pad = (height_kernel_size - 1) // 2
        self.width_pad = (width_kernel_size - 1) // 2
        self.time_pad = time_kernel_size - 1
        self.time_kernel_size = time_kernel_size

        stride = stride if isinstance(stride, tuple) else (stride, 1, 1)
        dilation_tuple = (dilation, 1, 1)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )

    def __call__(
        self, inputs: mx.array, conv_cache: Optional[mx.array] = None
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass.

        Args:
            inputs: (B, D, H, W, C) channels-last tensor.
            conv_cache: Optional cached temporal frames from previous call.

        Returns:
            Tuple of (output, conv_cache). conv_cache is None for replicate mode.
        """
        new_cache = None

        if self.pad_mode == "replicate":
            # Replicate-pad temporally by repeating the first frame
            if self.time_pad > 0:
                first = mx.repeat(inputs[:, :1], self.time_pad, axis=1)
                inputs = mx.concatenate([first, inputs], axis=1)
            # Spatial padding
            if self.height_pad > 0 or self.width_pad > 0:
                inputs = mx.pad(
                    inputs,
                    [
                        (0, 0),
                        (0, 0),
                        (self.height_pad, self.height_pad),
                        (self.width_pad, self.width_pad),
                        (0, 0),
                    ],
                )
        else:
            # Constant pad mode with cache support
            if self.time_kernel_size > 1:
                if conv_cache is not None:
                    cached = [conv_cache]
                else:
                    cached = [mx.repeat(inputs[:, :1], self.time_pad, axis=1)]
                inputs = mx.concatenate(cached + [inputs], axis=1)

            new_cache = (
                inputs[:, -self.time_kernel_size + 1 :]
                if self.time_kernel_size > 1
                else None
            )

            # Spatial padding
            if self.height_pad > 0 or self.width_pad > 0:
                inputs = mx.pad(
                    inputs,
                    [
                        (0, 0),
                        (0, 0),
                        (self.height_pad, self.height_pad),
                        (self.width_pad, self.width_pad),
                        (0, 0),
                    ],
                )

        output = self.conv(inputs)
        return output, new_cache


class CogVideoXSpatialNorm3D(nn.Module):
    """Spatially conditioned normalization for 3D video data.

    See https://arxiv.org/abs/2209.09002.

    Args:
        f_channels: Number of channels for input to group norm and output.
        zq_channels: Number of channels for the quantized vector.
        groups: Number of groups for group normalization.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(
            num_groups=groups, dims=f_channels, pytorch_compatible=True
        )
        self.conv_y = CogVideoXCausalConv3d(
            zq_channels, f_channels, kernel_size=1, stride=1
        )
        self.conv_b = CogVideoXCausalConv3d(
            zq_channels, f_channels, kernel_size=1, stride=1
        )

    def __call__(
        self,
        f: mx.array,
        zq: mx.array,
        conv_cache: Optional[Dict[str, mx.array]] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Forward pass.

        Args:
            f: Feature tensor (B, D, H, W, C).
            zq: Quantized tensor (B, D', H', W', C_zq).
            conv_cache: Optional dict of conv caches.

        Returns:
            Tuple of (output, new_conv_cache).
        """
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        # Resize zq to match f's spatial/temporal dimensions
        # In NDHWC layout, spatial dims are indices 1,2,3
        f_d, f_h, f_w = f.shape[1], f.shape[2], f.shape[3]

        if f_d > 1 and f_d % 2 == 1:
            # Split first frame and rest, resize separately
            f_first_shape = (1, f_h, f_w)
            f_rest_shape = (f_d - 1, f_h, f_w)

            zq_first = zq[:, :1]
            zq_rest = zq[:, 1:]

            zq_first = _interpolate_3d(zq_first, f_first_shape)
            zq_rest = _interpolate_3d(zq_rest, f_rest_shape)
            zq = mx.concatenate([zq_first, zq_rest], axis=1)
        else:
            zq = _interpolate_3d(zq, (f_d, f_h, f_w))

        conv_y, new_conv_cache["conv_y"] = self.conv_y(
            zq, conv_cache=conv_cache.get("conv_y")
        )
        conv_b, new_conv_cache["conv_b"] = self.conv_b(
            zq, conv_cache=conv_cache.get("conv_b")
        )

        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
        return new_f, new_conv_cache


def _interpolate_3d(x: mx.array, target_shape: Tuple[int, int, int]) -> mx.array:
    """Nearest-neighbor interpolation for 3D (NDHWC) tensors to target (D, H, W).

    Args:
        x: Input tensor (B, D, H, W, C).
        target_shape: Target (D, H, W).

    Returns:
        Interpolated tensor (B, target_D, target_H, target_W, C).
    """
    target_d, target_h, target_w = target_shape
    B, D, H, W, C = x.shape

    if D == target_d and H == target_h and W == target_w:
        return x

    # Temporal interpolation (nearest neighbor)
    if D != target_d:
        indices = mx.arange(target_d) * D // target_d
        indices = mx.clip(indices, 0, D - 1)
        x = x[:, indices]

    # Spatial interpolation: reshape to (B*D, H, W, C), upsample, reshape back
    if H != target_h or W != target_w:
        B_new = x.shape[0]
        D_new = x.shape[1]
        x = x.reshape(B_new * D_new, x.shape[2], x.shape[3], C)
        # Nearest-neighbor spatial resize
        if H != target_h or W != target_w:
            h_indices = mx.arange(target_h) * H // target_h
            h_indices = mx.clip(h_indices, 0, H - 1)
            w_indices = mx.arange(target_w) * W // target_w
            w_indices = mx.clip(w_indices, 0, W - 1)
            x = x[:, h_indices][:, :, w_indices]
        x = x.reshape(B_new, D_new, target_h, target_w, C)

    return x


class CogVideoXUpsample3D(nn.Module):
    """A 3D upsampling layer for CogVideoX.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to input.
        compress_time: Whether to upsample the time dimension as well.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time: bool = False,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.compress_time = compress_time

    def __call__(self, inputs: mx.array) -> mx.array:
        """Forward pass.

        Args:
            inputs: (B, D, H, W, C) channels-last tensor.

        Returns:
            Upsampled tensor (B, D', H', W', C_out).
        """
        if self.compress_time:
            if inputs.shape[1] > 1 and inputs.shape[1] % 2 == 1:
                # Split first frame, upsample separately
                x_first = inputs[:, 0]  # (B, H, W, C)
                x_rest = inputs[:, 1:]  # (B, D-1, H, W, C)

                x_first = upsample_nearest(x_first, scale_factor=2)  # (B, 2H, 2W, C)
                B, D_rest, H_rest, W_rest, C = x_rest.shape
                x_rest = x_rest.reshape(B * D_rest, H_rest, W_rest, C)
                x_rest = upsample_nearest(x_rest, scale_factor=2)
                x_rest = x_rest.reshape(B, D_rest, x_rest.shape[1], x_rest.shape[2], C)

                # Temporal upsample the rest by 2x
                x_rest = mx.repeat(x_rest, 2, axis=1)

                x_first = mx.expand_dims(x_first, axis=1)  # (B, 1, 2H, 2W, C)
                inputs = mx.concatenate([x_first, x_rest], axis=1)
            elif inputs.shape[1] > 1:
                # Full 3D upsample (spatial + temporal)
                inputs = upsample_nearest(inputs, scale_factor=2)
            else:
                # Single frame: spatial-only upsample
                x = inputs[:, 0]  # (B, H, W, C)
                x = upsample_nearest(x, scale_factor=2)
                inputs = mx.expand_dims(x, axis=1)
        else:
            # Spatial-only 2x upsample
            B, D, H, W, C = inputs.shape
            inputs = inputs.reshape(B * D, H, W, C)
            inputs = upsample_nearest(inputs, scale_factor=2)
            inputs = inputs.reshape(B, D, inputs.shape[1], inputs.shape[2], C)

        # Apply 2D conv to each frame
        B, D, H, W, C = inputs.shape
        inputs = inputs.reshape(B * D, H, W, C)
        inputs = self.conv(inputs)
        inputs = inputs.reshape(B, D, inputs.shape[1], inputs.shape[2], inputs.shape[3])

        return inputs


def _get_activation(name: str):
    """Get activation function by name."""
    if name in ("swish", "silu"):
        return nn.silu
    elif name == "mish":
        return nn.mish
    elif name == "gelu":
        return nn.gelu
    elif name == "relu":
        return nn.relu
    else:
        raise ValueError(f"Unknown activation: {name}")


class CogVideoXResnetBlock3D(nn.Module):
    """A 3D ResNet block used in the CogVideoX model.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (defaults to in_channels).
        dropout: Dropout rate.
        temb_channels: Number of time embedding channels.
        groups: Number of groups for group normalization.
        eps: Epsilon for normalization layers.
        non_linearity: Activation function name.
        conv_shortcut: Whether to use a convolution shortcut.
        spatial_norm_dim: Dimension for spatial norm (if used instead of group norm).
        pad_mode: Padding mode for causal convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        conv_shortcut: bool = False,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = _get_activation(non_linearity)
        self.use_conv_shortcut = conv_shortcut
        self.spatial_norm_dim = spatial_norm_dim

        if spatial_norm_dim is None:
            self.norm1 = nn.GroupNorm(
                num_groups=groups, dims=in_channels, pytorch_compatible=True
            )
            self.norm2 = nn.GroupNorm(
                num_groups=groups, dims=out_channels, pytorch_compatible=True
            )
        else:
            self.norm1 = CogVideoXSpatialNorm3D(
                f_channels=in_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
            )
            self.norm2 = CogVideoXSpatialNorm3D(
                f_channels=out_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
            )

        self.conv1 = CogVideoXCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
        )

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.conv2 = CogVideoXCausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CogVideoXCausalConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    pad_mode=pad_mode,
                )
            else:
                self.nin_shortcut = nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def __call__(
        self,
        inputs: mx.array,
        temb: Optional[mx.array] = None,
        zq: Optional[mx.array] = None,
        conv_cache: Optional[Dict[str, mx.array]] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Forward pass.

        Args:
            inputs: (B, D, H, W, C) tensor.
            temb: Optional time embedding (B, temb_channels).
            zq: Optional spatial norm conditioning tensor.
            conv_cache: Optional dict of conv caches.

        Returns:
            Tuple of (output, new_conv_cache).
        """
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states = inputs

        if zq is not None:
            hidden_states, new_conv_cache["norm1"] = self.norm1(
                hidden_states, zq, conv_cache=conv_cache.get("norm1")
            )
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states, new_conv_cache["conv1"] = self.conv1(
            hidden_states, conv_cache=conv_cache.get("conv1")
        )

        if temb is not None:
            # temb is (B, temb_channels), project and broadcast to (B, 1, 1, 1, C)
            hidden_states = hidden_states + self.temb_proj(
                self.nonlinearity(temb)
            ).reshape(temb.shape[0], 1, 1, 1, -1)

        if zq is not None:
            hidden_states, new_conv_cache["norm2"] = self.norm2(
                hidden_states, zq, conv_cache=conv_cache.get("norm2")
            )
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states, new_conv_cache["conv2"] = self.conv2(
            hidden_states, conv_cache=conv_cache.get("conv2")
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                inputs, new_conv_cache["conv_shortcut"] = self.conv_shortcut(
                    inputs, conv_cache=conv_cache.get("conv_shortcut")
                )
            else:
                inputs = self.nin_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states, new_conv_cache
