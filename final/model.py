# model.py
#================================================================================================================================================
# CNN-only inner model for local spatiotemporal prediction with variable input timestamp history.
#================================================================================================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    K_VALUES,
    NUM_CNN_LAYERS,
    NUM_PHYSICAL_CHANNELS,
    SPATIAL_KERNEL_SIZE,
    TAYLOR_ALPHA,
    TAYLOR_X0,
    TAYLOR_CLAMP_VALUE,
    HIDDEN_CHANNELS,
    TEMPORAL_KERNEL_SIZE,
)








#==============================================================================================================
# TAYLOR SERIES ACTIVATION
#==============================================================================================================

class TaylorActivation(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = TAYLOR_ALPHA
        self.x0 = TAYLOR_X0
        self.clamp_value = float(TAYLOR_CLAMP_VALUE)

        #------------------------------------------------------------------------------------------------------
        # sigma(z) = sum_{n = 0}^{alpha} theta_n / n! * (z - x0)^n
        # theta is trainable. theta[1] = 1 initializes the activation near identity when x0 = 0.
        #------------------------------------------------------------------------------------------------------

        theta = torch.zeros(self.alpha + 1)
        theta[1] = 1.0

        self.theta = nn.Parameter(theta)

        factorials = torch.tensor(
            [math.factorial(power) for power in range(self.alpha + 1)],
            dtype = torch.float32,
        )

        self.register_buffer("factorials", factorials)



    def forward(self, x):
        #------------------------------------------------------------------------------------------------------
        # Elementwise Taylor-series activation. Shape is unchanged.
        #------------------------------------------------------------------------------------------------------

        out = torch.zeros_like(x)

        #------------------------------------------------------------------------------------------------------
        # Clamp before forming powers. Without this, large but finite convolution features can overflow in
        # shifted_x ** power. Even when theta[power] is initialized to zero, PyTorch can still evaluate
        # 0 * inf as NaN. The clamp therefore preserves finite forward passes while keeping theta shared and
        # trainable.
        #------------------------------------------------------------------------------------------------------

        shifted_x = torch.clamp(
            x - self.x0,
            min = -self.clamp_value,
            max = self.clamp_value,
        )

        for power in range(self.alpha + 1):
            out = out + (
                self.theta[power] / self.factorials[power]
            ) * torch.pow(shifted_x, power)

        return out










#==============================================================================================================
# CNN-ONLY INNER MODEL
#==============================================================================================================

class InnerK(nn.Module):
    """
    CNN-only local spatiotemporal model.

    The shared CNN feature extractor mixes physical channels, local spatial structure, and temporal history.
    TaylorActivation layers inject nonlinearities between convolution layers.

    Because the learned final collapse uses a Conv3d kernel that spans the remaining time dimension, the final
    collapse head is k-specific. The convolutional feature extractor is shared across all k values in one run.
    """

    def _validate_input(self, x):
        #------------------------------------------------------------------------------------------------------
        # Expected input: [B, 4, k, s, s], where s = spatial_kernel_size.
        # If self.k is None, the model accepts any k in self.k_values.
        #------------------------------------------------------------------------------------------------------

        if x.ndim != 5:
            raise ValueError(
                f"Expected input with shape [B, 4, k, s, s], but got tensor with shape {x.shape}."
            )

        if x.shape[1] != self.num_physical_channels:
            raise ValueError(
                f"Expected channel dimension {self.num_physical_channels}, but got {x.shape[1]}."
            )

        if x.shape[2] < self.temporal_kernel_size:
            raise ValueError(
                f"Expected time dimension k >= {self.temporal_kernel_size}, but got k = {x.shape[2]}."
            )

        if self.k is not None and x.shape[2] != self.k:
            raise ValueError(
                f"This InnerK instance was initialized with k = {self.k}, but got input k = {x.shape[2]}."
            )

        if str(int(x.shape[2])) not in self.heads:
            raise ValueError(
                f"This InnerK instance has no final collapse head for k = {int(x.shape[2])}. "
                f"Available k values: {self.k_values}."
            )

        if x.shape[3] != self.spatial_kernel_size or x.shape[4] != self.spatial_kernel_size:
            raise ValueError(
                f"Expected spatial patch size "
                f"{self.spatial_kernel_size} x {self.spatial_kernel_size}, "
                f"but got {x.shape[3]} x {x.shape[4]}."
            )





    def _time_after_feature_extractor(self, k_value):
        #------------------------------------------------------------------------------------------------------
        # The first Conv3d uses no temporal padding, so k -> k - temporal_kernel_size + 1.
        # Later Conv3d layers use same-style temporal padding and preserve that reduced time length.
        #------------------------------------------------------------------------------------------------------

        return int(k_value) - self.temporal_kernel_size + 1





    def __init__(
        self,
        k = None,
        k_values = None,
    ):
        super().__init__()

        self.k = int(k) if k is not None else None
        self.num_physical_channels = NUM_PHYSICAL_CHANNELS
        self.spatial_kernel_size = SPATIAL_KERNEL_SIZE
        self.hidden_channels = HIDDEN_CHANNELS
        self.temporal_kernel_size = TEMPORAL_KERNEL_SIZE
        self.num_cnn_layers = NUM_CNN_LAYERS

        if k_values is None:
            if self.k is None:
                k_values = K_VALUES
            else:
                k_values = [self.k]

        self.k_values = sorted(set(int(k_value) for k_value in k_values))

        if self.k is not None and self.k not in self.k_values:
            self.k_values.append(self.k)
            self.k_values = sorted(set(self.k_values))

        #------------------------------------------------------------------------------------------------------
        # Basic checks
        #------------------------------------------------------------------------------------------------------

        if self.num_cnn_layers < 1:
            raise ValueError(
                f"NUM_CNN_LAYERS must be at least 1, but got {self.num_cnn_layers}."
            )

        if self.temporal_kernel_size % 2 == 0:
            raise ValueError(
                "TEMPORAL_KERNEL_SIZE must be odd so the later temporal-only Conv3d layers preserve k_out."
            )

        for k_value in self.k_values:
            if self.temporal_kernel_size > k_value:
                raise ValueError(
                    f"TEMPORAL_KERNEL_SIZE = {self.temporal_kernel_size} cannot be larger than k = {k_value}."
                )

        if self.spatial_kernel_size < 1:
            raise ValueError(
                f"SPATIAL_KERNEL_SIZE must be positive, but got {self.spatial_kernel_size}."
            )

        if self.num_physical_channels != 4:
            raise ValueError(
                f"This model assumes 4 physical channels, but NUM_PHYSICAL_CHANNELS = {self.num_physical_channels}."
            )

        temporal_padding = self.temporal_kernel_size // 2

        #------------------------------------------------------------------------------------------------------
        # Shared CNN feature extractor:
        #
        #     [B, 4, k, s, s]
        #         -> Conv3d + TaylorActivation
        #     [B, hidden_channels, k_out, 1, 1]
        #         -> temporal Conv3d + TaylorActivation blocks
        #     [B, hidden_channels, k_out, 1, 1]
        #
        # The first kernel spans the full local spatial patch. Later kernels keep the spatial size at 1 x 1
        # and continue mixing through the remaining temporal feature dimension.
        #------------------------------------------------------------------------------------------------------

        feature_layers = [
            nn.Conv3d(
                in_channels = self.num_physical_channels,
                out_channels = self.hidden_channels,
                kernel_size = (
                    self.temporal_kernel_size,
                    self.spatial_kernel_size,
                    self.spatial_kernel_size,
                ),
                padding = 0,
            ),
            TaylorActivation(),
        ]

        for _ in range(self.num_cnn_layers - 1):
            feature_layers.extend(
                [
                    nn.Conv3d(
                        in_channels = self.hidden_channels,
                        out_channels = self.hidden_channels,
                        kernel_size = (
                            self.temporal_kernel_size,
                            1,
                            1,
                        ),
                        padding = (
                            temporal_padding,
                            0,
                            0,
                        ),
                    ),
                    TaylorActivation(),
                ]
            )

        self.feature_extractor = nn.Sequential(*feature_layers)

        #------------------------------------------------------------------------------------------------------
        # Learned final Conv3d collapse heads:
        #
        #     [B, hidden_channels, k_out, 1, 1]
        #         -> k-specific final Conv3d
        #     [B, 4, 1, 1, 1]
        #         -> squeeze
        #     [B, 4]
        #
        # This is the CNN-only replacement for temporal pooling + MLP. It learns how to weight every remaining
        # temporal feature and hidden channel instead of averaging them.
        #------------------------------------------------------------------------------------------------------

        self.heads = nn.ModuleDict()

        for k_value in self.k_values:
            k_out = self._time_after_feature_extractor(
                k_value = k_value,
            )

            self.heads[str(k_value)] = nn.Conv3d(
                in_channels = self.hidden_channels,
                out_channels = self.num_physical_channels,
                kernel_size = (
                    k_out,
                    1,
                    1,
                ),
                padding = 0,
            )








    def forward(self, x):
        #------------------------------------------------------------------------------------------------------
        # Full model:
        #
        #     [B, 4, k, s, s]
        #         -> shared CNN feature extractor with Taylor nonlinearities
        #     [B, hidden_channels, k_out, 1, 1]
        #         -> learned k-specific Conv3d collapse head
        #     [B, 4, 1, 1, 1]
        #         -> squeeze
        #     [B, 4]
        #------------------------------------------------------------------------------------------------------

        self._validate_input(x)

        k_value = str(int(x.shape[2]))

        cnn_features = self.feature_extractor(x)

        prediction = self.heads[k_value](cnn_features)

        prediction = prediction.squeeze(-1).squeeze(-1).squeeze(-1)

        return prediction










#==============================================================================================================
# EXPONENTIAL VRMSE-CURVE SLOPE LOSS
#==============================================================================================================

class ExponentialSlopeLoss(nn.Module):
    def __init__(
        self,
        k_ref,
        slope_weight = 1.0,
        prediction_weight = 1.0,
        fit_steps = 25,
        fit_lr = 5.0e-2,
        eps = 1.0e-8,
    ):
        super().__init__()

        self.k_ref = float(k_ref)
        self.slope_weight = float(slope_weight)
        self.prediction_weight = float(prediction_weight)
        self.fit_steps = int(fit_steps)
        self.fit_lr = float(fit_lr)
        self.eps = float(eps)

        if self.fit_steps < 1:
            raise ValueError(
                f"fit_steps must be at least 1, but got {self.fit_steps}."
            )

        if self.fit_lr <= 0.0:
            raise ValueError(
                f"fit_lr must be positive, but got {self.fit_lr}."
            )





    def _curve_from_raw_parameters(
        self,
        k_values,
        raw_p,
        raw_A,
        k_shift,
        raw_w,
    ):
        #------------------------------------------------------------------------------------------------------
        # Evaluate:
        #
        #     E(k) = p + A exp(-w (k - k_shift))
        #
        # p, A, and w are made positive through exp. These are temporary fit variables, not trainable model
        # parameters. The exponent is clamped only to avoid numerical overflow during the inner curve fit.
        #------------------------------------------------------------------------------------------------------

        p = torch.exp(raw_p) + self.eps
        A = torch.exp(raw_A) + self.eps
        w = torch.exp(raw_w) + self.eps

        exponent = torch.clamp(
            -1.0 * w * (k_values - k_shift),
            min = -60.0,
            max = 60.0,
        )

        curve_values = p + A * torch.exp(exponent)

        return curve_values, p, A, k_shift, w





    def _initial_fit_variables(
        self,
        k_values,
        vrmse_values,
    ):
        #------------------------------------------------------------------------------------------------------
        # Initialize the temporary curve-fit variables from the current VRMSE(k) scale.
        #
        # The initialization is detached because these values are only starting guesses for the inner fit. The
        # fitted parameters still depend on the current VRMSE values through the differentiable fit iterations.
        #------------------------------------------------------------------------------------------------------

        dtype = vrmse_values.dtype
        device = vrmse_values.device

        detached_vrmse = torch.clamp(
            vrmse_values.detach(),
            min = self.eps,
        )

        min_vrmse = torch.min(detached_vrmse)
        max_vrmse = torch.max(detached_vrmse)

        p0 = torch.clamp(
            0.9 * min_vrmse,
            min = self.eps,
        )

        A0 = torch.clamp(
            max_vrmse - p0,
            min = self.eps,
        )

        k_min = torch.min(k_values.detach())
        k_max = torch.max(k_values.detach())
        k_span = torch.clamp(
            k_max - k_min,
            min = torch.tensor(1.0, device = device, dtype = dtype),
        )

        w0 = 1.0 / k_span
        k_shift0 = k_min

        raw_p = torch.log(p0).clone().detach().requires_grad_(True)
        raw_A = torch.log(A0).clone().detach().requires_grad_(True)
        raw_w = torch.log(w0).clone().detach().requires_grad_(True)
        k_shift = k_shift0.clone().detach().requires_grad_(True)

        return raw_p, raw_A, k_shift, raw_w





    def fit_curve_parameters(
        self,
        k_values,
        vrmse_values,
    ):
        #------------------------------------------------------------------------------------------------------
        # Refit p, A, k_shift, and w from the current VRMSE(k) values.
        #
        # This fit is used only to compute the exponential slope term. The fit error is not added to the
        # training loss. The manual gradient-descent steps are differentiable during training, so the final
        # slope term can still backpropagate to the CNN through the VRMSE values.
        #------------------------------------------------------------------------------------------------------

        k_values = k_values.to(
            device = vrmse_values.device,
            dtype = vrmse_values.dtype,
        )

        raw_p, raw_A, k_shift, raw_w = self._initial_fit_variables(
            k_values = k_values,
            vrmse_values = vrmse_values,
        )

        create_graph = bool(vrmse_values.requires_grad)

        scale = torch.clamp(
            torch.mean(torch.abs(vrmse_values.detach())),
            min = self.eps,
        )

        with torch.enable_grad():
            for _ in range(self.fit_steps):
                curve_values, _, _, _, _ = self._curve_from_raw_parameters(
                    k_values = k_values,
                    raw_p = raw_p,
                    raw_A = raw_A,
                    k_shift = k_shift,
                    raw_w = raw_w,
                )

                fit_objective = torch.mean(
                    ((curve_values - vrmse_values) / scale) ** 2
                )

                gradients = torch.autograd.grad(
                    fit_objective,
                    (raw_p, raw_A, k_shift, raw_w),
                    create_graph = create_graph,
                    retain_graph = create_graph,
                )

                raw_p = raw_p - self.fit_lr * gradients[0]
                raw_A = raw_A - self.fit_lr * gradients[1]
                k_shift = k_shift - self.fit_lr * gradients[2]
                raw_w = raw_w - self.fit_lr * gradients[3]

            curve_values, p, A, k_shift, w = self._curve_from_raw_parameters(
                k_values = k_values,
                raw_p = raw_p,
                raw_A = raw_A,
                k_shift = k_shift,
                raw_w = raw_w,
            )

        return curve_values, p, A, k_shift, w





    def exponential_slope(
        self,
        k_value,
        A,
        k_shift,
        w,
    ):
        #------------------------------------------------------------------------------------------------------
        # Evaluate dE/dk for:
        #
        #     E(k) = p + A exp(-w (k - k_shift))
        #
        #     dE/dk = -A w exp(-w (k - k_shift))
        #------------------------------------------------------------------------------------------------------

        exponent = torch.clamp(
            -1.0 * w * (k_value - k_shift),
            min = -60.0,
            max = 60.0,
        )

        slope = -1.0 * A * w * torch.exp(exponent)

        return slope





    def forward(
        self,
        k_values,
        vrmse_values,
    ):
        #------------------------------------------------------------------------------------------------------
        # Inputs:
        #
        #     k_values:     [num_k]
        #     vrmse_values: [num_k]
        #
        # Total loss:
        #
        #     pooled VRMSE over all k values
        #     + exponential slope regularization evaluated only at SLOPE_K / k_ref
        #
        # The exponential parameters are refit from the current VRMSE(k) values. They are not nn.Parameters and
        # they are not included in the optimizer.
        #------------------------------------------------------------------------------------------------------

        if k_values.ndim != 1:
            raise ValueError(
                f"Expected k_values with shape [num_k], but got {k_values.shape}."
            )

        if vrmse_values.ndim != 1:
            raise ValueError(
                f"Expected vrmse_values with shape [num_k], but got {vrmse_values.shape}."
            )

        if k_values.shape[0] != vrmse_values.shape[0]:
            raise ValueError(
                f"k_values and vrmse_values must have the same length, but got "
                f"{k_values.shape[0]} and {vrmse_values.shape[0]}."
            )

        if k_values.shape[0] < 4:
            raise ValueError(
                "At least four k values are required to fit p, A, k_shift, and w."
            )

        k_values = k_values.to(
            device = vrmse_values.device,
            dtype = vrmse_values.dtype,
        )

        k_ref = torch.tensor(
            self.k_ref,
            device = vrmse_values.device,
            dtype = vrmse_values.dtype,
        )

        reference_mask = torch.isclose(
            k_values,
            k_ref,
        )

        if not torch.any(reference_mask):
            raise ValueError(
                f"k_ref = {self.k_ref} must appear in k_values."
            )

        prediction_loss = torch.mean(vrmse_values)

        _, p, A, k_shift, w = self.fit_curve_parameters(
            k_values = k_values,
            vrmse_values = vrmse_values,
        )

        slope = self.exponential_slope(
            k_value = k_ref,
            A = A,
            k_shift = k_shift,
            w = w,
        )

        slope_loss = torch.sqrt(
            slope ** 2 + self.eps
        )

        total_loss = (
            self.prediction_weight * prediction_loss
            + self.slope_weight * slope_loss
        )

        diagnostics = {
            "total_loss": total_loss.detach(),
            "prediction_loss": prediction_loss.detach(),
            "slope_loss": slope_loss.detach(),
            "slope": slope.detach(),
            "p": p.detach(),
            "A": A.detach(),
            "k_shift": k_shift.detach(),
            "w": w.detach(),
        }

        return total_loss, diagnostics
