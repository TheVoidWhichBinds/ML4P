# inner_model.py
#================================================================================================================================================
# Inner model for local spatiotemporal prediction with variable input timestamp history.
#================================================================================================================================================

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    NUM_PHYSICAL_CHANNELS,
    SPATIAL_KERNEL_SIZE,
    TAYLOR_ALPHA,
    TAYLOR_X0,
    HIDDEN_CHANNELS,
    TEMPORAL_KERNEL_SIZE,
    MLP_HIDDEN_DIMS,
)





#==============================================================================================================
# TAYLOR SERIES ACTIVATION
#==============================================================================================================

class TaylorActivation(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = TAYLOR_ALPHA
        self.x0 = TAYLOR_X0

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
        shifted_x = x - self.x0

        for power in range(self.alpha + 1):
            out = out + (
                self.theta[power] / self.factorials[power]
            ) * torch.pow(shifted_x, power)

        return out










#==============================================================================================================
# INNER CNN MODEL
#==============================================================================================================

class InnerK(nn.Module):
    """
    
    
    """

    def _validate_input(self, x):
        #------------------------------------------------------------------------------------------------------
        # Expected input: [B, 4, k, s, s], where s = spatial_kernel_size.
        # If self.k is None, the model accepts any k >= temporal_kernel_size.
        #------------------------------------------------------------------------------------------------------

        if x.ndim != 5:
            raise ValueError(
                f"Expected input with shape [B, 4, k, s, s], but got tensor with shape {x.shape}."
            )

        if x.shape[1] != self.num_physical_channels:
            raise ValueError(
                f"Expected channel dimension 4, but got {x.shape[1]}."
            )

        if x.shape[2] < self.temporal_kernel_size:
            raise ValueError(
                f"Expected time dimension k >= {self.temporal_kernel_size}, but got k = {x.shape[2]}."
            )

        if self.k is not None and x.shape[2] != self.k:
            raise ValueError(
                f"This InnerK instance was initialized with k = {self.k}, but got input k = {x.shape[2]}."
            )

        if x.shape[3] != self.spatial_kernel_size or x.shape[4] != self.spatial_kernel_size:
            raise ValueError(
                f"Expected spatial patch size "
                f"{self.spatial_kernel_size} x {self.spatial_kernel_size}, "
                f"but got {x.shape[3]} x {x.shape[4]}."
            )





    def __init__(
        self,
        k = None,
    ):
        super().__init__()

        self.k = k
        self.num_physical_channels = NUM_PHYSICAL_CHANNELS
        self.spatial_kernel_size = SPATIAL_KERNEL_SIZE
        self.hidden_channels = HIDDEN_CHANNELS
        self.temporal_kernel_size = TEMPORAL_KERNEL_SIZE
        self.mlp_hidden_dims = MLP_HIDDEN_DIMS

        #------------------------------------------------------------------------------------------------------
        # Basic checks
        #------------------------------------------------------------------------------------------------------

        if self.k is not None and self.temporal_kernel_size > self.k:
            raise ValueError(
                f"TEMPORAL_KERNEL_SIZE = {self.temporal_kernel_size} cannot be larger than k = {self.k}."
            )

        if self.spatial_kernel_size < 1:
            raise ValueError(
                f"SPATIAL_KERNEL_SIZE must be positive, but got {self.spatial_kernel_size}."
            )

        if self.num_physical_channels != 4:
            raise ValueError(
                f"This model assumes 4 physical channels, but NUM_PHYSICAL_CHANNELS = {self.num_physical_channels}."
            )

        #------------------------------------------------------------------------------------------------------
        # CNN step:
        #
        #     [B, 4, k, s, s] -> [B, hidden_channels, k_out, 1, 1]
        #
        # The spatial kernel spans the whole local patch. The temporal dimension is not flattened anymore.
        #------------------------------------------------------------------------------------------------------
        self.cnn = nn.Sequential(
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
        )

        #------------------------------------------------------------------------------------------------------
        # Temporal pooling step:
        #
        #     [B, hidden_channels, k_out, 1, 1] -> [B, hidden_channels, 1, 1, 1]
        #
        # This makes the MLP input dimension independent of k.
        #------------------------------------------------------------------------------------------------------
        self.temporal_pool = nn.AdaptiveAvgPool3d(
            output_size = (
                1,
                1,
                1,
            )
        )

        #------------------------------------------------------------------------------------------------------
        # MLP step:
        #
        #     [B, hidden_channels] -> [B, 4]
        #
        # Since temporal pooling removes the k_out dependence, this same InnerK instance can be used for
        # multiple k values.
        #------------------------------------------------------------------------------------------------------
        mlp_layers = []
        previous_dim = self.hidden_channels

        for hidden_dim in self.mlp_hidden_dims:
            mlp_layers.append(
                nn.Linear(
                    in_features = previous_dim,
                    out_features = hidden_dim,
                )
            )

            mlp_layers.append(TaylorActivation())

            previous_dim = hidden_dim

        mlp_layers.append(
            nn.Linear(
                in_features = previous_dim,
                out_features = self.num_physical_channels,
            )
        )

        self.mlp = nn.Sequential(*mlp_layers)






    def forward(self, x):
        #------------------------------------------------------------------------------------------------------
        # Full model:
        #
        #     [B, 4, k, s, s]
        #         -> CNN
        #     [B, hidden_channels, k_out, 1, 1]
        #         -> temporal pooling
        #     [B, hidden_channels, 1, 1, 1]
        #         -> flatten
        #     [B, hidden_channels]
        #         -> MLP
        #     [B, 4]
        #------------------------------------------------------------------------------------------------------
        self._validate_input(x)

        cnn_features = self.cnn(x)

        pooled_features = self.temporal_pool(cnn_features)

        mlp_input = pooled_features.flatten(start_dim = 1)

        prediction = self.mlp(mlp_input)

        return prediction










#==============================================================================================================
# EXPONENTIAL VRMSE-CURVE SLOPE LOSS
#==============================================================================================================

class ExponentialSlopeLoss(nn.Module):
    def __init__(
        self,
        k_ref,
        fit_weight = 1.0,
        slope_weight = 1.0,
        prediction_weight = 1.0,
        eps = 1.0e-8,
    ):
        super().__init__()

        self.k_ref = float(k_ref)
        self.fit_weight = fit_weight
        self.slope_weight = slope_weight
        self.prediction_weight = prediction_weight
        self.eps = eps

        #------------------------------------------------------------------------------------------------------
        # Raw trainable curve parameters.
        #
        # The fitted curve is:
        #
        #     E(k) = p + A exp(-w (k - k_shift))
        #
        # p, A, and w are passed through softplus to keep them positive.
        # k_shift is unconstrained.
        #------------------------------------------------------------------------------------------------------

        self.raw_p = nn.Parameter(torch.tensor(0.0))
        self.raw_A = nn.Parameter(torch.tensor(0.0))
        self.k_shift = nn.Parameter(torch.tensor(0.0))
        self.raw_w = nn.Parameter(torch.tensor(-4.0))





    def get_curve_parameters(self):
        #------------------------------------------------------------------------------------------------------
        # Return constrained curve parameters:
        #
        #     E(k) = p + A exp(-w (k - k_shift))
        #------------------------------------------------------------------------------------------------------

        p = F.softplus(self.raw_p) + self.eps
        A = F.softplus(self.raw_A) + self.eps
        w = F.softplus(self.raw_w) + self.eps

        return p, A, self.k_shift, w





    def exponential_curve(
        self,
        k_values,
    ):
        #------------------------------------------------------------------------------------------------------
        # Evaluate the fitted exponential VRMSE curve at the supplied k values.
        #------------------------------------------------------------------------------------------------------

        p, A, k_shift, w = self.get_curve_parameters()

        curve_values = np.exp(p) + np.exp(A)* torch.exp(
            -1.0 * np.exp(w) * (k_values - k_shift)
        )

        return curve_values





    def exponential_slope(
        self,
        k_value,
    ):
        #------------------------------------------------------------------------------------------------------
        # Evaluate dE/dk for:
        #
        #     E(k) = p + A exp(-w (k - k_shift))
        #
        #     dE/dk = -A w exp(-w (k - k_shift))
        #------------------------------------------------------------------------------------------------------

        _, A, k_shift, w = self.get_curve_parameters()

        slope = -1.0 * A * w * torch.exp(
            -1.0 * w * (k_value - k_shift)
        )

        return slope





    def forward(
        self,
        k_values,
        vrmse_values,
    ):
        #------------------------------------------------------------------------------------------------------
        # Inputs:
        #
        #     k_values:   [num_k]
        #     vrmse_values: [num_k]
        #
        # Total loss:
        #
        #     pooled VRMSE over all k values
        #     + differentiable exponential-curve fit loss
        #     + slope regularization evaluated only at SLOPE_K / k_ref
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

        curve_values = self.exponential_curve(
            k_values = k_values,
        )

        fit_loss = torch.mean(
            torch.abs(vrmse_values - curve_values)
        )

        slope = self.exponential_slope(
            k_value = k_ref,
        )

        slope_loss = torch.sqrt(
            slope ** 2 + self.eps
        )

        total_loss = (
            self.prediction_weight * prediction_loss
            + self.fit_weight * fit_loss
            + self.slope_weight * slope_loss
        )

        p, A, k_shift, w = self.get_curve_parameters()

        diagnostics = {
            "total_loss": total_loss.detach(),
            "prediction_loss": prediction_loss.detach(),
            "fit_loss": fit_loss.detach(),
            "slope_loss": slope_loss.detach(),
            "slope": slope.detach(),
            "p": p.detach(),
            "A": A.detach(),
            "k_shift": k_shift.detach(),
            "w": w.detach(),
        }

        return total_loss, diagnostics
