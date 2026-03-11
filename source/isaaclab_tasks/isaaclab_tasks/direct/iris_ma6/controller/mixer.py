# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixer matrix for X-configuration quadcopter (PX4 convention, ENU frame).

Motor Layout (top view, looking down from +Z):
```
              Front (+X)
                  ^
                  |
        M3        |        M1
        (CW)      |      (CCW)
           \      |      /
            \     |     /
             \    |    /
              \   |   /
               \  |  /
                \ | /
   Left (+Y) <---\|/---> Right (-Y)
                 /|\
                / | \
               /  |  \
              /   |   \
             /    |    \
            /     |     \
        M2        |        M4
       (CCW)      |       (CW)
                  |
                  v
              Rear (-X)
```

Motor Positions (FLU body frame):
- M1: Front-Right (+L/sqrt2, -L/sqrt2, 0) CCW
- M2: Rear-Left  (-L/sqrt2, +L/sqrt2, 0) CCW
- M3: Front-Left (+L/sqrt2, +L/sqrt2, 0) CW
- M4: Rear-Right (-L/sqrt2, -L/sqrt2, 0) CW

Frame Convention:
- World: ENU (East-North-Up)
- Body: FLU (Forward-Left-Up)
- +tau_z = CCW from above (nose left)
"""

from __future__ import annotations

import math
import torch


class MixerMatrix:
    """Mixer for X-configuration quadcopter following PX4 convention.

    Converts between body wrench [F, tau_x, tau_y, tau_z] and individual rotor thrusts [T1, T2, T3, T4].
    """

    def __init__(
        self,
        arm_length: float,
        k_f: float,
        k_m: float,
        device: str | torch.device = "cpu",
    ):
        """Initialize mixer matrix.

        Args:
            arm_length: Distance from center to rotor (L) in meters.
            k_f: Thrust coefficient [N/(rad/s)^2].
            k_m: Torque coefficient [Nm/(rad/s)^2].
            device: Torch device.
        """
        self.device = torch.device(device)
        self.arm_length = arm_length
        self.k_f = k_f
        self.k_m = k_m

        # Moment arm for X-configuration: d = L / sqrt(2)
        d = arm_length / math.sqrt(2.0)

        # Torque-to-thrust ratio
        c = k_m / k_f

        # Build mixer matrix: [F, tau_x, tau_y, tau_z]^T = M * [T1, T2, T3, T4]^T
        # From spec derivation:
        # F  = T1 + T2 + T3 + T4
        # τx = d(-T1 + T2 + T3 - T4)
        # τy = d(-T1 + T2 - T3 + T4)
        # τz = c(-T1 - T2 + T3 + T4)
        self._mixer_matrix = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],  # F
                [-d, d, d, -d],  # tau_x
                [-d, d, -d, d],  # tau_y
                [-c, -c, c, c],  # tau_z
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Compute pseudo-inverse for allocation: T = M^(-1) * [F, tau]
        self._mixer_matrix_inv = torch.linalg.pinv(self._mixer_matrix)

    @property
    def mixer_matrix(self) -> torch.Tensor:
        """Get the mixer matrix (4x4)."""
        return self._mixer_matrix

    @property
    def mixer_matrix_inv(self) -> torch.Tensor:
        """Get the inverse mixer matrix (4x4)."""
        return self._mixer_matrix_inv

    def aggregate(self, thrusts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert individual rotor thrusts to total thrust and body moments.

        Forward direction: [T1, T2, T3, T4] -> [F, tau_x, tau_y, tau_z]

        Args:
            thrusts: (N, 4) individual rotor thrusts [N].

        Returns:
            total_thrust: (N,) total thrust [N].
            moments: (N, 3) body moments [tau_x, tau_y, tau_z] [Nm].
        """
        # wrench = M @ thrusts
        wrench = torch.matmul(thrusts, self._mixer_matrix.T)  # (N, 4)

        total_thrust = wrench[:, 0]
        moments = wrench[:, 1:4]

        return total_thrust, moments

    def allocate(
        self,
        thrust_cmd: torch.Tensor,
        moment_cmd: torch.Tensor,
        thrust_min: float = 0.0,
        thrust_max: float | None = None,
    ) -> torch.Tensor:
        """Allocate total thrust and moments to individual rotor thrusts.

        Inverse direction: [F, tau_x, tau_y, tau_z] -> [T1, T2, T3, T4]

        Args:
            thrust_cmd: (N,) total thrust command [N].
            moment_cmd: (N, 3) moment commands [tau_x, tau_y, tau_z] [Nm].
            thrust_min: Minimum thrust per rotor [N].
            thrust_max: Maximum thrust per rotor [N]. If None, no upper limit.

        Returns:
            thrusts: (N, 4) individual rotor thrusts [N], clamped to [thrust_min, thrust_max].
        """
        # Build wrench vector [F, tau_x, tau_y, tau_z]
        wrench = torch.cat(
            [thrust_cmd.unsqueeze(-1), moment_cmd], dim=-1
        )  # (N, 4)

        # thrusts = M^(-1) @ wrench
        thrusts = torch.matmul(wrench, self._mixer_matrix_inv.T)  # (N, 4)

        # Clamp to physical limits
        thrusts = torch.clamp(thrusts, min=thrust_min)
        if thrust_max is not None:
            thrusts = torch.clamp(thrusts, max=thrust_max)

        return thrusts

    def thrust_to_omega(self, thrusts: torch.Tensor) -> torch.Tensor:
        """Convert rotor thrusts to angular velocities.

        Uses: T = k_f * omega^2 => omega = sqrt(T / k_f)

        Args:
            thrusts: (N, 4) rotor thrusts [N].

        Returns:
            omega: (N, 4) rotor angular velocities [rad/s].
        """
        # Ensure non-negative before sqrt
        thrusts_safe = torch.clamp(thrusts, min=0.0)
        omega = torch.sqrt(thrusts_safe / self.k_f)
        return omega

    def omega_to_thrust(self, omega: torch.Tensor) -> torch.Tensor:
        """Convert rotor angular velocities to thrusts.

        Uses: T = k_f * omega^2

        Args:
            omega: (N, 4) rotor angular velocities [rad/s].

        Returns:
            thrusts: (N, 4) rotor thrusts [N].
        """
        return self.k_f * omega**2

    def omega_to_torque(self, omega: torch.Tensor) -> torch.Tensor:
        """Convert rotor angular velocities to reaction torques.

        Uses: Q = k_m * omega^2

        Args:
            omega: (N, 4) rotor angular velocities [rad/s].

        Returns:
            torques: (N, 4) rotor reaction torques [Nm].
        """
        return self.k_m * omega**2
