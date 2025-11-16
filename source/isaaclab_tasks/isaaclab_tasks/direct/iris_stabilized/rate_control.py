import torch
import math

class RateControl:
    """
    A PID 3-axis angular rate/velocity controller, implemented in PyTorch for 
    batched operations. This controller is designed for high-performance 
    robotics simulations.
    """

    def __init__(self,
                 p_gains: list[float],
                 i_gains: list[float],
                 d_gains: list[float],
                 ff_gains: list[float] | None = None,
                 integrator_limit: list[float] | None = None,
                 num_envs: int = 1,
                 device: str = "cpu"):
        """
        Initializes the RateControl object.

        Args:
            p_gains (list[float]): Proportional gains for body [roll, pitch, yaw] axes.
            i_gains (list[float]): Integral gains for body [roll, pitch, yaw] axes.
            d_gains (list[float]): Derivative gains for body [roll, pitch, yaw] axes.
            ff_gains (list[float], optional): Feed-forward gains. Defaults to zeros.
            integrator_limit (list[float], optional): Integrator term maximum absolute value.
            num_envs (int): The number of parallel environments to simulate.
            device (str): The device to create tensors on (e.g., "cpu" or "cuda").
        """
        self.device = device
        self.num_envs = num_envs

        # Convert gain and limit lists to tensors on the specified device
        self._gain_p = torch.tensor(p_gains, device=self.device)
        self._gain_i = torch.tensor(i_gains, device=self.device)
        self._gain_d = torch.tensor(d_gains, device=self.device)
        
        if ff_gains is None:
            ff_gains = [0.0, 0.0, 0.0]
        self._gain_ff = torch.tensor(ff_gains, device=self.device)

        if integrator_limit is None:
            integrator_limit = [0.2, 0.2, 0.1]
        self._lim_int = torch.tensor(integrator_limit, device=self.device)

        # Initialize state tensors for all environments
        self._rate_int = torch.zeros((self.num_envs, 3), device=self.device)
        self._control_allocator_saturation_positive = torch.zeros((self.num_envs, 3), dtype=torch.bool, device=self.device)
        self._control_allocator_saturation_negative = torch.zeros((self.num_envs, 3), dtype=torch.bool, device=self.device)

    def set_saturation_status(self, saturation_positive: torch.Tensor, saturation_negative: torch.Tensor) -> None:
        """
        Sets the saturation status from an external control allocator. This is used
        for anti-windup on the integral term.

        Args:
            saturation_positive (torch.Tensor): A boolean tensor of shape (num_envs, 3) 
                                                indicating positive saturation on each axis.
            saturation_negative (torch.Tensor): A boolean tensor of shape (num_envs, 3) 
                                                indicating negative saturation on each axis.
        """
        self._control_allocator_saturation_positive = saturation_positive
        self._control_allocator_saturation_negative = saturation_negative

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Resets the integral term to zero for specified environments.

        Args:
            env_ids (torch.Tensor, optional): A tensor of environment indices to reset. 
                                              If None, resets all environments.
        """
        if env_ids is None:
            self._rate_int.zero_()
        else:
            self._rate_int[env_ids] = 0.0

    def update(self, rate: torch.Tensor, rate_sp: torch.Tensor, angular_accel: torch.Tensor, dt: float, landed: torch.Tensor) -> torch.Tensor:
        """
        Runs one control loop cycle calculation for all environments.

        Args:
            rate (torch.Tensor): The current vehicle angular rates (num_envs, 3).
            rate_sp (torch.Tensor): The desired vehicle angular rate setpoints (num_envs, 3).
            angular_accel (torch.Tensor): The current angular accelerations (num_envs, 3).
            dt (float): The time step for integration.
            landed (torch.Tensor): A boolean tensor of shape (num_envs,) indicating if a vehicle is landed.

        Returns:
            torch.Tensor: The computed torque vector to apply to the vehicle (num_envs, 3).
        """
        # Calculate the angular rate error
        rate_error = rate_sp - rate

        # Vectorized PID control with feed-forward. Gains are broadcast across the batch.
        torque = self._gain_p * rate_error + self._rate_int - self._gain_d * angular_accel + self._gain_ff * rate_sp

        # Create a mask to update integrals only for environments that are not landed
        not_landed_mask = ~landed.view(-1, 1)
        self._update_integral(rate_error, dt, not_landed_mask)

        return torque

    def _update_integral(self, rate_error: torch.Tensor, dt: float, update_mask: torch.Tensor) -> None:
        """
        Updates the integral term in a vectorized manner.

        Args:
            rate_error (torch.Tensor): The rate error (num_envs, 3).
            dt (float): The time step.
            update_mask (torch.Tensor): A boolean mask (num_envs, 1) to select which integrals to update.
        """
        # Clone the error for modification during anti-windup
        rate_error_for_integral = rate_error.clone()

        # Anti-windup: prevent integral from growing further if control is saturated
        rate_error_for_integral = torch.where(self._control_allocator_saturation_positive,
                                              torch.clamp(rate_error_for_integral, max=0.0),
                                              rate_error_for_integral)
        
        rate_error_for_integral = torch.where(self._control_allocator_saturation_negative,
                                              torch.clamp(rate_error_for_integral, min=0.0),
                                              rate_error_for_integral)

        # I-term factor to counteract non-linear effects from large rate errors
        i_factor = rate_error_for_integral / math.radians(400.0)
        i_factor = torch.clamp(1.0 - i_factor * i_factor, min=0.0)

        # Perform the integration step
        rate_i = self._rate_int + i_factor * self._gain_i * rate_error_for_integral * dt

        # Ensure the new integral value is finite before applying it
        finite_mask = torch.isfinite(rate_i)
        clamped_rate_i = torch.clamp(rate_i, -self._lim_int, self._lim_int)

        # Final mask: update only if not landed AND the result is finite
        final_update_mask = update_mask & finite_mask

        # Selectively update the integral state where the update is valid
        self._rate_int = torch.where(final_update_mask, clamped_rate_i, self._rate_int)


if __name__ == '__main__':
    # --- Configuration ---
    num_environments = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Rate Controller Test using device: {device} ---")

    # --- Controller Initialization ---
    rate_ctrl = RateControl(
        p_gains=[0.15, 0.15, 0.2],
        i_gains=[0.05, 0.05, 0.02],
        d_gains=[0.003, 0.003, 0.0],
        ff_gains=[0.1, 0.1, 0.0],
        integrator_limit=[0.2, 0.2, 0.1],
        num_envs=num_environments,
        device=device
    )

    # --- Simulation Data ---
    # Create batched tensors for states and setpoints
    current_rate = torch.randn((num_environments, 3), device=device) * 0.1
    rate_setpoint = torch.tensor([[0.5, -0.5, 0.1],
                                  [-0.5, 0.5, -0.1],
                                  [0.0, 0.0, 0.5],
                                  [0.2, 0.3, 0.0]], device=device)
    angular_acceleration = torch.randn((num_environments, 3), device=device) * 0.01
    dt = 0.004  # Corresponds to a 250 Hz control loop

    # Landed status for each environment (e.g., env 2 is landed)
    is_landed = torch.tensor([False, False, True, False], device=device)

    # --- Run Controller Update ---
    print("\n--- First Update Step ---")
    torque_command = rate_ctrl.update(current_rate, rate_setpoint, angular_acceleration, dt, is_landed)

    # --- Print Results ---
    print(f"Rate Setpoint:\n{rate_setpoint}")
    print(f"Is Landed Status:\n{is_landed}")
    print(f"Computed Torque Command:\n{torque_command.round(decimals=4)}")
    print(f"Integrator State (env 2 should be 0):\n{rate_ctrl._rate_int.round(decimals=4)}")

    # --- Simulate Saturation and Reset ---
    print("\n--- Simulating Saturation and Reset ---")
    # Saturate positive roll for env 0 and negative pitch for env 1
    sat_pos = torch.zeros((num_environments, 3), dtype=torch.bool, device=device)
    sat_neg = torch.zeros((num_environments, 3), dtype=torch.bool, device=device)
    sat_pos[0, 0] = True
    sat_neg[1, 1] = True
    rate_ctrl.set_saturation_status(sat_pos, sat_neg)
    print("Saturation status set for env 0 (pos roll) and env 1 (neg pitch).")

    # Run another update
    torque_command_sat = rate_ctrl.update(current_rate, rate_setpoint, angular_acceleration, dt, is_landed)
    print(f"\nComputed Torque Command with Saturation:\n{torque_command_sat.round(decimals=4)}")
    print(f"Integrator State after Saturation:\n{rate_ctrl._rate_int.round(decimals=4)}")

    # Reset environment 3
    print("\nResetting environment 3...")
    rate_ctrl.reset(env_ids=torch.tensor([3], device=device))
    print(f"Integrator State after Reset:\n{rate_ctrl._rate_int.round(decimals=4)}")

    # Reset all environments
    print("\nResetting all environments...")
    rate_ctrl.reset()
    print(f"Integrator State after Full Reset:\n{rate_ctrl._rate_int.round(decimals=4)}")
