# Delay System Requirements
- date: 2026-01-22 18:35
- author: Seungwook

## Objective
The objective of the delay system is to simulate various sensor and actuator delays that happens in real world to multi-agent RL environment. Generalizing these sim-to-real gap, I have identified four types of "imperfections":

1. Random Latency: The transport delay, such as communication delay, computation delay. Parameterized by min, max latency from uniform distribution.
2. First-order Lag: The dynamics delay, such as the filter delay. Parameterized by time constant. Should behave the same with the same time constants even under different sampling rates.
3. Staleness: Some sensors report slower than the control loop. The control stack then has to either use stale data that is identical to the previous step(s), or extrapolate using filters, etc.
4. Dropout: e.g when the communication pipeline drops a packet, we need to decide if we want to use the stale data, or at least signal that we don't have the latest information.

Also, for realistic simulation settings, noise needs to be introduced as well. There are also two type of noises:

1. "Fast" Noises: noises that change fast, such as noise in IMU, inaccuracies in bounding box location.
2. "Slow" Noises: noises that shift slowly, such as GPS drift.

Bringing these into my multi-agent RL scheme, the architecture should cover these items on every training steps:

1. A central "data bus/buffer", easily accessable for all the agents, that stores both raw sensor data without noise for reward computation, and sensor data with controlled noise added to it for observation computation.
2. Compute the derived data, e.g gimbal's global pose derived from the robot's pose and gimbal orientation.
3. Sample from the two data buffers(clean, noisy) to each agents' corresponding delays, e.g agent0 gets 1 step delay for it's IMU data, 5 steps for it's GPS data which is stale between these steps, 15 step delayed pose data from agent1, agent2, ... which are sometimes stale and sometimes not.
4. Repeat step 3 for other agents.
5. Compute rewards and observations or all the agents.

Note that step 4 and 5 can switch orders depending on the computational efficiency and/or architectural decision.

## Technical Details
- We need to set independent delay and noises for each data.
- For each step and for all agents, it would be efficient if dump all the clean data into a buffer, index data based on the required delay, lag, staleness, then circulate the buffer on step. Current isaaclab's `delay_buffer.py` @source/isaaclab/isaaclab/utils/buffers/delay_buffer.py implementation circulates the buffer right after it is accessed, so we would need our own implementation. May be it could be enough to make minimal tweaks to the current one.
- We do need to keep track of the data sampled in the last step for staleness and dropout.
- The same pipeline applys to noisy data. 
- For the noisy data, the noise should be added immediately after the clean data is picked up and assigned into the noisy data buffer.
- Some delay/noise parameters should be sampled per episode, while some should be per step.