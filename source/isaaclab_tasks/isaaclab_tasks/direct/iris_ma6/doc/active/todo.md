# TODO list for iris_ma6
- [x] Move the gimbal mount to the front of the drone for better visibility
- [ ] Per-agent randomization for heterogeneous fleet sim-to-real: noise scale, dropout rate, and max_lin_vel. Requires `set_noise_scale`/`set_dropout_rate` to accept per-agent values; `_max_lin_vel` shape change from `(N,)` to `(N, num_agents)`. Also prevents catastrophic forgetting. (added on 2026-03-23)
- [ ] Complete the first successful train run. (added on 2026-03-19)
- [ ] Add gimbal stability/responsiveness to curriculum/randomization after first successful train run. (added on 2026-03-19)
- [ ] Integrate domain randomization to the env after first successful train run. (added on 2026-03-19)