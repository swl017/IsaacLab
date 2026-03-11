# Visualization

## Camera frustum
- Visualize the camera FoV frustum
- Changes with zoom level

## Detection indicator
- Draw a line (LOS line) from each agents to the target
- Color turns to green if the agent detects the target
- Use delayed + noisy observation for ego pose and gimbal orientation, GT for target position for constructing the LOS line
- Use delayed + noisy observation for detection indicating color

## Triangulation estimation covariance
- Visualize the square root of x, y, z components of estimation covariance.
- Reuse the covariance computed in observation stage using delayed and noisy states.
- Implement this only after integrating the triangulation & estimation stack

## Bug fix
- Current implementation `iris_ma5` suffers runtime crash with more than 4 envs. The race condition(no valid covariance compmuted while the visualization stack tries to visualize it) is suspected to be the potential cause, but it's not identified, yet.