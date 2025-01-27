

class RewardCalculator:
    def __init__(self):
        self.weights = {
            'triangulation': 1.0,
            'detection': 0.5,
            'formation': 0.3,
            'smoothness': 0.1
        }
    
    def compute_reward(self):
        """Compute the reward for the current state.
        
        The reward consists of several components:
        1. Triangulation accuracy: How well the estimated position matches ground truth
        2. Detection quality: Whether targets are visible in both cameras
        3. Formation quality: Maintaining optimal distance between drones
        4. Motion smoothness: Penalize jerky movements
        
        Returns:
            float: Combined reward value
        """
        reward = 0.0
        
        # Normalize weights for better stability
        weights = {
            'triangulation': 1.0,
            'detection': 0.5,
            'formation': 0.3,
            'smoothness': 0.1
        }

        # 1. Triangulation reward
        if self.is_triangulation_updated():
            tri_error = np.linalg.norm(self.triangulation.position - self.target_odom_gt.position)
            tri_reward = weights['triangulation'] * gaussian_reward(tri_error, sigma=1.0)
            reward += tri_reward

        # 2. Detection reward for each drone
        for drone_id in range(self.num_drones):
            if self.is_target_detected2d(drone_id):
                # Calculate detection quality based on target size and position in image
                det_quality = self.compute_detection_quality(drone_id)
                reward += weights['detection'] * det_quality

        # 3. Formation reward
        formation_error = self.compute_formation_error()
        formation_reward = weights['formation'] * gaussian_reward(formation_error, sigma=2.0)
        reward += formation_reward

        # 4. Smoothness penalty
        if hasattr(self, 'last_action'):
            action_diff = np.linalg.norm(self.current_action - self.last_action)
            smoothness_penalty = weights['smoothness'] * action_diff
            reward -= smoothness_penalty

        return reward

    def gaussian_reward(error, sigma):
        """Convert error to reward using Gaussian function.
        
        Args:
            error (float): Error value to convert
            sigma (float): Standard deviation of Gaussian
        
        Returns:
            float: Reward value between 0 and 1
        """
        return np.exp(-0.5 * (error / sigma)**2)

    def compute_detection_quality(self, drone_id):
        """Compute quality score for target detection.
        
        Considers:
        - Target size in image
        - Position relative to image center
        - Detection confidence
        
        Returns:
            float: Quality score between 0 and 1
        """
        detection = self.get_current_detection(drone_id)
        if detection is None:
            return 0.0
            
        # Normalize target size score
        size_score = self.compute_size_score(detection)
        
        # Penalize targets near image edges
        center_score = self.compute_center_score(detection)
        
        # Combine scores
        return size_score * center_score * detection.confidence