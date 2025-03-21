"""
This module implements various algorithms for solving Multi-Armed Bandit problems,
including UCB1, UCB-Tuned, and Thompson Sampling.
"""
import numpy as np


class UCB1:
    """Upper Confidence Bound (UCB1) algorithm for Multi-Armed Bandit problems.
    
    This implementation uses the standard UCB1 formula to balance exploration and exploitation.
    
    Args:
        n_arms (int): Number of arms in the bandit problem
    """
    def __init__(self, n_arms):
        self.arm_selection_counts = np.zeros(n_arms).astype(np.float)
        self.rewards = np.zeros(n_arms).astype(np.float)

    def select(self):
        """Select the next arm to play based on UCB1 strategy.
        
        Returns:
            int: Index of the selected arm
        """
        unplayed_arms = np.where(self.arm_selection_counts == 0)[0]
        if len(unplayed_arms) > 0:
            self.arm_selection_counts[unplayed_arms[0]] += 1
            return unplayed_arms[0]

        average_reward = self.rewards / self.arm_selection_counts
        total_selections = np.sum(self.arm_selection_counts)

        ucb_values = self._calculate_ucb_values(
            total_selections,
            self.arm_selection_counts,
            average_reward
            )
        chosen_arm = np.argmax(ucb_values)

        self.arm_selection_counts[chosen_arm] += 1

        return chosen_arm

    def _calculate_ucb_values(self, total_selections, arm_selections, avg_reward):
        """Calculate UCB values for each arm.
        
        Args:
            total_selections (int): Total number of arm selections
            arm_selections (np.array): Number of selections for each arm
            avg_reward (np.array): Average reward for each arm
            
        Returns:
            np.array: UCB values for each arm
        """
        exploration_factor = np.sqrt(2 * np.log(total_selections) / arm_selections)
        return avg_reward + exploration_factor

    def reward(self, chosen_arm):
        """Update the reward for the chosen arm.
        
        Args:
            chosen_arm (int): Index of the arm that was played
        """
        self.rewards[chosen_arm] += 1


class UCBTuned(UCB1):
    """Upper Confidence Bound Tuned (UCB-Tuned) algorithm for Multi-Armed Bandit problems.
    
    This variant of UCB uses the empirical variance of the rewards to better
    balance exploration and exploitation.
    
    Args:
        n_arms (int): Number of arms in the bandit problem
    """
    def __init__(self, n_arms):
        super().__init__(n_arms)

    def _calculate_ucb_values(self, total_selections, arm_selections, avg_reward):
        """Calculate UCB-Tuned values for each arm.
        
        Args:
            total_selections (int): Total number of arm selections
            arm_selections (np.array): Number of selections for each arm
            avg_reward (np.array): Average reward for each arm
            
        Returns:
            np.array: UCB-Tuned values for each arm
        """
        reward_variance = (np.sum(np.square(self.rewards - avg_reward)))
        empirical_variance = (1 / arm_selections) * reward_variance

        base_exploration = np.sqrt(2 * np.log(total_selections) / arm_selections)
        variance_term = empirical_variance + base_exploration

        exploration_bound = np.minimum(1/4, variance_term)
        exploration_factor = np.sqrt((np.log(total_selections) / arm_selections) * exploration_bound)

        return avg_reward + exploration_factor


class ThompsonSampling:
    """Thompson Sampling algorithm for Multi-Armed Bandit problems.
    
    This implementation uses Bayesian inference to balance exploration and exploitation.
    
    Args:
        n_arms (int): Number of arms in the bandit problem
    """
    def __init__(self, n_arms):
        self.success_counts = np.zeros(n_arms).astype(np.float)
        self.failure_counts = np.zeros(n_arms).astype(np.float)
        self.n_arms = n_arms

    def select(self):
        """Select the next arm to play based on Thompson Sampling strategy.
        
        Returns:
            int: Index of the selected arm
        """
        theta = self.success_counts + 1, self.failure_counts + 1
        theta_value = np.random.beta(theta)

        chosen_arm = np.argmax(theta_value)

        self.failure_counts[chosen_arm] += 1

        return chosen_arm

    def reward(self, chosen_arm):
        """Update the reward for the chosen arm.
        
        Args:
            chosen_arm (int): Index of the arm that was played
        """
        self.success_counts[chosen_arm] += 1
