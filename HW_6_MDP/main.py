import numpy as np
from collections import defaultdict

class MDP:
    def __init__(self, states, actions, transition_probabilities, rewards, gamma):
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.gamma = gamma
        self.optimal_policy = None
        self.optimal_value_function = None

    def _get_value_from_policy(self, policy):
        """Evaluate the value function for a given policy."""
        value = {s: 0 for s in self.states}
        while True:
            new_value = value.copy()
            for s in self.states:
                a = policy[s]
                new_value[s] = sum(
                    self.transition_probabilities[(s, a, s_p)] *
                    (self.rewards[(s, a)] + self.gamma * value[s_p])
                    for s_p in self.states
                )

            if max(abs(new_value[s] - value[s]) for s in self.states) < 1e-6:
                break

            value = new_value
        return value
    
    def _improve_policy(self, value):
        """Update the policy greedily based on the value function."""
        policy = {}
        for s in self.states:
            policy[s] = max(
                self.actions,
                key=lambda a: sum(
                    self.transition_probabilities[(s, a, s_p)] *
                    (self.rewards[(s, a)] + self.gamma * value[s_p])
                    for s_p in self.states
                )
            )
        return policy
    
    # Algorithm 5
    def find_optimal_policy(self):
        max_iterations = 1000

        # Initialize value function to zero
        value_function = {s: 0 for s in self.states} 
        
        # Initialize policy randomly
        policy = {s: np.random.choice(self.actions) for s in self.states}

        # Policy Iteration loop
        for _ in range(max_iterations):
            value_function = self._get_value_from_policy(policy)
            new_policy = self._improve_policy(value_function)

            if new_policy == policy:
                break 
            policy = new_policy
        
        # Save results
        self.optimal_policy = policy
        self.optimal_value_function = value_function
    
    # Algorithm 6
    def estimate_probabilities_from_trials(self, trials):
        state_action_counts = defaultdict(int)
        state_action_next_state_counts = defaultdict(int)
        
        for (s, a, s_next) in trials:
            state_action_counts[(s, a)] += 1
            state_action_next_state_counts[(s, a, s_next)] += 1
        
        for (s, a) in state_action_counts:
            for s_next in self.states:
                count_sa = state_action_counts[(s, a)]
                count_sa_next = state_action_next_state_counts[(s, a, s_next)]
                if count_sa > 0:
                    self.transition_probabilities[(s, a, s_next)] = count_sa_next / count_sa
                else:
                    self.transition_probabilities[(s, a, s_next)] = 1 / len(self.states)
    
if __name__ == "__main__":
    # Example Usage:
    states = [0, 1, 2]
    actions = ['a', 'b']
    gamma = 0.9
    rewards = {
        (0, 'a'): 10, (0, 'b'): 5,
        (1, 'a'): -1, (1, 'b'): 2,
        (2, 'a'): 0, (2, 'b'): 3
    }

    myMDP = MDP(states, actions, {}, rewards, gamma)

    # Example trials: (state, action, next_state)
    trials = [
        (0, 'a', 1), (0, 'b', 2), (1, 'a', 2),
        (1, 'b', 0), (2, 'a', 0), (2, 'b', 1),
        (0, 'a', 1), (1, 'b', 2), (2, 'b', 2)
    ]

    myMDP.estimate_probabilities_from_trials(trials)
    myMDP.find_optimal_policy()

    print("Estimated Transition Probabilities:", myMDP.transition_probabilities)
    print("Optimal Policy:", myMDP.optimal_policy)
    print("Optimal Value Function:", myMDP.optimal_value_function)