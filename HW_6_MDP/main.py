import numpy as np
from collections import defaultdict

def policy_iteration(states, actions, transition_probabilities, rewards, gamma, max_iterations=1000, tolerance=1e-6):
    """
    Policy Iteration Algorithm to find the optimal policy for an MDP.

    Args:
        states (list): List of states in the MDP.
        actions (list): List of actions in the MDP.
        transition_probabilities (dict): Transition probabilities, P(s' | s, a),
            where P[(s, a, s')] gives the probability of transitioning from state s
            to s' by taking action a.
        rewards (dict): Reward function, R(s, a, s').
            rewards[(s, a, s')] gives the reward for transitioning from state s
            to s' by taking action a.
        gamma (float): Discount factor (0 <= gamma < 1).
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence threshold for policy evaluation.

    Returns:
        tuple: Optimal policy and value function.
    """

    # Initialize policy randomly and value function to zero
    policy = {s: np.random.choice(actions) for s in states}
    value_function = {s: 0 for s in states} 

    def policy_evaluation(policy):
        """Evaluate the value function for a given policy."""
        value = {s: 0 for s in states}
        for _ in range(max_iterations):
            delta = 0
            new_value = value.copy()
            for s in states:
                a = policy[s]
                new_value[s] = sum(
                    transition_probabilities[(s, a, s_p)] *
                    (rewards[(s, a)] + gamma * value[s_p])
                    for s_p in states
                )
                delta = max(delta, abs(new_value[s] - value[s]))
            value = new_value
            if delta < tolerance:
                break
        return value


    def policy_improvement(value):
        """Update the policy greedily based on the value function."""
        new_policy = {}
        for s in states:
            best_action = max(
                actions,
                key=lambda a: sum(
                    transition_probabilities[(s, a, s_p)] *
                    (rewards[(s, a)] + gamma * value[s_p]) 
                    for s_p in states
                )
            )
            new_policy[s] = best_action
        return new_policy

    # Policy Iteration loop
    for _ in range(max_iterations):
        value_function = policy_evaluation(policy)
        new_policy = policy_improvement(value_function)
        if new_policy == policy:
            break  # Convergence
        policy = new_policy

    return policy, value_function

def learn_mdp_with_unknown_probabilities(
    states, actions, rewards, gamma, trials, max_iterations=1000, tolerance=1e-6
):
    """
    Learn an MDP with unknown transition probabilities.

    Args:
        states (list): List of states in the MDP.
        actions (list): List of actions in the MDP.
        rewards (dict): Reward function R(s, a).
            rewards[(s, a)] gives the reward for taking action a in state s.
        gamma (float): Discount factor (0 <= gamma < 1).
        trials (list of tuples): Observed trials in the form of
            (state, action, next_state).
        max_iterations (int): Maximum number of iterations for policy iteration.
        tolerance (float): Convergence threshold for policy evaluation.

    Returns:
        tuple: Estimated transition probabilities, optimal policy, and value function.
    """
    # Initialize counts for estimating P*
    state_action_counts = defaultdict(int)
    state_action_next_state_counts = defaultdict(int)
    
    def estimate_transition_probabilities():
        """Estimate transition probabilities from observed data."""
        transition_probabilities = {
            (s, a, s_next): 1 / len(states) for s in states for a in actions for s_next in states
        }
        
        # Update probabilities based on observed trials
        for (s, a, s_next) in trials:
            state_action_counts[(s, a)] += 1
            state_action_next_state_counts[(s, a, s_next)] += 1
        
        for (s, a) in state_action_counts:
            for s_next in states:
                count_sa = state_action_counts[(s, a)]
                count_sa_next = state_action_next_state_counts[(s, a, s_next)]
                if count_sa > 0:
                    transition_probabilities[(s, a, s_next)] = count_sa_next / count_sa
                else:
                    transition_probabilities[(s, a, s_next)] = 1 / len(states)
        
        return transition_probabilities
    
    # Learning loop
    for _ in range(max_iterations):
        # Step 1: Estimate transition probabilities from data
        transition_probabilities = estimate_transition_probabilities()

        # Step 2 and 3: Use policy iteration to find the optimal policy
        optimal_policy, optimal_value_function = policy_iteration(
            states, actions, transition_probabilities, rewards, gamma, max_iterations, tolerance
        )

    return transition_probabilities, optimal_policy, optimal_value_function


# Example Usage:
states = [0, 1, 2]
actions = ['a', 'b']
gamma = 0.9

# Rewards for each (state, action) pair
rewards = {
    (0, 'a'): 10, (0, 'b'): 5,
    (1, 'a'): -1, (1, 'b'): 2,
    (2, 'a'): 0, (2, 'b'): 3
}

# Example trials: (state, action, next_state)
trials = [
    (0, 'a', 1), (0, 'b', 2), (1, 'a', 2),
    (1, 'b', 0), (2, 'a', 0), (2, 'b', 1),
    (0, 'a', 1), (1, 'b', 2), (2, 'b', 2)
]

transition_probabilities, optimal_policy, optimal_value_function = learn_mdp_with_unknown_probabilities(
    states, actions, rewards, gamma, trials
)

print("Estimated Transition Probabilities:", transition_probabilities)
print("Optimal Policy:", optimal_policy)
print("Optimal Value Function:", optimal_value_function)