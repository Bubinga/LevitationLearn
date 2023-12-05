import numpy as np

class Policy:
    def select_action(self, state):
        """
        Takes in State:
        Num -> length states
        logits -> random sampling from states
        probs -> Weighting of probability of random sampling
        """
        num = len(state)
        logits = np.random.rand(num)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        return probs

class StochasticPolicy(Policy):
    def __init__(self, num):
        self.num =  num
        self.policy_params = np.random.rand(num)

    def update(self, states, actions, deltas, learning_rate=0.01):
        """
        Update the policy parameters using policy gradient.

        Parameters:
        - states: List of states encountered during an episode.
        - actions: List of corresponding actions taken in each state.
        - deltas: List of discounted future rewards.
        - learning_rate: Learning rate for the update.
        """
        grad = self.policy_gradient(states, actions, deltas)
        self.policy_params += learning_rate * grad

    def get_probability(self, state, action):
        """
        Get the probability of selecting a specific action in the given state.

        Parameters:
        - state: Current state.
        - action: Action to get the probability for.

        Returns:
        - Probability of selecting the specified action.
        """
        logits = np.dot(self.policy_params, state)
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        return probabilities[action]

    def policy_gradient(self, states, actions, deltas):
        """
        Compute the policy gradient.

        Parameters:
        - states: List of states encountered during an episode.
        - actions: List of corresponding actions taken in each state.
        - deltas: List of discounted future rewards.

        Returns:
        - grad: Gradient of the expected cumulative reward with respect to the policy parameters.
        """
        grad = np.zeros_like(self.policy_params)

        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            probability = self.get_probability(state, action)

            # Compute the gradient of the log probability of the chosen action
            grad += state * (1.0 / probability) * deltas[t]

        return grad / len(states)


class MyMDP:
    def __init__(self, states, actions, transitions, rewards, terminal_states, discount_factor, initial_state, goal_states):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.terminal_states = terminal_states
        self.discount_factor = discount_factor
        self.initial_state = initial_state
        self.goal_states = goal_states

    def get_states(self):
        return self.states

    def get_actions(self, state):
        return self.actions[state]

    def get_transitions(self, state, action):
        return self.transitions[state][action]

    def get_reward(self, state, action, next_state):
        return self.rewards[state][action][next_state]

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_discount_factor(self):
        return self.discount_factor

    def get_initial_state(self):
        return self.initial_state

    def get_goal_states(self):
        return self.goal_states
    
    class PolicyGradient:
        def __init__(self, mdp, policy, alpha) -> None:
            super().__init__()
            self.alpha = alpha  # Learning rate (gradient update step-size)
            self.mdp = mdp
            self.policy = policy

        """ Generate and store an entire episode trajectory to use to update the policy """

        def execute(self, episodes=100):
            for _ in range(episodes):
                actions = []
                states = []
                rewards = []

                state = self.mdp.get_initial_state()
                episode_reward = 0
                while not self.mdp.is_terminal(state):
                    action = self.policy.select_action(state)
                    next_state, reward = self.mdp.execute(state, action)

                # Store the information from this step of the trajectory
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    state = next_state

                deltas = self.calculate_deltas(rewards)
                self.policy.update(states=states, actions=actions, deltas=deltas)

        def calculate_deltas(self, rewards):
            """
            Generate a list of the discounted future rewards at each step of an episode
            Note that discounted_reward[T-2] = rewards[T-1] + discounted_reward[T-1] * gamma.
            We can use that pattern to populate the discounted_rewards array.
            """
            T = len(rewards)
            discounted_future_rewards = [0 for _ in range(T)]
            # The final discounted reward is the reward you get at that step
            discounted_future_rewards[T - 1] = rewards[T - 1]
            for t in reversed(range(0, T - 1)):
                discounted_future_rewards[t] = (
                    rewards[t]
                    + discounted_future_rewards[t + 1] * self.mdp.get_discount_factor()
                )
            deltas = []
            for t in range(len(discounted_future_rewards)):
                deltas += [
                    self.alpha
                    * (self.mdp.get_discount_factor() ** t)
                    * discounted_future_rewards[t]
                ]
            return deltas

