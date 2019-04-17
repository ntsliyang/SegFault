""" Local-Prediction-Error-Learning Graph API"""
import networkx as nx
import numpy as np
from scipy import stats
import torch
import pyro
from pyro.distributions import BetaBinomial, Bernoulli

class LPLGraph(object):

    def __init__(self, state_hash_size, num_actions, maximum_reward, reward_est_alpha=0.1, policy_model_generator=None,
                 num_particles=5, particle_init_mean=0, particle_init_std=1, learning_rate=0.1, p_critical=1e-7):
        """

        :param state_hash_size: the length of the binary hash code for states. The total number of state representations
                                is 2 ** state_hash_size
        :param num_actions: The number of discrete (or discretized) action representations
        :param maximum_reward: The maximum possible value of reward in this environment
        :param reward_est_alpha: The hyperparameter alpha for the moving average estimate of the reward corresponding
                                 to a state node.
        :param policy_model: A pyro model generator that can generate stochastic pyro functions (models) that represent
                             the current stochastic policy transition from state to action.
                             Requirements:
                             - param State_code: The hash code representation of the given canonical state as in the
                                                 causal graph (base 10).
                             - param Prefix: The name prefix to use when sampling variables using pyro distributions
                             - return: The hash code representation of the resulting choice of action.
        :param num_particles:   The number of particles (layers) for each node in each causal link
        :param particle_init_mean:  Mean of particles' initial values
        :param particle_init_std:   Standard deviation of particles' initial values
        :param learning_rate: Learning rate of LPL algorithm
        :param p_val: P value in the t-test when determine whether a causal link is definite
        """
        # set parameters
        self._policy_model_generator = policy_model_generator
        self._maximum_reward = maximum_reward
        self._reward_est_alpha = reward_est_alpha
        self._num_actions = num_actions
        self._num_particles = num_particles
        self._particle_init_mean = particle_init_mean
        self._particle_init_std = particle_init_std
        self._lr = learning_rate
        self._p_critical = p_critical

        # Initialize the graph
        self.G = nx.DiGraph()

        assert (type(state_hash_size) is int) and (state_hash_size > 0), "state_hash_size should be a positive integer."
        assert (type(num_actions) is int) and (num_actions > 0), "num_actions should be a positive integer."

        # The number of state nodes is 2 ** state_hash_size
        self.G.add_nodes_from(["state_" + str(i) for i in range(2 ** state_hash_size)])

        # The number of action nodes is (2 ** state_hash_size) * num_actions
        self.G.add_nodes_from(["state_" + str(i) + "_action_" + str(j)
                               for i in range(2 ** state_hash_size) for j in range(num_actions)])

        # Add action choice nodes for probability inference. The number of action choice nodes is 2 ** state_hash_size
        self.G.add_nodes_from(["state_" + str(i) + "_action_choice" for i in range(2 ** state_hash_size)])

        # Add optimality variables and corresponding edges and pyro stochastic models.
        self.G.add_nodes_from(["optimality_" + str(i) for i in range(2 ** state_hash_size)])
        self.G.add_edges_from(("state_" + str(i), "optimality_" + str(i),
                               {
                                   "reward_est": 0.,
                                   "opt_model": None
                               }) for i in range(2 ** state_hash_size))

    def update_policy_model_generator(self, generator):
        """
            Update the policy model generator. Used when the actual policy is updated.
        :param generator:
        :return:
        """
        self._policy_model_generator = generator

    def _convert_state_node_code(self, state):
        """
            Turn state node integer representation into string representation

        :param state:
        :return:
        """
        if type(state) is not str:
            state_node = "state_" + str(state)
        else:
            state_node = state

        return state_node

    def _convert_action_node_code(self, state_node, action):
        """
            Turn state node integer representation into string representation

        :param state_node:
        :param action:
        :return:
        """
        if type(action) is not str:
            action_node = state_node + "_action_" + str(action)
        else:
            action_node = action

        return action_node

    def _add_causal_edge(self, action_node, state_node):
        """
            Add a causal edge between an action node and state node

        :param action_node:  action node. Integer or string
        :param state_node:   state node. Integer or string
        """
        # Check if the edge already exists. If yes, return
        if state_node in list(self.G.neighbors(action_node)):
            return

        # Crate edge attribute. Initialize particles for estimating causal strength
        #   Edge type defaults to "potential"
        #   V_a: particles for the action as cause
        #   V_s: particles for the state as outcome
        attr = {"type": "potential",
                "V_a": np.random.normal(self._particle_init_mean, self._particle_init_std, size=(self._num_particles))
                }

        # Create edge with attribute
        self.G.add_edges_from([(action_node, state_node, attr)])

    def _add_inference_edge(self, action_choice_node, state_node):
        """
            Add a inference edge between an action choice node and state node, and modify the state node's Beta
                distribution hyperparameters.

        :param action_choice_node:
        :param state_node:
        :return:
        """

        # Check if the edge already exists. If yes, return
        if state_node in list(self.G.neighbors(action_choice_node)):
            return

        # Modify the stat node's Beta distribution hyperparameters. Add another dimension for the new associated action.
        #   Note that the hyperparameters shall be positive. So initialize the "prior" hyperparameters to be all one.
        # The Beta hyperparameter matrix for a state should have shape (#associated action nodes, #actions, 2)
        #   e.g. beta[3][5][0] is the alpha counts when the 3rd associated action nodes taking the 5th action and
        #   results in the current state node.

        # Initialize new slice
        beta = np.ones((1, self._num_actions, 2))
        beta_index = 0
        if "beta" in self.G.nodes[state_node]:
            beta_index = self.G.nodes[state_node]["beta"].shape[0]
            self.G.nodes[state_node]["beta"] = np.append(self.G.nodes[state_node]["beta"], beta, axis=0)
        else:
            self.G.nodes[state_node]["beta"] = beta

        # Store the beta tensor index in the edge attribute
        attr = {"beta_index": beta_index}

        # Create the edge with attribute
        self.G.add_edges_from([(action_choice_node, state_node, attr)])

        # Create the stochastic transition model and attribute it to the edge
        stoc_model = self._generate_transition_model(action_choice_node, state_node)
        self.G[action_choice_node][state_node]["stoc_model"] = stoc_model

    def _generate_transition_model(self, action_choice_node, state_node):
        """
            Return a pyro stochastic function (model) of the action -> state stochastic transition that comprises of
                a Beta-Binomial distribution, where the success probability of the state turned up as the outcome of
                the given choice of action is unknown and drawn from a beta prior distribution. The function will first
                sample the success probability and use that to sample the outcome (0 or 1).

        :param action_choice_node:
        :param state_node:
        :return:
        """
        beta_index = self.G[action_choice_node][state_node]["beta_index"]
        param = self.G.nodes[state_node]["beta"]

        def stoc_model(action_choice):
            alpha = param[beta_index][action_choice][0]
            beta = param[beta_index][action_choice][1]
            alpha = torch.tensor(alpha, dtype=torch.float32)
            beta = torch.tensor(beta, dtype=torch.float32)
            outcome = pyro.sample(action_choice_node + "->" + state_node, BetaBinomial(alpha, beta))
            return outcome

        return stoc_model

    def _generate_optimality_model(self, state_node):
        """
            Return a pyro stochastic function (model) of the state -> optimality probability
        :return:
        """
        state_idx = state_node[-1]
        optimality_node = "optimality_" + state_idx
        reward_est = self.G[state_node][optimality_node]["reward_est"]
        reward_normalized = reward_est - self._maximum_reward   # Make sure the normalized reward is always negative

        def stoc_model():
            p = np.exp(reward_normalized)
            opt = pyro.sample(state_node + "_optimality", Bernoulli(p))
            return opt

        return stoc_model

    def _expected(self, layer, state, potential_action=None):
        """
            Calculate the expected value of the given state as an effect variable in the specified layer, using the particles
                form the actions that has been established as the definite causes of this state
            If potential_action is given, then it serves as the potential cause of the state in question. Its particles will
                be taken into account when calculating the expected value.
            Otherwise, we only take into account the particles of those definite causes

        :param state:   The state whose expected value we are to calculate
        :param potential_action:    The potential cause we are considering
        :return:    The expected effect value of the given state
        """
        # Turn node integer representation into string representation
        state_node = self._convert_state_node_code(state)

        if potential_action is not None:
            potential_action_node = self._convert_action_node_code(state_node, potential_action)

            # Check if the edge already exists. If not, create the edge first
            if state_node not in list(self.G.neighbors(potential_action_node)):
                self._add_causal_edge(potential_action_node, state_node)

        expected, cum_prod = 1, 1

        # Iterate through all definite causes
        for action_node in self.G.predecessors(state_node):
            if ("action_choice" not in action_node) and self.G[action_node][state_node]["type"] == "definite":
                # Particle value of the action in the specified layer
                v = self.G[action_node][state_node]["V_a"][layer]
                # Cumulative product term in the noise-or formula
                cum_prod *= (1 - v)

        # If given, also take into account the causal effect of potential_action
        if potential_action is not None:
            v = self.G[potential_action_node][state_node]["V_a"][layer]
            cum_prod *= (1 - v)

        # Final expected value
        #expected = cum_prod * (1 - cum_prod)
        expected = (1 - cum_prod)

        return expected

    def _change_edge_type(self, action, state):
        """
            Perform t-test on the action's particles to determine whether to modify the edge type.
                E.g. from "potential" to "definite" or from "potential" to "none"

        :param action:
        :param state:
        """
        # Turn node integer representation into string representation
        state_node = self._convert_state_node_code(state)
        action_node = self._convert_action_node_code(state_node, action)

        # Obtain all particles
        v_val = self.G[action_node][state_node]["V_a"]

        # First test: null hypothesis: miu = 0, alternative hypothesis: miu > 0
        #   Reject and change edge type to "definite" if p_val < p_critical
        # Note that scipy uses two-sided test, so we half the calculated p value
        _, p_val = stats.ttest_1samp(v_val, 0)
        p_val /= 2
        if p_val < self._p_critical:
            self.G[action_node][state_node]["type"] = "definite"

        # Second test: null hypothesis: miu = 1, alternative hypothesis: miu < 1
        #   Reject and change edge type to "none" if p_val < p_critical
        _, p_val = stats.ttest_1samp(v_val, 1)
        p_val /= 2
        if p_val < self._p_critical:
            self.G[action_node][state_node]["type"] = "none"

    def _update_causal_strength(self, layer, action_node, state_node, expected_val):
        """

        :param action_node:
        :param state_node:
        :param expected_val:
        :return:
        """
        # Update particle value in the specified layer
        self.G[action_node][state_node]["V_a"][layer] += self._lr * (1 - expected_val)

        # Clip the values so that they do not explode
        self.G[action_node][state_node]["V_a"][layer] = self.G[action_node][state_node]["V_a"][layer].clip(-1.1, 1.1)

        # Hypothesis test to determine whether to modify the edge type
        self._change_edge_type(action_node, state_node)

    def update_transition(self, prev_state, action, next_state, reward=None):
        """

        :param prev_state:
        :param action:
        :param next_state:
        :return:
        """
        # Turn node integer representation into string representation
        prev_state_node = self._convert_state_node_code(prev_state)
        action_node = self._convert_action_node_code(prev_state_node, action)
        next_state_node = self._convert_state_node_code(next_state)


        # Update causal strength
        # If the edge (action -> next_state) does not exist, create one
        if next_state_node not in list(self.G.neighbors(action_node)):
            self._add_causal_edge(action_node, next_state_node)

        # Otherwise, iterate through each layer and update particle values
        else:
            for i in range(self._num_particles):
                # Update causal strength (s_t, a_t) -> s_t+1 with observed = 1
                next_state_expected_val = self._expected(i, next_state_node, action_node)
                self._update_causal_strength(i, action_node, next_state_node, next_state_expected_val)

                # For every other (observed) potential cause (s, a) of s_t+1, update with observed = 0
                for potential_action_node in list(self.G.predecessors(next_state_node)):
                    if ("action_choice" not in potential_action_node) and (potential_action_node != action_node) and \
                            (self.G[potential_action_node][next_state_node]["type"] != "none"):
                        self._update_causal_strength(i, potential_action_node, next_state_node, next_state_expected_val)

                # For every other (observed) potential outcome s of (s_t, a_t), update with observed = 0
                for potential_state_node in list(self.G.successors(action_node)):
                    if potential_state_node != next_state_node and \
                            self.G[action_node][potential_state_node]["type"] != "none":
                        # Need to calculate the expected effect value of each potential outcome state
                        state_expected_val = self._expected(i, potential_state_node, action_node)
                        self._update_causal_strength(i, action_node, potential_state_node, state_expected_val)


        # Update state nodes' beta distribution hyperparameters
        # Obtain action choice and action choice node
        if type(action) is str:
            action_choice = int(action_node[-1])
        else:
            action_choice = action
        action_choice_node = prev_state_node + "_action_choice"

        # If the edge (action -> next_state) does not exist, create one
        if next_state_node not in list(self.G.neighbors(action_choice_node)):
            self._add_inference_edge(action_choice_node, next_state_node)

        # Update alpha count for s_t+1
        beta_index = self.G[action_choice_node][next_state_node]["beta_index"]
        self.G.nodes[next_state_node]["beta"][beta_index][action_choice][0] += 1    # 0 is alpha cound

        # Update the stochastic transition model
        stoc_model = self._generate_transition_model(action_choice_node, next_state_node)
        self.G[action_choice_node][next_state_node]["stoc_model"] = stoc_model

        # For every other (observed) outcome s of (s_t, a_t), update its beta count and the corresponding stochastic
        #   transition model.
        #   i.e. action a_t took place but did not result in s
        for outcome_state_node in list(self.G.successors(action_choice_node)):
            beta_index = self.G[action_choice_node][outcome_state_node]["beta_index"]
            self.G.nodes[outcome_state_node]["beta"][beta_index][action_choice][1] += 1     # 1 is beta count
            stoc_model = self._generate_transition_model(action_choice_node, outcome_state_node)
            self.G[action_choice_node][outcome_state_node]["stoc_model"] = stoc_model

        # Update reward estimate for this state node (next_state), if the reward is given
        if reward is not None:
            optimality_node = "optimality_" + next_state_node[-1]
            # Moving average estimate
            self.G[next_state_node][optimality_node]["reward_est"] += self._reward_est_alpha * reward
            # Update optimality model
            self.G[next_state_node][optimality_node]["opt_model"] = self._generate_optimality_model(next_state_node)

    def get_particles(self, state, action, next_state=None):
        """
            If next_state is given, then return the particles of the action for that causal link specifically
            Otherwise, return all the particles of the action

        :param state:
        :param action:
        :param next_state:
        :return:
        """
        state_node = self._convert_state_node_code(state)
        action_node = self._convert_action_node_code(state_node, action)
        if next_state is not None:
            next_state_node = self._convert_state_node_code(next_state)
            return self.G[action_node][next_state_node]["V_a"]
        else:
            particle_list = [self.G[action_node][s]["V_a"] for s in self.G.successors(action_node)]
            return particle_list

    def action_confidence(self, state, action, next_state=None):
        """
            Calculate the variance of the particles (i.e. the level of not being confident) of the specified action
                if next_state is given, calculate the variance for that specific causal link
            Otherwise, calculate the average variance for all potential outcomes of taking this action
            If the specified action has no observed potential outcomes, i.e., do not have outgoing causal link, then
                return a variance of particle_init_std ** 2

        :param state:
        :param action:
        :return:
        """
        state_node = self._convert_state_node_code(state)
        action_node = self._convert_action_node_code(state_node, action)
        if next_state is not None:
            next_state_node = self._convert_state_node_code(next_state)
            var = np.var(self.G[action_node][next_state_node]["V_a"])
            return var
        else:
            var_list = [np.var(self.G[action_node][s]["V_a"]) for s in self.G.successors(action_node)]
            if len(var_list) > 0:
                var_mean = np.mean(var_list)
            else:
                var_mean = self._particle_init_std ** 2
            return var_mean

    def causal_strength(self, state, action, next_state):
        """
            Calculate the mean of the particles (i.e. the level of not being confident) of the specified causal link
        :param state:
        :param action:
        :param next_state:
        :return:
        """
        state_node = self._convert_state_node_code(state)
        action_node = self._convert_action_node_code(state_node, action)
        next_state_node = self._convert_state_node_code(next_state)
        return np.mean(self.G[action_node][next_state_node]["V_a"])

if __name__ == "__main__":

    # Test
    state_hash_size = 2
    num_actions = 2
    G = LPLGraph(state_hash_size, num_actions)

    # Adhoc data
    prev_s = 0
    a = 0
    next_s = 1

    G.update_causal_strength(prev_s, a, next_s)
