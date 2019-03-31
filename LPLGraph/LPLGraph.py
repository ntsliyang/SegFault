""" Local-Prediction-Error-Learning Graph API"""
import networkx as nx
import numpy as np
from scipy import stats
import torch
import pyro


class LPLGraph(object):

    def __init__(self, state_hash_size, num_actions, num_particles=5, particle_init_mean=0, particle_init_std=1, learning_rate=0.1, p_critical=1e-7):
        """

        :param state_hash_size: the length of the binary hash code for states. The total number of state representations
                                is 2 ** state_hash_size
        :param num_actions: The number of discrete (or discretized) action representations
        :param num_particles:   The number of particles (layers) for each node in each causal link
        :param particle_init_mean:  Mean of particles' initial values
        :param particle_init_std:   Standard deviation of particles' initial values
        :param learning_rate: Learning rate of LPL algorithm
        :param p_val: P value in the t-test when determine whether a causal link is definite
        """

        # set parameters
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

    def _convert_state_node_code(self, state):
        """
            Turn state node integer representation into string representation

        :param state:
        :return:
        """
        if type(state) is int:
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
        if type(action) is int:
            action_node = state_node + "_action_" + str(action)
        else:
            action_node = action

        return action_node

    def _add_edge(self, action, state):
        """
            Add an edge between an action and state

        :param action:  action node. Integer or string
        :param state:   state node. Integer or string
        """

        # Turn node integer representation into string representation
        state_node = self._convert_state_node_code(state)
        action_node = self._convert_action_node_code(state_node, action)

        # Check if the edge already exists. If yes, return
        if state_node in list(self.G.neighbors(action_node)):
            return

        # Crate edge attribute. Initialize particles for estimating causal strength
        #   Edge type defaults to "potential"
        #   V_a: particles for the action as cause
        #   V_s: particles for the state as outcome
        attr = {"type": "potential",
                "V_a": np.random.normal(self._particle_init_mean, self._particle_init_std, size=(self._num_particles)),
                "V_s": np.random.normal(self._particle_init_mean, self._particle_init_std, size=(self._num_particles))}

        # Create edge with attribute
        self.G.add_edges_from([(action_node, state_node, attr)])

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
                self._add_edge(potential_action_node, state_node)

        expected, cum_prod = 1, 1

        # Iterate through all definite causes
        for action_node in self.G.predecessors(state_node):
            if self.G[action_node][state_node]["type"] == "definite":
                # Particle value of the action in the specified layer
                v = self.G[action_node][state_node]["V_a"][layer]
                # Cumulative product term in the noise-or formula
                cum_prod *= (1 - v)

        # If given, also take into account the causal effect of potential_action
        if potential_action is not None:
            v = self.G[potential_action_node][state_node]["V_a"][layer]
            cum_prod *= (1 - v)

        # Final expected value
        expected = cum_prod * (1 - cum_prod)

        return expected

    def _modify_edge(self, action, state):
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

    def update_causal_strength(self, prev_state, action, next_state):
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

        # If the edge (action -> next_state) does not exist, create one
        if next_state_node not in list(self.G.neighbors(action_node)):
            self._add_edge(action_node, next_state_node)
            return

        # Otherwise, iterate through each layer and update particle values
        for i in range(self._num_particles):

            # Update causal strength (s_t, a_t) -> s_t+1 with observed = 1
            next_state_expected_val = self._expected(i, next_state_node, action_node)
            # print("layer: ", i, ", previous particle_val: ", self.G[action_node][next_state_node]["V_a"][i])
            self.G[action_node][next_state_node]["V_a"][i] += self._lr * (1 - next_state_expected_val)

            # Clip the values so that they do not explode
            self.G[action_node][next_state_node]["V_a"][i] = self.G[action_node][next_state_node]["V_a"][i].clip(-1, 1)
            # print("\t next_state_expected_val: ", next_state_expected_val, ", particle_val: ", self.G[action_node][next_state_node]["V_a"][i])

            # Hypothesis test to determine whether to modify the edge type
            self._modify_edge(action_node, next_state_node)

            # For every other (established) potential cause (s, a) of s_t+1, update with observed = 0
            for potential_action_node in list(self.G.predecessors(next_state_node)):
                if potential_action_node != action_node and \
                        self.G[potential_action_node][next_state_node]["type"] != "none":
                    self.G[potential_action_node][next_state_node]["V_a"][i] += self._lr * (0 - next_state_expected_val)

                    # Clip the values so that they do not explode
                    self.G[potential_action_node][next_state_node]["V_a"][i] = \
                        self.G[potential_action_node][next_state_node]["V_a"][i].clip(-1, 1)

                    # Hypothesis test to determine whether to modify the edge type
                    self._modify_edge(potential_state_node, next_state_node)

            # For every other (established) potential outcome s of (s_t, a_t), update with observed = 0
            for potential_state_node in list(self.G.successors(action_node)):
                if potential_state_node != next_state_node and \
                        self.G[action_node][potential_state_node]["type"] != "none":
                    # Need to calculate the expected effect value of each potential outcome state
                    state_expected_val = self._expected(i, potential_state_node, action_node)
                    self.G[action_node][potential_state_node]["V_a"][i] += self._lr * (0 - state_expected_val)

                    # Clip the values so that they do not explode
                    self.G[action_node][potential_state_node]["V_a"][i] = \
                        self.G[action_node][potential_state_node]["V_a"][i].clip(-1, 1)

                    # Hypothesis test to determine whether to modify the edge type
                    self._modify_edge(action_node, potential_state_node)

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
            otherwise, calculate the average variance for all potential outcomes of taking this action

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
            var_mean = np.mean(var_list)
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
