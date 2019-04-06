import torch


class Memory(object):
    """
        Utility class that stores transition tuples:
            (current_state, action, next_state, intrinsic reward, extrinsic reward,
             intrinsic_value_estimate, extrinsic_value_estimate)
    """

    def __init__(self, capacity=100, device='cpu'):
        """
        :param capacity: The maximum number of trajectories to keep.
        """
        self.device = device
        self.capacity = capacity
        self.memory = {
            'states': [],
            'actions': [],
            'act_log_prob': [],
            'in_rews': [],
            'ex_rews': [],
            'in_val_est': [],
            'ex_val_est': []
        }

    def set_initial_state(self, initial_state, initial_in_val_est=None, initial_ex_val_est=None):
        """
            Call this function after calling env.reset(), when at the start of a new trajectory.
        :param initial_state:
        """
        initial_state = torch.tensor(initial_state, device=self.device).unsqueeze(dim=0)    # Unsqueeze at dim=0 to form a list

        if initial_in_val_est is None:
            initial_in_val_est = torch.tensor([0.], device=self.device)
        else:
            assert len(initial_in_val_est.shape) == 0, "intrinsic value estimate should be a scalar value"
            initial_in_val_est = initial_in_val_est.unsqueeze(dim=0)

        if initial_ex_val_est is None:
            initial_ex_val_est = torch.tensor([0.], device=self.device)
        else:
            assert len(initial_ex_val_est.shape) == 0, "extrinsic value estimate should be a scalar value"
            initial_ex_val_est = initial_ex_val_est.unsqueeze(dim=0)

        # Check capacity
        if len(self.memory['states']) >= self.capacity:
            for key in self.memory.keys():
                self.memory[key].pop(0)     # pop first item
        # Set initial state and value estimates for the new trajectory
        self.memory['states'].append(initial_state)
        self.memory['in_val_est'].append(initial_in_val_est)
        self.memory['ex_val_est'].append(initial_ex_val_est)

    def add_transition(self, action, action_log_prob, next_state, intrinsic_reward=0, extrinsic_reward=0,
                       intrinsic_value_estimate=None, extrinsic_value_estimate=None):
        """
            Add a transition.
        :param action:
        :param action_log_prob:
        :param next_state:
        :param intrinsic_reward:
        :param extrinsic_reward:
        :param intrinsic_value_estimate:    intrinsic value estimate for NEXT STATE
        :param extrinsic_value_estimate:    extrinsic value estimate for NEXT STATE
        :return:
        """
        if intrinsic_value_estimate is None:
            intrinsic_value_estimate = torch.tensor(0., device=self.device)
        if extrinsic_value_estimate is None:
            extrinsic_value_estimate = torch.tensor(0., device=self.device)

        # Check types
        assert type(action) is torch.Tensor, "action should be a torch.Tensor"
        assert type(action_log_prob) is torch.Tensor, "action_log_prob should be a torch.Tensor"
        assert type(intrinsic_value_estimate) is torch.Tensor, "intrinsic_value_estimate should be a torch.Tensor"
        assert type(extrinsic_value_estimate) is torch.Tensor, "extrinsic_value_estimate should be a torch.Tensor"

        # Check dimensions
        assert len(action_log_prob.shape) == 0, "action log-probability should be a scalar value"
        assert len(intrinsic_value_estimate.shape) == 0, "intrinsic value estimate should be a scalar value"
        assert len(extrinsic_value_estimate.shape) == 0, "extrinsic value estimate should be a scalar value"

        action = action.unsqueeze(dim=0)
        action_log_prob = action_log_prob.unsqueeze(dim=0)
        next_state = torch.tensor(next_state, device=self.device).unsqueeze(dim=0)
        intrinsic_reward = torch.tensor(intrinsic_reward, dtype=torch.float32, device=self.device).unsqueeze(dim=0)
        extrinsic_reward = torch.tensor(extrinsic_reward, dtype=torch.float32, device=self.device).unsqueeze(dim=0)
        intrinsic_value_estimate = intrinsic_value_estimate.unsqueeze(dim=0)
        extrinsic_value_estimate = extrinsic_value_estimate.unsqueeze(dim=0)

        # Check if it is the first transition in this trajectory
        if self.memory['states'][-1].shape[0] == 1:
            self.memory['actions'].append(action)
            self.memory['act_log_prob'].append(action_log_prob)
            self.memory['in_rews'].append(intrinsic_reward)
            self.memory['ex_rews'].append(extrinsic_reward)
        else:
            self.memory['actions'][-1] = torch.cat([self.memory['actions'][-1], action], dim=0)
            self.memory['act_log_prob'][-1] = torch.cat([self.memory['act_log_prob'][-1], action_log_prob], dim=0)
            self.memory['in_rews'][-1] = torch.cat([self.memory['in_rews'][-1], intrinsic_reward], dim=0)
            self.memory['ex_rews'][-1] = torch.cat([self.memory['ex_rews'][-1], extrinsic_reward], dim=0)

        self.memory['states'][-1] = torch.cat([self.memory['states'][-1], next_state], dim=0)
        self.memory['in_val_est'][-1] = torch.cat([self.memory['in_val_est'][-1], intrinsic_value_estimate], dim=0)
        self.memory['ex_val_est'][-1] = torch.cat([self.memory['ex_val_est'][-1], extrinsic_value_estimate], dim=0)

    def trajectory_intrinsic_return(self, batch_size):
        """
            Return the intrinsic return (total intrinsic reward) for a batch of latest trajectories.
        :param batch_size:
        :return:
        """
        rets = [torch.sum(self.memory['in_rews'][-(i + 1)]) for i in reversed(range(batch_size))]
        return rets

    def trajectory_extrinsic_return(self, batch_size):
        """
            Return the extrinsic return (total extrinsic reward) for a batch of latest trajectories.
        :param batch_size:
        :return:
        """
        rets = [torch.sum(self.memory['ex_rews'][-(i + 1)]) for i in reversed(range(batch_size))]
        return rets

    def intrinsic_rtg(self, batch_size):
        """
            Compute intrinsic reward-to-go. This is computed without end-of-episode reward cut off.
        :param batch_size: The number of latest trajectories to consider as a batch
        :return: a list of intrinsic reward-to-go for each trajectory in the batch.
        """
        assert batch_size < self.capacity, "batch size need to be smaller than memory capacity"

        rtg_list = []
        rews_cat = torch.cat(self.memory['in_rews'][-batch_size:], dim=0)
        rtg_all = [torch.sum(rews_cat[i:]) for i in range(rews_cat.shape[0])]
        start_idx = 0
        for i in reversed(range(batch_size)):
            length = self.memory['in_rews'][-(i+1)].shape[0]
            rtg_list.append(torch.tensor(rtg_all[start_idx : length]), device=self.device)
            start_idx += length
        return rtg_list

    def extrinsic_rtg(self, batch_size):
        """
            Compute extrinsic reward-to-go. This is computed with end-of-episode reward cut off.
        :param batch_size: The number of latest trajectories to consider as a batch
        :return: a list of intrinsic reward-to-go for each trajectory in the batch.
        """
        assert batch_size < self.capacity, "batch size need to be smaller than memory capacity"

        rtg_list = []
        for i in reversed(range(batch_size)):
            traj = self.memory['ex_rews'][-(i+1)]
            rtg_traj = torch.tensor([torch.sum(traj[j:]) for j in range(traj.shape[0])], device=self.device)
            rtg_list.append(rtg_traj)
        return rtg_list

    def act_log_prob(self, batch_size):
        """
            Return a batch of action log probabilities
        :param batch_size:
        :return:
        """
        return self.memory['act_log_prob'][-batch_size:]

    def intrinsic_val_est(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        assert batch_size < self.capacity, "batch size need to be smaller than memory capacity"

        return self.memory['in_val_est'][-batch_size:]

    def extrinsic_val_est(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        assert batch_size < self.capacity, "batch size need to be smaller than memory capacity"

        return self.memory['ex_val_est'][-batch_size:]

    def intrinsic_gae(self, batch_size, gamma=0.98, lam=0.96):
        """
            Compute GAE for intrinsic rewards. This is computed without end-of-episode reward cut off.
        :param batch_size:
        :param gamma:
        :param lam:
        :return: a one-dimensional tensor
        """
        assert batch_size < self.capacity, "batch size need to be smaller than memory capacity"

        # Assume for each episode, value estimate of length T, rewards of length T-1
        # Compensate for missing reward at the start of each episode - insert a 0 reward
        rews_list = []
        for i in reversed(range(batch_size)):
            rew = torch.cat([torch.tensor([0.], device=self.device), self.memory['in_rews'][-(i+1)]], dim=0)
            rews_list.append(rew)
        rews_cat = torch.cat(rews_list, dim=0)
        # Delete the first reward compensate
        rews_cat = rews_cat[1:]

        val_cat = torch.cat(self.memory['in_val_est'][-batch_size:], dim=0)

        assert rews_cat.shape[0] == val_cat.shape[0] - 1, "the length of concatenated rewards is not 1 less than the concatenated value estimates."

        delta = rews_cat + gamma * val_cat[1:] - val_cat[:-1]

        weights = torch.tensor([(gamma * lam) ** i for i in range(delta.shape[0])], device=self.device)

        weighted_delta = delta * weights

        gae = torch.tensor([torch.sum(weighted_delta[i:]) for i in range(weighted_delta.shape[0])], device=self.device)

        return gae

    def extrinsic_gae(self, batch_size, gamma=0.98, lam=0.96):
        """
            Compute GAE for extrinsic rewards. This is computed with end-of-episode reward cut off.
        :param batch_size:
        :param gamma:
        :param lam:
        :return: a list of one-dimensional tensors
        """
        assert batch_size < self.capacity, "batch size need to be smaller than memory capacity"

        gae_list = []

        # Compute GAE for each trajectory
        for i in reversed(range(batch_size)):
            rews = self.memory['ex_rews'][-(i+1)]
            vals = self.memory['ex_val_est'][-(i+1)]

            assert rews.shape[0] == vals.shape[0] - 1, "the length of rewards is not 1 less than the value estimates."

            delta = rews + gamma * vals[1:] - vals[:-1]

            weights = torch.tensor([(gamma * lam) ** i for i in range(delta.shape[0])], device=self.device)

            weighted_delta = delta * weights

            gae = torch.tensor([torch.sum(weighted_delta[i:]) for i in range(weighted_delta.shape[0])], device=self.device)

            gae_list.append(gae)

        return gae_list

    def batch_policy_gradient(self, batch_size):
        """
            Compute the policy gradient loss, i.e. average of weighted action log-probability, in the specified batch.
        :param batch_size:
        :return:
        """
        assert batch_size < self.capacity, "batch size need to be smaller than memory capacity"

        act_log_prob = self.memory['act_log_prob'][-batch_size:]

        in_gae = self.intrinsic_gae(batch_size)
        ex_gae = self.extrinsic_gae(batch_size)

        # Concatenate act_log_prob and ex_gae. Note that we are missing one value at each episode, so we compensate by
        #   inserting a value 0.
        act_log_prob_list = []
        ex_gae_list = []
        for i in range(batch_size):
            act_log_prob_com = torch.cat([torch.tensor([0.], device=self.device), act_log_prob[i]], dim=0)
            ex_gae_com = torch.cat([torch.tensor([0.], device=self.device), ex_gae[i]], dim=0)
            act_log_prob_list.append(act_log_prob_com)
            ex_gae_list.append(ex_gae_com)
        alp_cat = torch.cat(act_log_prob_list, dim=0)
        ex_gae_cat = torch.cat(ex_gae_list, dim=0)
        # Remove the leading 0
        alp_cat = alp_cat[1:]
        ex_gae_cat = ex_gae_cat[1:]

        # Calculate policy gradient. Intrinsic GAE and extrinsic GAE are added together
        pg_loss = torch.sum(alp_cat * (ex_gae_cat + in_gae)) / torch.tensor(batch_size, device=self.device)

        return pg_loss

    def sample_states(self, batch_size):
        """
            Sample a batch of states from the memory.
        :param batch_size: The number of states to sample. (Note that here batch_size is not the number of trajectories)
        :return: return a concatenated tensor
        """
        states_cat = torch.cat(self.memory['states'], dim=0)
        random_idx = torch.randint(0, states_cat.shape[0], (batch_size,))
        states_sampled = states_cat[random_idx]
        return states_sampled
