import torch
import torch.nn as nn
import torch.optim as optim
import rollouts

from abc import abstractmethod


class ActModel(nn.Module):
    def __init__(self, ob_space, act_space, hidden_space=15):
        super(ActModel, self).__init__()
        self.linear_1 = nn.Linear(ob_space, hidden_space)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_space, act_space)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        linear_1 = self.linear_1(x)
        relu = self.relu(linear_1)
        linear_2 = self.linear_2(relu)
        softmax = self.softmax(linear_2)
        return softmax


class ValueModel(nn.Module):
    def __init__(self, ob_space, hidden_space_1, hidden_space_2):
        super(ValueModel, self).__init__()
        self.linear_1 = nn.Linear(ob_space, hidden_space_1)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_space_1, hidden_space_2)
        self.tanh = nn.Tanh()
        self.linear_3 = nn.Linear(hidden_space_2, 1)

    def forward(self, x):
        linear_1 = self.linear_1(x)
        relu = self.relu(linear_1)
        linear_2 = self.linear_2(relu)
        tanh = self.tanh(linear_2)
        linear_3 = self.linear_3(tanh)
        return linear_3


class Agent:
    """
    Agent in Reinforcement learning. It contains the following function:
    * move to the selected device
    * predict the optimal action in the specific state using actor model and value model
    * compute the gradient and backward to the network
    """
    def __init__(self, ob_space, act_space, params, device='cpu'):
        self.act_model = ActModel(ob_space, act_space, hidden_space=15)
        self.value_model = ValueModel(ob_space, hidden_space_1=15, hidden_space_2=7)
        self.act_optimizer = optim.Adam(self.act_model.parameters(), lr=1e-3)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=1e-3)
        self.value_loss = nn.MSELoss()
        self.device = device
        self.params = params

    def to(self, device):
        self.device = device
        self.act_model.to(device)
        self.value_model.to(device)

    def step(self, ob):
        """
        Execute the action according to the observation.\n
        :param ob: observation of the environment
        :return: the probability of actions, and optimal action
        """
        ob = torch.as_tensor(ob).to(self.device)
        actions = self.act_model(ob)
        # optimal_action = torch.argmax(actions, dim=1)
        return actions

    @abstractmethod
    def learn(self, rollout: rollouts.Rollouts):
        pass


class AdvantageACAgent(Agent):
    """
    Actor-Critic algorithm.\n
    * update actor parameters: decrease -Iq(S,A)ln(pi(A|S))
    * update critic parameters: decrease [R+rq(S',A')-q(S,A)]^2
    """
    def __init__(self, ob_space, act_space, params, device='cpu'):
        """
        Init Actor Critic algorithm agent
        :param ob_space: observation space
        :param act_space: action space
        :param params: gamma, alpha
        :param device: target device
        """
        super(AdvantageACAgent, self).__init__(ob_space, act_space, params, device)
        assert len(params) == 2
        self.gamma, self.alpha = self.params

    def learn(self, rollout: rollouts.Rollouts):
        states_tensor = torch.as_tensor(rollout.states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.as_tensor(rollout.actions, dtype=torch.float32).to(self.device)
        rewards_tensor = torch.as_tensor(rollout.rewards, dtype=torch.float32).to(self.device)
        done_tensor = torch.as_tensor(rollout.end_masks, dtype=torch.float32).to(self.device)

        action_probs = self.step(states_tensor)
        action_log_probs = torch.log(torch.clamp(action_probs, 1e-6, 1.))
        psi_tensor = torch.gather(action_log_probs, 1, actions_tensor).squeeze(1)
        predict_value = self.value_model(states_tensor)
        predict_value_no_back = predict_value.detach()

        # update policy parameters
        arange_tensor = torch.arange(states_tensor.shape[0], dtype=torch.float32, device=self.device)
        # [1, r, r^2, ..., r^n]
        discount_tensor = (self.gamma ** arange_tensor).detach()
        discount_I_tensor = torch.ones(states_tensor.shape[0], dtype=torch.float32, device=self.device)\
                            * discount_tensor
        policy_loss_tensor = -(discount_I_tensor * predict_value_no_back * psi_tensor)
        self.act_optimizer.zero_grad()
        policy_loss_tensor.backward()
        self.act_optimizer.step()

        # update value parameters
        # remove the first present state as we use the next state prediction here
        predict_next_value = torch.cat((predict_value[1:], predict_value[-1:])).detach()
        U_tensor = rewards_tensor + discount_tensor * predict_next_value
        value_loss = self.value_loss(predict_value, U_tensor)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


class EligibilityTraceA2CAgent(Agent):
    """
    Advantage Actor-Critic algorithm with eligibility trace
    """
    def __init__(self, ob_space, act_space, params, device='cpu'):
        super(EligibilityTraceA2CAgent, self).__init__(ob_space, act_space, params, device)
        assert len(params) == 3
        self.gamma, self.alpha, self.p_lambda = params

    def learn(self, rollout: rollouts.Rollouts):
        z_act, z_value, I_value = [], [], 1
        for para in self.act_model.parameters():
            z_act.append(torch.zeros(para.shape, device=self.device))
        for para in self.value_model.parameters():
            z_value.append(torch.zeros(para.shape, device=self.device))

        for index, state in enumerate(rollout.states):
            # resize tensor to [1, n] to keep consistent with other algorithm without iterating arrays
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_tensor = torch.as_tensor(rollout.actions[index], dtype=torch.float32).unsqueeze(0).to(self.device)
            reward_tensor = torch.as_tensor(rollout.rewards[index], dtype=torch.float32).unsqueeze(0).to(self.device)

            if index < (len(rollout.states) - 1):
                next_state_tensor = torch.as_tensor(rollout.states[index+1], dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_next_value = self.value_model(next_state_tensor)
                u_value = reward_tensor + self.gamma * pred_next_value
            else:
                u_value = reward_tensor

            # update actor network parameters
            action_probs = self.act_model(state_tensor)
            action_log_probs = torch.log(torch.clamp(action_probs, 1e-6, 1.))
            psi_tensor = torch.gather(action_log_probs, 1, action_tensor)

            self.act_optimizer.zero_grad()
            psi_tensor.backward()
            pred_value = self.value_model(state_tensor)
            pred_value_no_backward = pred_value.detach()

            for z_v, param in zip(z_act, self.act_model.parameters()):
                z_v.data.copy_(self.gamma * self.p_lambda * z_v + I_value * param.grad)
                param.grad.copy_(-(u_value - pred_value_no_backward) * self.alpha * z_v)

            self.act_optimizer.step()

            # update value network parameters
            pred_value.backward()
            self.value_optimizer.zero_grad()

            for z_v, param in zip(z_value, self.value_model.parameters()):
                z_v.data.copy_(self.gamma * self.p_lambda * z_v + param.grad)
                param.grad.copy_(-(u_value-pred_value_no_backward) * self.alpha * z_v)
            self.value_optimizer.step()


class PPOAgent(Agent):
    def __init__(self, ob_space, act_space, params, device='cpu'):
        super(PPOAgent, self).__init__(ob_space, act_space, params, device)
        assert len(params) == 3
        self.epsilon, self.gamma, self.p_lambda = self.params

    def learn(self, rollout: rollouts.Replay):
        states, actions, old_pis, advantages, returns = rollout.sample(batch_size=64)
        state_tensor = torch.as_tensor(states, dtype=torch.float32)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        old_pi_tensor = torch.as_tensor(old_pis, dtype=torch.float32)
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32)

        new_pi_tensor = self.step(state_tensor)
        pi_tensor = torch.gather(new_pi_tensor, 1, action_tensor.unsqueeze(1)).squeeze(1)
        surrogate_advantage_tensor = (pi_tensor / old_pi_tensor) * advantage_tensor
        clip_advantage_tensor = 0.1 * surrogate_advantage_tensor
        max_surrogate_advantage_tensor = advantage_tensor + torch.where(advantage_tensor > 0., clip_advantage_tensor,
                                                                        -clip_advantage_tensor)
        clipped_surrogate_advantage_tensor = torch.min(surrogate_advantage_tensor,
                                                       max_surrogate_advantage_tensor)
        actor_loss_tensor = -clipped_surrogate_advantage_tensor.mean()
        self.act_optimizer.zero_grad()
        actor_loss_tensor.backward()
        self.act_optimizer.step()

        pred_value = self.value_model(state_tensor)
        critic_loss = self.value_loss(pred_value, return_tensor)
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()
