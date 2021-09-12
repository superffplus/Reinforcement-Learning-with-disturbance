from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions import Categorical
import rollouts


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
        self.ob_space = ob_space
        self.act_space = act_space
        self.act_model = ActModel(ob_space, act_space, hidden_space=15)
        self.value_model = ValueModel(ob_space, hidden_space_1=15, hidden_space_2=7)
        self.act_optimizer = optim.Adam(self.act_model.parameters(), lr=1e-3)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=1e-3)
        self.value_loss = nn.MSELoss()
        self.device = device
        self.to(device)
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
        ob = torch.as_tensor(ob, dtype=torch.float32).to(self.device)
        action_probs = self.act_model(ob)
        action_tensor = Categorical(action_probs).sample()
        # optimal_action = torch.argmax(actions, dim=1)
        return action_tensor, action_probs

    @abstractmethod
    def learn(self, rollout: rollouts.Replay):
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

    def learn(self, rollout: rollouts.Replay):
        states, actions, rewards, ends, action_probs, values, advantages, returns, next_value = rollout.sample(50)

        states_tensor = torch.from_numpy(states).to(self.device).view(-1, self.ob_space)
        actions_tensor = torch.from_numpy(actions).to(self.device).view(-1, 1)
        rewards_tensor = torch.from_numpy(rewards).to(self.device).view(-1, 1)
        action_probs_tensor = torch.from_numpy(action_probs).to(self.device).view(-1, self.act_space)
        values_tensor = torch.from_numpy(values).to(self.device).view(-1, 1)
        advantages_tensor = torch.from_numpy(advantages).to(self.device).view(-1, 1)
        returns_tensor = torch.from_numpy(returns).to(self.device).view(-1, 1)
        next_values_tensor = torch.from_numpy(next_value).to(self.device).view(-1, 1)

        action, action_probs = self.step(states_tensor)
        action_log_probs = torch.log(torch.clamp(action_probs, 1e-6, 1.))
        psi_tensor = torch.gather(action_log_probs, 1, actions_tensor)
        predict_value = self.value_model(states_tensor)
        predict_value_no_back = predict_value.detach()
        arange_tensor = torch.arange(states.shape[0], dtype=torch.float32, device=self.device)
        discount_tensor = (self.gamma ** arange_tensor).detach()
        policy_loss_tensor = -(discount_tensor * predict_value_no_back * psi_tensor).mean()
        self.act_optimizer.zero_grad()
        policy_loss_tensor.backward()
        self.act_optimizer.step()

        # optimize value network
        discount_I_tensor = discount_tensor.view(-1, 1)
        u_tensor = rewards_tensor + discount_I_tensor * next_values_tensor
        value_loss = self.value_loss(predict_value, u_tensor)
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

    def learn(self, rollout: rollouts.Replay):
        states, actions, rewards, ends, action_probs, values, advantages, returns, next_value = rollout.sample(50)
        z_act, z_value, I_value = [], [], 1
        for para in self.act_model.parameters():
            z_act.append(torch.zeros(para.shape, device=self.device))
        for para in self.value_model.parameters():
            z_value.append(torch.zeros(para.shape, device=self.device))

        for index, state in enumerate(states):
            # resize tensor to [1, n] to keep consistent with other algorithm without iterating arrays
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_tensor = torch.as_tensor(actions[index], dtype=torch.int64).view(-1, 1).to(self.device)
            reward_tensor = torch.as_tensor(rewards[index], dtype=torch.float32).view(-1, 1).to(self.device)
            next_value_tensor = torch.as_tensor(next_value[index], dtype=torch.float32).view(-1, 1).to(self.device)

            # update actor network parameters
            u_tensor = (reward_tensor + self.gamma * next_value_tensor).view(-1)
            action, action_prob = self.step(state_tensor)
            action_log_probs = torch.log(torch.clamp(action_prob, 1e-6, 1.))
            psi_tensor = torch.gather(action_log_probs, 1, action_tensor)

            self.act_optimizer.zero_grad()
            psi_tensor.backward()
            pred_value = self.value_model(state_tensor)
            pred_value_no_backward = pred_value.detach().view(-1)

            for z_v, param in zip(z_act, self.act_model.parameters()):
                z_v.data.copy_(self.gamma * self.p_lambda * z_v + I_value * param.grad)
                param.grad.copy_(-(u_tensor - pred_value_no_backward) * z_v)

            self.act_optimizer.step()

            # update value network parameters
            pred_value.backward()
            self.value_optimizer.zero_grad()

            for z_v, param in zip(z_value, self.value_model.parameters()):
                z_v.data.copy_(self.gamma * self.p_lambda * z_v + param.grad)
                param.grad.copy_(-(u_tensor-pred_value_no_backward) * z_v)
            self.value_optimizer.step()


class PPOAgent(Agent):
    def __init__(self, ob_space, act_space, params, device='cpu'):
        super(PPOAgent, self).__init__(ob_space, act_space, params, device)
        assert len(params) == 3
        self.epsilon, self.gamma, self.p_lambda = self.params

    def learn(self, rollout: rollouts.Replay):
        states, actions, rewards, ends, action_probs, values, advantages, returns, next_value = rollout.sample(batch_size=64)
        state_tensor = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.long).to(self.device)
        action_probs_tensor = torch.as_tensor(action_probs, dtype=torch.float32).to(self.device)
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32).to(self.device).view(-1)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32).to(self.device).view(-1, 1)

        old_pi_tensor = torch.gather(action_probs_tensor, 1, action_tensor.view(-1, 1)).view(-1)
        new_action, new_pi_tensor = self.step(state_tensor)
        pi_tensor = torch.gather(new_pi_tensor, 1, new_action.unsqueeze(1)).squeeze(1)
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
        
        
class TRPOAgent(Agent):
    def __init__(self, ob_space, act_space, params, device='cpu'):
        super(TRPOAgent, self).__init__(ob_space, act_space, params, device)
        self.max_kl = 0.01

    def learn(self, rollout: rollouts.Replay):
        states, actions, old_pis, advantages, returns = rollout.sample(batch_size=64)
        state_tensor = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.long).to(self.device)
        old_pi_tensor = torch.as_tensor(old_pis, dtype=torch.float32).to(self.device)
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32).to(self.device)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32).to(self.device)

        action, new_pi_tensor = self.step(state_tensor)
        pi_tensor = torch.gather(new_pi_tensor, 1, action_tensor.unsqueeze(1)).squeeze(1)
        surrogate_tensor = (pi_tensor / old_pi_tensor) * advantage_tensor
        loss_tensor = surrogate_tensor.mean()
        loss_grads = autograd.grad(loss_tensor, self.act_model.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in loss_grads]).detach()

        # calculate conjugate gradient: Fx=g
        def f(x):
            prob_tensor = self.act_model(state_tensor)
            prob_old_tensor = prob_tensor.detach()
            # the values of prob_old_tensor and prob_tensor are the same
            kld_tensor = (prob_old_tensor * torch.log(prob_old_tensor/prob_tensor).clamp(1e-6, 1e6)).sum(axis=1)
            kld_loss_tensor = kld_tensor.mean()
            grads = autograd.grad(kld_loss_tensor, self.act_model.parameters(), create_graph=True)
            flatten_grad_tensor = torch.cat([grad.view(-1) for grad in grads])
            grad_matmul_x = torch.dot(flatten_grad_tensor, x)
            grad_grads = autograd.grad(grad_matmul_x, self.act_model.parameters())
            flatten_grad_grad = torch.cat([grad.contiguous().view(-1) for grad in grad_grads]).detach()
            fx = flatten_grad_grad + x * 0.01
            return fx
        x, fx = self.conjugate_gradient(f, loss_grad)
        natural_gradient_tensor = torch.sqrt(2 * self.max_kl / torch.dot(fx, x)) * x

        def set_actor_net_params(flatten_params):
            begin = 0
            for param in self.act_model.parameters():
                end = begin + param.numel()
                param.data.copy_(flatten_params[begin:end].view(param.size()))
                begin = end

        old_param = torch.cat([param.view(-1) for param in self.act_model.parameters()])
        expected_improve = torch.dot(loss_grad, natural_gradient_tensor)

        for learning_step in [0.,] + [.5 ** j for j in range(10)]:
            new_param = old_param + learning_step * natural_gradient_tensor
            set_actor_net_params(new_param)
            all_pi_tensor = self.act_model(state_tensor)
            new_pi_tensor = all_pi_tensor.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
            new_pi_tensor = new_pi_tensor.detach()
            surrogate_tensor = (new_pi_tensor / pi_tensor) * advantage_tensor
            objective = surrogate_tensor.mean().item()
            if np.isclose(learning_step, 0.):
                old_objective = objective
            else:
                if objective - old_objective > 0.1 * expected_improve * learning_step:
                    break
                else:
                    set_actor_net_params(old_param)

        pred_tensor = self.value_model(state_tensor)
        critic_loss_tensor = self.value_loss(pred_tensor, return_tensor)
        self.value_optimizer.zero_grad()
        critic_loss_tensor.backward()
        self.value_optimizer.step()

    @staticmethod
    def conjugate_gradient(f, b, iter_count=10, epsilon=1e-12, tol=1e-6):
        x = b * 0.
        r = b.clone()
        p = b.clone()
        rho = torch.dot(r, r)
        for i in range(iter_count):
            z = f(p)
            alpha = rho / (torch.dot(p, z) + epsilon)
            x += alpha * p
            r -= alpha * z
            rho_new = torch.dot(r, r)
            p = r + (rho_new / rho) * p
            rho = rho_new
            if rho < tol:
                break
        return x, f(x)
