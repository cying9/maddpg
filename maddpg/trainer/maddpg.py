from copy import copy, deepcopy

import torch
from torch import nn
from torch.optim import Adam

from maddpg.trainer.replay_buffer import ReplayBuffer
from maddpg.common.torch_utils import init_params


@torch.no_grad()
def make_update_exp(source, target, polyak=1e-2):
    for p, p_targ in zip(source.parameters(), target.parameters()):
        p_targ.mul_(1 - polyak).add_(polyak * p.data)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        super().__init__()
        self.layer0 = nn.Linear(input_dim, num_units)
        self.layer1 = nn.Linear(num_units, num_units)
        self.layer2 = nn.Linear(num_units, output_dim)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.activ(self.layer0(x))
        x = self.activ(self.layer1(x))
        x = self.layer2(x)
        return x


class Critic(nn.Module):
    def __init__(self, obs_shape_n, act_info_n, q_index, num_units=64, local_q_func=False):
        super(Critic, self).__init__()
        self.local_q_func = local_q_func
        self.q_index = q_index
        if self.local_q_func:
            input_dim = obs_shape_n[q_index] + act_info_n[q_index][0]
        else:
            input_dim = sum(obs_shape_n) + sum([x[0] for x in act_info_n])
        self.model = MLP(input_dim, 1, num_units)

    def forward(self, x, a):
        if self.local_q_func:
            inputs = torch.cat([x[self.q_index], a[self.q_index]], axis=1)
        else:
            inputs = torch.cat(x + a, axis=1)
        return self.model(inputs)


class MADDPG():
    def __init__(self, obs_shape_n, act_info_n, agent_index, args, local_q_func=False):
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.grad_norm_clipping = 0.5
        # Networks
        self.device = args.device

        self.vf = Critic(
            obs_shape_n=obs_shape_n,
            act_info_n=act_info_n,
            num_units=args.num_units,
            q_index=agent_index,
            local_q_func=local_q_func,
        ).to(self.device)

        act_dim, self.pdtype = act_info_n[agent_index]
        self.pi = MLP(obs_shape_n[agent_index],
                      act_dim,
                      num_units=args.num_units).to(self.device)

        # Initialize
        init_params(self.vf)
        init_params(self.pi)

        # Target Networks
        self.pi_targ = deepcopy(self.pi)
        for p in self.pi_targ.parameters():
            p.requires_grad = False
        self.vf_targ = deepcopy(self.vf)
        for p in self.vf_targ.parameters():
            p.requires_grad = False

        # Optimizer
        self.pi_optim = Adam(self.pi.parameters(), lr=args.lr)
        self.vf_optim = Adam(self.vf.parameters(), lr=args.lr)

        # Create Replay Buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    @torch.no_grad()
    def action(self, x):
        return self.pdtype(
            self.pi(torch.FloatTensor(x).to(
                self.device)).cpu()).sample().numpy()

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            return
        if not (t % 100 == 0):
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[
                i].replay_buffer.sample_index(index)
            obs_n.append(torch.FloatTensor(obs).to(self.device))
            obs_next_n.append(torch.FloatTensor(obs_next).to(self.device))
            act_n.append(torch.FloatTensor(act).to(self.device))
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        # Create tensors
        rew = torch.FloatTensor(rew).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Calculate q loss
        num_sample = 1
        target_q = 0.0
        with torch.no_grad():
            for i in range(num_sample):
                target_act_next_n = [self.pdtype(agents[i].pi_targ(obs_next_n[i])).sample() for i in range(self.n)]
                target_q_next = self.vf_targ(obs_next_n, target_act_next_n).squeeze(-1)
                target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q = self.vf(obs_n, act_n).squeeze(-1)
        vf_loss = torch.mean(torch.square(q - target_q))

        # optimization step
        self.vf_optim.zero_grad(set_to_none=True)
        vf_loss.backward()
        nn.utils.clip_grad_norm_(self.vf.parameters(), self.grad_norm_clipping)
        self.vf_optim.step()

        # Calculate policy loss
        for p in self.vf.parameters():
            p.requires_grad = False

        piflat = self.pi(obs_n[self.agent_index])
        p_reg = torch.mean(torch.square(piflat))
        act_input_n = copy(act_n)
        act_input_n[self.agent_index] = self.pdtype(piflat).sample()
        pg_loss = - self.vf(obs_n, act_input_n).mean()
        pi_loss = pg_loss + p_reg * 1e-3

        self.pi_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), self.grad_norm_clipping)
        self.pi_optim.step()

        for p in self.vf.parameters():
            p.requires_grad = True

        make_update_exp(self.pi, self.pi_targ)
        make_update_exp(self.vf, self.vf_targ)

        return [pi_loss.item(), vf_loss.item()]


if __name__ == "__main__":
    model = MLP(15, 1, 64)
    print(model)
