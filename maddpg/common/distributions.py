import torch
from torch import distributions as dist
from torch.nn import functional as F
from functools import partial


class DiagGaussian(dist.Normal):
    def __init__(self, loc, logstd):
        super(DiagGaussian, self).__init__(loc=loc, scale=torch.exp(logstd))

    def mode(self):
        return self.mean

    def log_prob(self, action):
        return super().log_prob(action).sum(axis=-1)

    def entropy(self):
        return super().entropy().sum(axis=-1)


class SoftCategorical(dist.Categorical):
    def __init__(self, logits):
        super(SoftCategorical, self).__init__(logits=logits)

    def mode(self):
        return self.probs

    def sample(self):
        # Gumbel softmax output
        u = torch.rand(self.logits.shape, device=self.logits.device)
        return F.softmax(self.logits - torch.log(-torch.log(u)), dim=-1)

    def log_prob(self, value):
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        return value.mul(log_pmf).sum(-1)


class MultiCategorical(dist.Distribution):
    def __init__(self, inputs, input_lens):
        inputs_split = inputs.split(tuple(input_lens), dim=-1)
        self.cats = [
            dist.Categorical(logits=input_) for input_ in inputs_split
        ]

    def sample(self):
        return torch.stack([cat.sample() for cat in self.cats], dim=1)

    def mode(self):
        return torch.stack([torch.argmax(cat.probs, -1) for cat in self.cats],
                           dim=-1)

    def log_prob(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = torch.unbind(actions, dim=1)
        logps = torch.stack(
            [cat.log_prob(act) for cat, act in zip(self.cats, actions)])
        return torch.sum(logps, dim=0)

    def multi_entropy(self):
        return torch.stack([cat.entropy() for cat in self.cats], dim=1)

    def entropy(self):
        return torch.sum(self.multi_entropy(), dim=1)


class SoftMultiCategorical(MultiCategorical):
    def __init__(self, inputs, input_lens):
        self.nvec = tuple(input_lens)
        self.inputs = inputs
        self.cats = list(map(SoftCategorical, inputs.split(self.nvec, dim=-1)))

    def sample(self):
        return torch.cat([cat.sample() for cat in self.cats], dim=-1)

    def log_prob(self, x):
        if isinstance(x, torch.Tensor):
            x = torch.split(x, self.nvec, dim=1)
        return torch.stack(
            [cat.log_prob(act) for cat, act in zip(self.cats, x)]).sum(0)

    def mode(self):
        return torch.cat([cat.mode() for cat in self.cats], -1)


def make_pdtype(ac_space):
    from gym import spaces
    from multiagent.multi_discrete import MultiDiscrete
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return ac_space.shape[0], DiagGaussian
    elif isinstance(ac_space, spaces.Discrete):
        return ac_space.n, SoftCategorical
    elif isinstance(ac_space, MultiDiscrete):
        return sum(ac_space.nvec), partial(SoftMultiCategorical, input_lens=ac_space.nvec)


if __name__ == "__main__":
    pass
