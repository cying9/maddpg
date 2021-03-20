# Pytorch Implementation of Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

This is a reproduction of the [MADDPG realization of OpenAI](https://github.com/openai/maddpg). Similarly, this code is to be run in conjunction with environments from a revised version of [Multi-Agent Particle Environments (MPE)]()


## Installation

- To install, `cd` into the root directory and type `pip install -e .`

- Known dependencies: Python (3.8.8), OpenAI gym (0.18.0), pytorch (1.8.0), numpy (1.19.2)

## Property

1. Similar document structure to the OpenAI's implementation
2. Similar performance under pure CPU mode
3. Removed operations for benchmarking, saving, or loading files.

## Deficit

1. The implementation with GPU (Quadro RTX 6000) runs slower than that with pure CPU [TODO].

## References

- original MADDPG paper

<pre>
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
</pre>

- OpenAI's [implementation]((https://github.com/openai/maddpg)) with tensorflow
