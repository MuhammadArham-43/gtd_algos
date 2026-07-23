from abc import abstractmethod
from typing import Callable, Iterable

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp

from gtd_algos.src.nets.MLP import MLP, layer_norm
from gtd_algos.src.agents.ActorCritic import act_funcs

## Networks for backward-view algos
class QNetwork(nn.Module):
    """Action-value function: Q(s,a)"""
    action_dim: int
    layer_norm: bool
    activation: str
    kernel_init: Callable
    bias_init: Callable = constant(0.0)

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError


class DenseQNetwork(QNetwork):
    hiddens: Iterable[int] = ()

    @nn.compact
    def __call__(self, x):
        no_batch_dim = (x.ndim == 1)
        if no_batch_dim:
            x = x[None]
        activation = act_funcs[self.activation]
        x = MLP(
            hiddens=self.hiddens,
            act=activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            pre_act_norm=self.layer_norm,
        )(x)
        x = activation(x)
        x = nn.Dense(self.action_dim, kernel_init=self.kernel_init, bias_init=constant(0.0))(x)
        if no_batch_dim:
            x = jnp.squeeze(x, axis=0)
        return x


class MinAtarQNetwork(QNetwork):
    @nn.compact
    def __call__(self, x):
        no_batch_dim = (x.ndim == 3)
        if no_batch_dim:
            x = x[None]
        assert x.ndim == 4, "input must have shape (N, H, W, C) or (H, W, C)"

        def activation(x):
            if self.layer_norm:
                x = layer_norm(x)
            return act_funcs[self.activation](x)

        x = nn.Conv(
            16,
            kernel_size=[3, 3],
            strides=1,
            padding='VALID',
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = activation(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            128,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = activation(x)

        x = nn.Dense(
            self.action_dim,
            kernel_init=self.kernel_init,
            bias_init=constant(0.0),
        )(x)
        if no_batch_dim:
            x = jnp.squeeze(x, axis=0)
        return x

class DoubleQNetwork(QNetwork):
    """Double Q-network: two Q-networks for double Q-learning"""
    hiddens: Iterable[int] = ()

    def __call__(self, x):
        q1 = DenseQNetwork(
            action_dim=self.action_dim,
            layer_norm=self.layer_norm,
            activation=self.activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            hiddens=self.hiddens,
        )(x)
        q2 = DenseQNetwork(
            action_dim=self.action_dim,
            layer_norm=self.layer_norm,
            activation=self.activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            hiddens=self.hiddens,
        )(x)
        return q1, q2

class QNetworkContinuousAction(nn.Module):
    """Action-value function: Q(s,a) for continuous action spaces"""
    layer_norm: bool = False
    activation: str = "relu"
    kernel_init: Callable = orthogonal(jnp.sqrt(2))
    bias_init: Callable = constant(0.0)

    @nn.compact
    def __call__(self, obs, action):
        raise NotImplementedError("This method should be implemented in subclasses.")

class DenseQNetworkContinuousAction(QNetworkContinuousAction):
    hiddens: Iterable[int] = ()

    @nn.compact
    def __call__(self, obs, action):
        no_batch_dim = (obs.ndim == 1)
        if no_batch_dim:
            obs = obs[None]
            action = action[None]
        activation = act_funcs[self.activation]
        x = jnp.concatenate([obs, action], axis=-1)
        x = MLP(
            hiddens=self.hiddens,
            act=activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            pre_act_norm=self.layer_norm,
        )(x)
        x = activation(x)
        x = nn.Dense(1, kernel_init=self.kernel_init, bias_init=constant(0.0))(x)
        x = jnp.squeeze(x, -1)
        if no_batch_dim:
            x = jnp.squeeze(x, axis=0)
        return x

class DenseDoubleQNetworkContinuousAction(QNetworkContinuousAction):
    hiddens: Iterable[int] = ()

    @nn.compact
    def __call__(self, obs, action):
        q1 = DenseQNetworkContinuousAction(
            layer_norm=self.layer_norm,
            activation=self.activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            hiddens=self.hiddens,
        )(obs, action)
        q2 = DenseQNetworkContinuousAction(
            layer_norm=self.layer_norm,
            activation=self.activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            hiddens=self.hiddens,
        )(obs, action)
        return q1, q2
