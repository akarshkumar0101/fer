from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from jax.random import split
import flax
import flax.linen as nn

from einops import rearrange

import evosax

from color import hsv2rgb

cache = lambda x: x
identity = lambda x: x
cos = jnp.cos
sin = jnp.sin
tanh = jnp.tanh
sigmoid = lambda x: jax.nn.sigmoid(x) * 2. - 1.
gaussian = lambda x: jnp.exp(-x**2) * 2. - 1.
relu = jax.nn.relu
activation_fn_map = dict(cache=cache, identity=identity, cos=cos, sin=sin, tanh=tanh, sigmoid=sigmoid, gaussian=gaussian, relu=relu)

class CPPN(nn.Module):
    """
    CPPN Flax Model.
    Possible activations: cache (identity), identity, cos, sin, tanh, sigmoid, gaussian, relu.

    arch: str should be in the form "12;cache:15,gaussian:4,identity:2,sin:1" which means 12 layers, with each layer containing 15 neurons using cache, 4 neurons using gaussian, 2 neurons using identity, 1 neuron using sin
    inputs: str should be in the form "y,x,d,b" which means the inputs are y, x, d, b. Don't change this.
    init_scale: str should be in the form "default" or float. If default uses the default flax initialization scheme (lecun init). If float, it is the scale of the initialization variance (see code).
    """
    arch: str = "12;cache:15,gaussian:4,identity:2,sin:1" # means 12 layers, 15 neurons with cache, 4 neurons with gaussian, 2 neurons with identity, 1 neuron with sin
    inputs: str = "y,x,d,b" # "x,y,d,b,xabs,yabs"
    init_scale: str = "default"

    @nn.compact
    def __call__(self, x):
        n_layers, activation_neurons = self.arch.split(";")
        n_layers = int(n_layers)

        activations = [i.split(":")[0] for i in activation_neurons.split(",")]
        d_hidden = [int(i.split(":")[-1]) for i in activation_neurons.split(",")]
        dh_cumsum = list(np.cumsum(d_hidden))

        features = [x]
        for i_layer in range(n_layers):
            if self.init_scale == "default":
                x = nn.Dense(sum(d_hidden), use_bias=False)(x)
            else:
                kernel_init = nn.initializers.variance_scaling(scale=float(self.init_scale), mode="fan_in", distribution="truncated_normal")
                x = nn.Dense(sum(d_hidden), use_bias=False, kernel_init=kernel_init)(x)

            x = jnp.split(x, dh_cumsum)
            x = [activation_fn_map[activation](xi) for xi, activation in zip(x, activations)]
            x = jnp.concatenate(x)

            features.append(x)
        x = nn.Dense(3, use_bias=False)(x)
        features.append(x)
        # h, s, v = jax.nn.tanh(x) # CHANGED THIS TO TANH
        h, s, v = x
        return (h, s, v), features

    def generate_image(self, params, img_size=256, return_features=False):
        """
        Generate an image from the CPPN given the parameters at the resolution specified by img_size.
        If return_features is True, return the intermediate activations of the CPPN as well.
        """
        inputs = {}
        x = y = jnp.linspace(-1, 1, img_size)
        inputs['x'], inputs['y'] = jnp.meshgrid(x, y, indexing='ij')
        inputs['d'] = jnp.sqrt(inputs['x']**2 + inputs['y']**2) * 1.4
        inputs['b'] = jnp.ones_like(inputs['x'])
        inputs['xabs'], inputs['yabs'] = jnp.abs(inputs['x']), jnp.abs(inputs['y'])
        inputs = [inputs[input_name] for input_name in self.inputs.split(",")]
        inputs = jnp.stack(inputs, axis=-1)
        (h, s, v), features = jax.vmap(jax.vmap(partial(self.apply, params)))(inputs)
        r, g, b = hsv2rgb((h+1)%1, s.clip(0,1), jnp.abs(v).clip(0, 1))
        rgb = jnp.stack([r, g, b], axis=-1)
        if return_features:
            return rgb, features
        else:
            return rgb

class FlattenCPPNParameters():
    """
    Using evosax==0.1.6 (only old version works!!), flatten the parameters of the CPPN to a single vector.
    Simplifies and makes useful for various things, like analysis.
    """
    def __init__(self, cppn):
        self.cppn = cppn

        rng = jax.random.PRNGKey(0)
        d_in = len(self.cppn.inputs.split(","))
        self.param_reshaper = evosax.ParameterReshaper(self.cppn.init(rng, jnp.zeros((d_in,))))
        self.n_params = self.param_reshaper.total_params
    
    def init(self, rng):
        d_in = len(self.cppn.inputs.split(","))
        params = self.cppn.init(rng, jnp.zeros((d_in,)))
        return self.param_reshaper.flatten_single(params)

    def generate_image(self, params, img_size=256, return_features=False):
        params = self.param_reshaper.reshape_single(params)
        return self.cppn.generate_image(params, img_size=img_size, return_features=return_features)