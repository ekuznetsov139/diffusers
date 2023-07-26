from torch import nn
import tensorflow as tf
from .tf_bridge import silu

def get_activation(act_fn):
    act_tf = get_activation_tf(act_fn)
    if act_fn in ["swish", "silu"]:
        rv = nn.SiLU()
    elif act_fn == "mish":
        rv = nn.Mish()
    elif act_fn == "gelu":
        rv = nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
    rv.tf = act_tf
    return rv

def get_activation_tf(act_fn):
    if act_fn in ["swish", "silu"]:
        return silu
    elif act_fn == "gelu":
        return tf.nn.gelu
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
