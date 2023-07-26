import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tensorflow as tf
import numpy as np


print("Creating tf_dtype")
tf_dtype = tf.float16
autoencoder_dtype = tf.float16

use_tf = True

def TFWrapper(parent, allow_tf=True, *args, **kwargs):
    class TFWrapper(parent):
        def __init__(self, allow_tf, *args, **kwargs):
            #print(parent, args, kwargs)
            super().__init__(*args, **kwargs)
            self.allow_tf = allow_tf
            self.full_path = "<unknown>"
            self.implementer = None

        def __call__(self, x):
            if use_tf and self.allow_tf:
                x = MaybeCast(x, (parent, self.full_path))
                if self.implementer is not None:
                    return self.implementer(x)
                return self.tf.__call__(x)
            x = MaybeUncast(x, (parent, self.full_path))
            return super().__call__(x)
    return TFWrapper(allow_tf, **kwargs)

def TFLinearWrapper(parent, allow_tf=True, allow_f8=True, *args, **kwargs):
    class TFLinearWrapper(parent):
        def __init__(self, allow_tf, *args, **kwargs):
            #print(parent, args, kwargs)
            super().__init__(*args, **kwargs)
            self.allow_tf = allow_tf
            self.allow_f8 = allow_f8
            self.full_path = "<unknown>"
            self.implementer = None

            @tf.function(jit_compile=True)
            def tf_call_nof8(x):
                with tf.compat.v1.get_default_graph()._attr_scope({"_nof8": tf.compat.v1.AttrValue(b=True)}):
                    return self.tf.__call__(x)

            @tf.function(jit_compile=True)
            def tf_call_f8(x):
                return self.tf.__call__(x)

            self.implementer = tf_call_f8 if allow_f8 else tf_call_nof8
            #self.tf_func = tf.function(lambda x: self.tf.__call__(x))
            self.concrete = None

        def __call__(self, x):
            if use_tf and self.allow_tf:
                x = MaybeCast(x, (parent, self.full_path))
                #if self.implementer is not None:
                #if self.allow_f8:
                #    return self.tf.__call__(x)
                if self.concrete is None:
                    self.concrete = self.implementer.get_concrete_function(x)
                return self.concrete(x)
                #return self.implementer(x)
                #return self.tf.__call__(x)
            x = MaybeUncast(x, (parent, self.full_path))
            return super().__call__(x)
    return TFLinearWrapper(allow_tf, **kwargs)

def TFConvWrapper(parent, allow_tf=True, stride=1, *args, **kwargs):
    class TFConvWrapper(parent):
        def __init__(self, allow_tf, stride, *args, **kwargs):
            super().__init__(stride=stride, *args, **kwargs)
            self.allow_tf = allow_tf
            self.stride = stride
            self.full_path = "<unknown>"

        def __call__(self, x):
            if use_tf and self.allow_tf:
                x = MaybeCast(x, (type(parent), self.full_path))
                if self.stride>1:
                    pad = ((0,0), (0,0), (1,0), (1,0))
                    x = tf.pad(x, pad, mode='CONSTANT')
                return self.tf.__call__(x)
            x = MaybeUncast(x, (type(parent), self.full_path))
            return super().__call__(x)

    return TFConvWrapper(allow_tf, stride, *args, **kwargs)


def WrapLinear(x, y, bias=True, allow_tf=True, dtype=None, allow_f8 = False):
    pt_layer = TFLinearWrapper(nn.Linear, allow_tf, allow_f8, in_features=x, out_features=y, bias=bias)
    tf_layer = tf.keras.layers.Dense(y, use_bias=bias, dtype=dtype or tf_dtype, nof8=not allow_f8)
    tf_layer.build(input_shape=(None, x))
    pt_layer.tf = tf_layer
    return pt_layer

def WrapConv2d(x, y, kernel_size, bias=True, stride=1, padding=None, allow_tf=True, dtype=None):
    if padding is None:
        padding = (kernel_size - 1) // 2
    pt_layer = TFConvWrapper(nn.Conv2d, allow_tf, stride=stride, in_channels=x, out_channels=y, kernel_size=kernel_size, padding=padding, bias=bias)
    tf_layer = tf.keras.layers.Conv2D(y, kernel_size=kernel_size, padding="same" if (padding>0 and stride==1) else "valid", data_format="channels_first", use_bias=bias, strides=stride, dtype=dtype or tf_dtype)
    tf_layer.build(input_shape=(x,None,None))
    pt_layer.tf = tf_layer
    return pt_layer

def WrapConvTranspose2d(x, y, kernel_size, stride, padding, bias=True, allow_tf=True, dtype=None):
    pt_layer = TFWrapper(nn.ConvTranspose2d, allow_tf, x, y, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    tf_layer = tf.keras.layers.ConvTranspose2D(y, kernel_size=kernel_size, padding="same" if (padding>0 and stride==1) else "valid", data_format="channels_first", use_bias=bias, strides=stride, dtype=dtype or tf_dtype)
    tf_layer.build(input_shape=(x,None,None))
    pt_layer.tf = tf_layer
    return pt_layer


def ExecConv(c, x):
    x = x.numpy()
    with tf.device("/device:gpu:0"):
        x = c.tf(x)
    x = torch.from_numpy(x.numpy())
    return x

#@tf.function(jit_compile=True, reduce_retracing=True, experimental_relax_shapes=True)
def LayerNormImpl(inputs, weights, bias, epsilon):
    tensor_shape = tf.shape(inputs)
    pre_dim = tf.size(inputs) // tensor_shape[-1]
    in_dim = tensor_shape[-1]
    squeezed_shape = [1, pre_dim, in_dim, 1]
    inputs = tf.reshape(inputs, squeezed_shape)
    scale = tf.ones([pre_dim], dtype=tf.float32)
    offset = tf.zeros([pre_dim], dtype=tf.float32)
    outputs, _, _ = tf.compat.v1.nn.fused_batch_norm(
        inputs,
        scale=scale,
        offset=offset,
        epsilon=epsilon,
        data_format="NCHW")
    outputs = tf.reshape(outputs, tensor_shape)
    return outputs*weights+bias

def WrapLayerNorm(dim, elementwise_affine=True, eps=1e-5, allow_tf=True, dtype=None):
    x = TFWrapper(nn.LayerNorm, allow_tf, normalized_shape=dim, elementwise_affine=elementwise_affine, eps=eps)
    y = tf.keras.layers.LayerNormalization(axis=-1, center=elementwise_affine, scale=elementwise_affine, epsilon=eps, dtype=dtype or tf_dtype)
    y.build(input_shape=(None,None,dim))
    x.implementer = lambda x: LayerNormImpl(x, y.weights[0], y.weights[1], eps)
    x.tf = y
    return x

@tf.function(jit_compile=True, reduce_retracing=True)
def _apply_normalization(reshaped_inputs, epsilon, weights, bias):
    group_reduction_axes = list(range(1, reshaped_inputs.shape.rank)) 
    axis = 0
    group_reduction_axes.pop(axis)

    mean, variance = tf.nn.moments(reshaped_inputs, group_reduction_axes, keepdims=True)

    sh = tf.shape(reshaped_inputs)
    wt_shape = [1, sh[1], sh[2]] + [1]*(len(sh)-3)
    gamma, beta = tf.reshape(weights, wt_shape), tf.reshape(bias, wt_shape)
    #print("_apply_norm", reshaped_inputs,mean,variance,gamma,beta,epsilon)
    normalized_inputs = tf.nn.batch_normalization(
        reshaped_inputs,
        mean=mean,
        variance=variance,
        scale=gamma,
        offset=beta,
        variance_epsilon=epsilon,
    )
    return normalized_inputs

def GroupNormImpl(inputs, axis, groups, weights, bias, epsilon):
    def _reshape_into_groups(inputs, axis, groups):
        input_shape = tf.shape(inputs)
        group_shape = [input_shape[i] for i in range(inputs.shape.rank)] 
        group_shape[axis] = input_shape[axis] // groups
        group_shape.insert(axis, groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs

    input_shape = tf.shape(inputs)
    reshaped_inputs = _reshape_into_groups(inputs, axis, groups)
    weights = tf.cast(weights, reshaped_inputs.dtype)
    bias = tf.cast(bias, reshaped_inputs.dtype)
    normalized_inputs = _apply_normalization(reshaped_inputs, tf.constant(epsilon,dtype=reshaped_inputs.dtype), weights, bias)
    return tf.reshape(normalized_inputs, input_shape)

def WrapGroupNorm(num_channels, num_groups, eps, ndim=4, allow_tf=True, dtype=None):
    x = TFWrapper(nn.GroupNorm, allow_tf, num_channels=num_channels, num_groups=num_groups, eps=eps, affine=True)
    y = tf.keras.layers.GroupNormalization(num_groups, epsilon=eps, axis=1, dtype=dtype or tf_dtype)
    if ndim==4:
        y.build(input_shape = (None, num_channels, None, None))
    else:
        y.build(input_shape = (None, num_channels, None))
    x.implementer = lambda x: GroupNormImpl(x, 1, num_groups, y.weights[0], y.weights[1], eps)
    x.tf = y
    return x

def WrapDropout(rate, allow_tf=True, dtype=None):
    pt_layer = TFWrapper(nn.Dropout, allow_tf, rate)
    tf_layer = lambda x: tf.nn.dropout(x, rate=rate)
    pt_layer.tf = tf_layer
    return pt_layer

def WrapAvgPool2d(kernel_size, stride, allow_tf=True, dtype=None):
    pt_layer = TFWrapper(nn.AvgPool2d, allow_tf, kernel_size=kernel_size, stride=stride)
    tf_layer = tf.keras.layers.AveragePooling2D(pool_size=kernel_size, stride=stride, data_format='channels_first', autocast = False, dtype=dtype or tf_dtype)
    pt_layer.tf = tf_layer
    return pt_layer

def CastToPT(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x.numpy())
    return x

def _lower_triangular_mask(shape):
    """Creates a lower-triangular boolean mask over the last 2 dimensions."""
    row_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-2)
    col_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-1)
    return tf.greater_equal(row_index, col_index)          

@tf.function(jit_compile=True, reduce_retracing=True)
def WrapScaledDotProductAttention(q, k, v, attn_mask, is_causal, dropout_p):
  with tf.compat.v1.get_default_graph()._attr_scope({"_nof8": tf.compat.v1.AttrValue(b=True)}):
    if not use_tf:
        q = MaybeUncast(q, "DotProductAttention")
        k = MaybeUncast(k)
        v = MaybeUncast(v)
        attn_mask = MaybeUncast(attn_mask)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

    if isinstance(attn_mask, torch.Tensor):
        attn_mask = tf.constant(attn_mask.numpy())
    if isinstance(q, torch.Tensor):
        q = q.numpy()
        k = k.numpy()
        v = v.numpy()

    shape = tf.shape(q)
    L = tf.shape(q)[-2]
    S = tf.shape(k)[-2]
    E = tf.shape(q)[-1]
    Ev = tf.shape(v)[-1]

    q = tf.reshape(q, [-1, L, E])
    k = tf.reshape(k, [-1, S, E])
    v = tf.reshape(v, [-1, S, Ev])
    if attn_mask is not None:
        attn_mask = tf.reshape(attn_mask, [-1, L, S])
    batch = tf.shape(q)[0]
    if is_causal:
        attn_mask = _lower_triangular_mask([batch,L,S])
    if attn_mask is None:
        attn_mask = 0.0
    elif attn_mask.dtype == tf.bool:
        attn_mask = tf.cast(tf.where(attn_mask, 0.0, -10000.0), q.dtype)
    attn_weight = tf.nn.softmax(tf.matmul(q, k, transpose_b = True) / tf.math.sqrt(tf.cast(E, q.dtype)) + attn_mask, axis=-1)
    if dropout_p>0.0:
        attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    result = tf.matmul(attn_weight, v)
    if len(shape)==4:
        return tf.reshape(result, [shape[0], shape[1], L, Ev])
    else:
        return tf.reshape(result, [shape[0], L, Ev])

def MaybeCast(x, caller=None):
    if x is None:
        return x
    if isinstance(x, torch.Tensor):
        x = tf.constant(x.numpy())
        if caller is not None:
            print("Casting input from torch to TF in", caller)
    elif isinstance(x, np.ndarray):
        x = tf.constant(x)
        if caller is not None:
            print("Casting input from numpy to TF in", caller)
    return x

def MaybeUncast(x, caller=None):
    if x is None:
        return x
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return x
    if caller is not None:
        print("Casting input from TF to torch in", caller)
    return torch.from_numpy(x.numpy())


# fuses 3 ops into 'fusion.kd' (1679 calls)
@tf.function(jit_compile=True, experimental_relax_shapes=True)
def silu(x):
    return tf.nn.silu(x)
