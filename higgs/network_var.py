from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

import logger

log = logger.get()




def weight_variable(shape,
                    init_method=None,
                    dtype=tf.float32,
                    init_param=None,
                    wd=None,
                    name=None,
                    trainable=True,
                    seed=0):
  """Declares a variable.

    Args:
        shape: Shape of the weights, list of int.
        init_method: Initialization method, "constant" or "truncated_normal".
        init_param: Initialization parameters, dictionary.
        wd: Weight decay, float.
        name: Name of the variable, str.
        trainable: Whether the variable can be trained, bool.

    Returns:
        var: Declared variable.
    """
  if dtype != tf.float32:
    log.warning("Not using float32, currently using {}".format(dtype))
  if init_method is None:
    initializer = tf.zeros_initializer(shape, dtype=dtype)
  elif init_method == "truncated_normal":
    if "mean" not in init_param:
      mean = 0.0
    else:
      mean = init_param["mean"]
    if "stddev" not in init_param:
      stddev = 0.1
    else:
      stddev = init_param["stddev"]
    log.info("Normal initialization std {:.3e}".format(stddev))
    initializer = tf.truncated_normal_initializer(
        mean=mean, stddev=stddev, seed=seed, dtype=dtype)
  elif init_method == "uniform_scaling":
    if "factor" not in init_param:
      factor = 1.0
    else:
      factor = init_param["factor"]
    log.info("Uniform initialization scale {:.3e}".format(factor))
    initializer = tf.uniform_unit_scaling_initializer(
        factor=factor, seed=seed, dtype=dtype)
  elif init_method == "constant":
    if "val" not in init_param:
      value = 0.0
    else:
      value = init_param["val"]
    initializer = tf.constant_initializer(value=value, dtype=dtype)
  elif init_method == "xavier":
    initializer = tf.contrib.layers.xavier_initializer(
        uniform=False, seed=seed, dtype=dtype)
  else:
    raise ValueError("Non supported initialization method!")
  try:
    shape_int = [int(ss) for ss in shape]
    log.info("Weight shape {}".format(shape_int))
  except:
    pass
  if wd is not None:
    if wd > 0.0:
      reg = lambda x: tf.multiply(tf.nn.l2_loss(x), wd)
      log.info("Weight decay {}".format(wd))
    else:
      log.warning("No weight decay")
      reg = None
  else:
    log.warning("No weight decay")
    reg = None
  var = tf.get_variable(
      name,
      shape,
      initializer=initializer,
      regularizer=reg,
      dtype=dtype,
      trainable=trainable)
  log.info("Initialized weight {}".format(var.name))
  return var


def weight_variable_cpu(shape,
                        init_method=None,
                        dtype=tf.float32,
                        init_param=None,
                        wd=None,
                        name=None,
                        trainable=True,
                        seed=0):
  """Declares variables on CPU."""
  with tf.device("/cpu:0"):
    return weight_variable(
        shape,
        init_method=init_method,
        dtype=dtype,
        init_param=init_param,
        wd=wd,
        name=name,
        trainable=trainable,
        seed=seed)


def batch_norm(x,
               is_training,
               gamma=None,
               beta=None,
               # axes=[0, 1, 2],
               axes=[0],
               eps=1e-10,
               name="bn_out",
               decay=0.99,
               dtype=tf.float32):
  """Applies batch normalization.
    Collect mean and variances on x except the last dimension. And apply
    normalization as below:
    x_ = gamma * (x - mean) / sqrt(var + eps) + beta

    Args:
      x: Input tensor, [B, ...].
      n_out: Integer, depth of input variable.
      gamma: Scaling parameter.
      beta: Bias parameter.
      axes: Axes to collect statistics.
      eps: Denominator bias.

    Returns:
      normed: Batch-normalized variable.
      mean: Mean used for normalization (optional).
  """
  n_out = x.get_shape()[-1]
  try:
    n_out = int(n_out)
    shape = [n_out]
  except:
    shape = None
  emean = tf.get_variable(
      "ema_mean",
      shape=shape,
      trainable=False,
      dtype=dtype,
      initializer=tf.constant_initializer(
          0.0, dtype=dtype))
  evar = tf.get_variable(
      "ema_var",
      shape=shape,
      trainable=False,
      dtype=dtype,
      initializer=tf.constant_initializer(
          1.0, dtype=dtype))
  if is_training:
    mean, var = tf.nn.moments(x, axes, name="moments")
    ema_mean_op = tf.assign_sub(emean, (emean - mean) * (1 - decay))
    ema_var_op = tf.assign_sub(evar, (evar - var) * (1 - decay))
    normed = tf.nn.batch_normalization(
        x, mean, var, beta, gamma, eps, name=name)
    return normed, [ema_mean_op, ema_var_op]
  else:
    normed = tf.nn.batch_normalization(
        x, emean, evar, beta, gamma, eps, name=name)
    return normed, None
