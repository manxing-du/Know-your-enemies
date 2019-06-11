import numpy as np
import tensorflow as tf
# from .mlp import mlp
import sys

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, x2, om, a_op,
                     x_market_price_dis,
                     x2_market_price_dis,
                     q_type,
                     hidden_sizes=(16,16),
                     # hidden_sizes=([2]),
                     activation=tf.nn.relu,
                     output_activation=tf.sigmoid, action_space=None):
    #print(x, a, x2, x_market_price_dis, x2_market_price_dis)

    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]

    if not om:
        with tf.variable_scope('main'):
            with tf.variable_scope('pi'):
                pi = act_limit * mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
            with tf.variable_scope('q'):
                q = tf.squeeze(mlp(tf.concat([x, a], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
            with tf.variable_scope('q', reuse=True):
                q_pi = tf.squeeze(mlp(tf.concat([x, pi], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
            #print(tf.shape(pi), tf.shape(q), tf.shape(q_pi))
        with tf.variable_scope('target'):
            with tf.variable_scope('pi'):
                pi_targ = act_limit * mlp(x2, list(hidden_sizes) + [act_dim], activation, output_activation)
            with tf.variable_scope('q'):
                q_targ = tf.squeeze(mlp(tf.concat([x2, a], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
            with tf.variable_scope('q', reuse=True):
                q_pi_targ = tf.squeeze(mlp(tf.concat([x2, pi_targ], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
            # print(tf.shape(pi_targ), tf.shape(q_targ), tf.shape(q_pi_targ))
        return pi, q, q_pi, q_pi_targ

    else:
        with tf.variable_scope('main'):

            with tf.variable_scope('pi'):
                # actor network with market price as input
                # pi = act_limit * mlp(tf.concat([x, x_market_price_dis], axis=-1), list(hidden_sizes) + [act_dim],
                #                     activation, output_activation)
                pi = act_limit * mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
                pi_ext = tf.tile(pi, [1, 301])
                indicator_pi = tf.greater(pi_ext, a_op)

            with tf.variable_scope('q'):
                if q_type == 'indi':
                    a_ext = tf.tile(a, [1, 301])
                    indicator_a = tf.greater(a_ext, a_op)
                    # change the input tensor into [batch, 301, 3]
                    q_in = tf.tile(tf.expand_dims(tf.concat([x, a], axis=-1), 1), [1, 301, 1])
                    q_concat = tf.concat([q_in, tf.expand_dims(a_op, 2)], axis=2)
                    q_o = tf.squeeze(mlp(q_concat, list(hidden_sizes) + [1], activation, None),axis=2)
                    q = tf.reduce_sum(tf.where(indicator_a, tf.multiply(q_o, x_market_price_dis),
                                               tf.zeros_like(x_market_price_dis)), axis=1)
                elif q_type == 'integ':

                    q_in = tf.tile(tf.expand_dims(tf.concat([x, a], axis=-1), 1), [1, 301, 1])
                    q_concat = tf.concat([q_in, tf.expand_dims(a_op, 2)], axis=2)
                    q_o = tf.squeeze(mlp(q_concat, list(hidden_sizes) + [1], activation, None), axis=2)
                    q = tf.reduce_sum(tf.multiply(q_o, x_market_price_dis), axis=1)

                else:
                    q = tf.squeeze(mlp(tf.concat([x, a, x_market_price_dis], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)

            with tf.variable_scope('q', reuse=True):
                if q_type == 'indi':
                    q_pi_in = tf.tile(tf.expand_dims(tf.concat([x, pi], axis=-1), 1), [1, 301, 1])
                    q_concat = tf.concat([q_pi_in, tf.expand_dims(a_op, 2)], axis=2)
                    q_o = tf.squeeze(mlp(q_concat, list(hidden_sizes) + [1], activation, None), axis=2)
                    q_pi = tf.reduce_sum(tf.where(indicator_pi, tf.multiply(q_o, x_market_price_dis),
                                                  tf.zeros_like(x_market_price_dis)), axis=1)
                elif q_type == 'integ':
                    q_pi_in = tf.tile(tf.expand_dims(tf.concat([x, pi], axis=-1), 1), [1, 301, 1])
                    q_concat = tf.concat([q_pi_in, tf.expand_dims(a_op, 2)], axis=2)
                    q_o = tf.squeeze(mlp(q_concat, list(hidden_sizes) + [1], activation, None), axis=2)
                    q_pi = tf.reduce_sum(tf.multiply(q_o, x_market_price_dis), axis=1)

                else:
                    q_pi = tf.squeeze(mlp(tf.concat([x, pi, x_market_price_dis], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
            #print(tf.shape(pi), tf.shape(q), tf.shape(q_pi))

        with tf.variable_scope('target'):
            with tf.variable_scope('pi'):
                # actor network with market price as input
                # pi_targ = act_limit * mlp(tf.concat([x, x2_market_price_dis], axis=-1), list(hidden_sizes) + [act_dim],
                #                                      activation, output_activation)
                pi_targ = act_limit * mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
                pi_targ_ext = tf.tile(pi_targ, [1, 301])
                indicator_pi_targ = tf.greater(pi_targ_ext, a_op)

            with tf.variable_scope('q'):
                if q_type == 'indi':
                    q_targ_in = tf.tile(tf.expand_dims(tf.concat([x2, a], axis=-1), 1), [1, 301, 1])
                    q_concat = tf.concat([q_targ_in, tf.expand_dims(a_op, 2)], axis=2)
                    q_o = tf.squeeze(mlp(q_concat, list(hidden_sizes) + [1], activation, None), axis=2)
                    q_targ = tf.reduce_sum(tf.where(indicator_a, tf.multiply(q_o, x2_market_price_dis),
                                                    tf.zeros_like(x2_market_price_dis)), axis=1)
                elif q_type == 'integ':
                    q_targ_in = tf.tile(tf.expand_dims(tf.concat([x2, a], axis=-1), 1), [1, 301, 1])
                    q_concat = tf.concat([q_targ_in, tf.expand_dims(a_op, 2)], axis=2)
                    q_o = tf.squeeze(mlp(q_concat, list(hidden_sizes) + [1], activation, None), axis=2)
                    q_targ = tf.reduce_sum(tf.multiply(q_o, x2_market_price_dis), axis=1)

                else:
                    q_targ = tf.squeeze(mlp(tf.concat([x2, a, x2_market_price_dis], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)

            with tf.variable_scope('q', reuse=True):
                if q_type == 'indi':
                    q_pi_targ_in = tf.tile(tf.expand_dims(tf.concat([x2, pi_targ], axis=-1), 1), [1, 301, 1])
                    q_concat = tf.concat([q_pi_targ_in, tf.expand_dims(a_op, 2)], axis=2)
                    q_o = tf.squeeze(mlp(q_concat, list(hidden_sizes) + [1], activation, None), axis=2)
                    q_pi_targ = tf.reduce_sum(tf.where(indicator_pi_targ, tf.multiply(q_o, x2_market_price_dis),
                                                       tf.zeros_like(x2_market_price_dis)), axis=1)

                elif q_type == 'integ':
                    q_pi_targ_in = tf.tile(tf.expand_dims(tf.concat([x2, pi_targ], axis=-1), 1), [1, 301, 1])
                    q_concat = tf.concat([q_pi_targ_in, tf.expand_dims(a_op, 2)], axis=2)
                    q_o = tf.squeeze(mlp(q_concat, list(hidden_sizes) + [1], activation, None), axis=2)
                    q_pi_targ = tf.reduce_sum(tf.multiply(q_o, x2_market_price_dis), axis=1)

                else:
                    q_pi_targ = tf.squeeze(mlp(tf.concat([x2, pi_targ, x2_market_price_dis], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)

        return pi, q, q_pi, q_pi_targ