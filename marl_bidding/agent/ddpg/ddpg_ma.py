import numpy as np
import tensorflow as tf
from ..base import BaseAgent
from . import core
from .core import get_vars
from .replay_buffer import ReplayBuffer
from .OUNoise import OUNoise
from .NNoise import NNoise

from spinup.utils.logx import EpochLogger
import sys

class DDPGAgent(BaseAgent):
    def __init__(self, observation_space, action_space, q_type, train_mode, model_path, model_name, scope,
                 batch_size=301,
                 market_price_dim=10, om=False,
                 actor_critic=core.mlp_actor_critic,
                 ac_kwargs=dict(), seed=0, replay_size=int(1e6), gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
                 logger_kwargs=dict(), save_freq=1):

        tf.set_random_seed(seed)
        np.random.seed(seed)

        with tf.variable_scope(scope):

            self.logger = EpochLogger(**logger_kwargs)
            self.logger.save_config(locals())
            self.market_price_dim = market_price_dim
            #
            self.batch_size = batch_size

            self.obs_dim = observation_space.shape[0]
            self.act_dim = action_space.shape[0]

            self.om = om
            self.o_dim = action_space.shape[0]

            # Action limit for clamping: critically, assumes all dimensions share the same bound!
            self.act_limit = 1

            # Share information about action space with policy architecture
            ac_kwargs['action_space'] = action_space

            # Inputs to computation graph

            self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(self.obs_dim, self.act_dim,
                                                                                       self.obs_dim, None, None)
            self.x_m_ph, self.x2_m_ph = None, None

            self.a_op = None
            self.model_path = model_path

            if self.om:
                self.x_m_ph, self.x2_m_ph, self.a_op = core.placeholders(self.market_price_dim, self.market_price_dim,
                                                                         self.market_price_dim)

            self.q_type = q_type

            self.pi, self.q, self.q_pi, self.q_pi_targ = \
                actor_critic(self.x_ph, self.a_ph, self.x2_ph, self.om, self.a_op, self.x_m_ph, self.x2_m_ph, self.q_type,
                             **ac_kwargs)

            # Experience buffer
            self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size,
                                              market_dim=self.market_price_dim)

            # Action Noise
            self.ou_noise = NNoise(action_dimension=self.act_dim)


            # print(self.pi.name)
            # Count variables
            # var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
            var_counts = tuple(core.count_vars(scope_i) for scope_i in [scope+'/main/pi', scope +'/main/q', scope+'/main'])
            print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n' % var_counts)

            # Bellman backup for Q function
            backup = tf.stop_gradient(self.r_ph + gamma * (1 - self.d_ph) * self.q_pi_targ)

            # DDPG losses
            self.pi_loss = -tf.reduce_mean(self.q_pi)
            self.q_loss = tf.reduce_mean((self.q - backup) ** 2)

            # Separate train ops for pi, q
            pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
            q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
            self.train_pi_op = pi_optimizer.minimize(self.pi_loss, var_list=get_vars(scope+'/main/pi'))
            self.train_q_op = q_optimizer.minimize(self.q_loss, var_list=get_vars(scope +'/main/q'))

            # Polyak averaging for target variables

            self.target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                      for v_main, v_targ in zip(get_vars(scope+'/main'), get_vars(scope +'/target'))])

            # Initializing targets to match main variables
            target_init = tf.group([tf.assign(v_targ, v_main)
                                    for v_main, v_targ in zip(get_vars(scope+'/main'), get_vars(scope + '/target'))])

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            self.tf_phs = {'x_ph': self.x_ph,
                      'a_ph': self.a_ph,
                      'x2_ph': self.x2_ph,
                      'r_ph': self.r_ph,
                      'd_ph': self.d_ph}
            if self.om:
                self.tf_phs.update({'x_m_ph': self.x_m_ph,
                                    'x2_m_ph': self.x2_m_ph,
                                    'a_op': self.a_op}
                                   )

            self.tf_outputs = {'pi': self.pi, 'q': self.q, 'q_pi': self.q_pi, 'q_pi_targ': self.q_pi_targ}
            self.logger.setup_tf_saver(self.sess, inputs=self.tf_phs,
                                       outputs=self.tf_outputs)

            self.sess.run(target_init)

            self.train_mode = train_mode
            self.model_name = model_name

        if not train_mode:
            self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
            # self.saver = tf.train.import_meta_graph(self.model_name + '.meta')
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
        else:
            self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))

    def act(self, o, noise_scale=0.1):

        a = self.sess.run(self.pi, feed_dict={self.x_ph: o.reshape(1, -1)})[0]
        # zero mean Gaussian noise
        a += noise_scale * np.random.randn(self.act_dim)
        # Decayed noise
        # a += self.ou_noise.noise() * noise_scale
        # return np.clip(a, -self.act_limit, self.act_limit)
        return np.clip(a, 0., self.act_limit)

    # not in use
    # def act_om(self, o, market, noise_scale=0.1):
    #     market=np.array(market)
    #     feed_dict_om = {self.x_ph: o.reshape(1, -1),
    #                     self.x_m_ph: np.array(market).reshape(1, -1)}
    #
    #     a = self.sess.run(self.pi, feed_dict=feed_dict_om)[0]
    #     a += noise_scale * np.random.randn(self.act_dim)
    #     # return np.clip(a, -self.act_limit, self.act_limit)
    #     return np.clip(a, 0., self.act_limit)

    def train(self,):
        batch = self.replay_buffer.sample_batch(self.batch_size)
        feed_dict = {
            self.x_ph: batch['obs1'],
            self.x2_ph: batch['obs2'],
            self.a_ph: batch['acts'],
            self.r_ph: batch['rews'],
            self.d_ph: batch['done'],
        }
        if self.om:
            feed_dict.update({
                self.x_m_ph: batch['x_m'],
                self.x2_m_ph: batch['x2_m'],
                self.a_op: batch['a_op']
            })

        # Q-learning update
        outs = self.sess.run([self.q_loss, self.q, self.train_q_op], feed_dict)
        self.logger.store(LossQ=outs[0], QVals=outs[1])
        # Policy update
        outs = self.sess.run([self.pi_loss, self.train_pi_op, self.target_update], feed_dict)
        self.logger.store(LossPi=outs[0])

    def final_sess_save(self,):
        self.saver.save(self.sess, self.model_name)
