import numpy as np

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size, market_dim):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.x_market_buf = np.zeros([size, market_dim], dtype=np.float32)
        self.x2_market_buf = np.zeros([size, market_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        self.op_act_buf = np.zeros([size, 301], dtype=np.float32)

    def store(self, obs, act, rew, next_obs, done, x_m, x2_m, a_op):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.x_market_buf[self.ptr] = x_m
        self.x2_market_buf[self.ptr] = x2_m

        self.op_act_buf[self.ptr] = a_op

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)

        num_negatives  = 5        # add 10 negative samples in
        clones = num_negatives+1
        acts = self.acts_buf[idxs]
        neg_acts = [np.random.uniform(acts) for _ in range((num_negatives))] # sample actions between 0 and market price, if lost then also 0, if won then assume 0 reward if lost.
        all_acts = [acts] + neg_acts

        return dict(obs1=np.concatenate([self.obs1_buf[idxs]]*clones),
                    obs2=np.concatenate([self.obs2_buf[idxs]]*clones),
                    acts=np.concatenate(all_acts),
                    rews=np.concatenate([self.rews_buf[idxs]]+[np.zeros(batch_size)]*num_negatives), # add zero rewards in for actions samples between 0 and self.acts_buf[idxs]
                    done=np.concatenate([self.done_buf[idxs]]*clones),
                    x_m=np.concatenate([self.x_market_buf[idxs]]*clones),
                    x2_m=np.concatenate([self.x2_market_buf[idxs]]*clones),
                    a_op=np.concatenate([self.op_act_buf[idxs]]*clones)
                    )
