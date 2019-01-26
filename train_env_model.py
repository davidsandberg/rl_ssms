"""Train an environment model
"""
# MIT License
# 
# Copyright (c) 2019 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.python.data import Dataset
import utils
import argparse


def conv_stack(X, k1, c1, k2, c2, k3, c3):
    """Implements the conv_stack module as described in figure 6 in the paper
    """
    conv1 = tf.contrib.layers.conv2d(X, num_outputs=c1, kernel_size=k1,
            stride=1, padding='same', activation_fn=None)
    conv1_relu = tf.nn.relu(conv1)
    conv2 = tf.contrib.layers.conv2d(conv1_relu, num_outputs=c2, kernel_size=k2,
            stride=1, padding='same', activation_fn=None)
    conv2_relu = tf.nn.relu(conv1_relu + conv2)
    conv3 = tf.contrib.layers.conv2d(conv2_relu, num_outputs=c3, kernel_size=k3,
            stride=1, padding='same', activation_fn=None)
    return conv3

def res_conv(X, use_extra_convolution=True):
    """Implements the res_conv module as described in figure 7 in the paper
    """
    if use_extra_convolution:
        c = tf.contrib.layers.conv2d(X, num_outputs=64, kernel_size=1,
                stride=1, padding='same', activation_fn=None)
    else:
        c = X
    rc1_relu = tf.contrib.layers.conv2d(c, num_outputs=32, kernel_size=3,
            stride=1, padding='same', activation_fn=tf.nn.relu)
    rc2_relu = tf.contrib.layers.conv2d(rc1_relu, num_outputs=32, kernel_size=5,
            stride=1, padding='same', activation_fn=tf.nn.relu)
    rc3 = tf.contrib.layers.conv2d(rc2_relu, num_outputs=64, kernel_size=3,
            stride=1, padding='same', activation_fn=None)
    rc = c + rc3
    return rc

def pool_inject(X):
    """Implements the pool & inject module as described in figure 8 in the paper
    """
    width, height = X.get_shape()[1:3]
    m = tf.layers.max_pooling2d(X, pool_size=(width, height), strides=(1, 1))
    tiled = tf.tile(m, (1, width, height, 1))
    pi = tf.concat([tiled, X], axis=-1)
    return pi

def state_transition_module(a, s, z):
    """Implements the state transition function g(s,z,a) as described in figure 9 in the paper
    An action, a state and a latent variable at time t-1 is transitioned to 
    a state at time t.
    """
    with tf.variable_scope('state_transition_module'):
        if z is None:
            c = tf.concat([a, s], axis=-1)
        else:
            c = tf.concat([a, s, z], axis=-1)
        rc1_relu = tf.nn.relu(res_conv(c))
        pi = pool_inject(rc1_relu)
        s_next = res_conv(pi)
    return s_next

def observation_encoder(o):
    """
    """
    with tf.variable_scope('observation_encoder_module'):
        std1 = tf.nn.space_to_depth(o, 4)
        cs1 = conv_stack(std1, 3, 16, 5, 16, 3, 64)
        std2 = tf.nn.space_to_depth(cs1, 2)
        cs2 = conv_stack(std2, 3, 32, 5, 32, 3, 64)
        e = tf.nn.relu(cs2)
    return e
  
def softclip(x, limit=0.1):
    return tf.nn.softplus(x-limit)+limit

def prior_module(s, a):
    """
    """
    with tf.variable_scope('prior_module'):
        c = tf.concat([s, a], axis=-1)
        mu = conv_stack(c, 1, 32, 3, 32, 3, 64)
        sigma = softclip(conv_stack(c, 1, 32, 3, 32, 3, 64))
    return mu, sigma

def posterior_module(mu, sigma, s, e, a):
    """
    """
    with tf.variable_scope('posterior_module'):
        c = tf.concat([mu, sigma, s, e, a], axis=-1)
        mu_hat = conv_stack(c, 1, 32, 3, 32, 3, 64)
        sigma_hat = softclip(conv_stack(c, 1, 32, 3, 32, 3, 64))
    return mu_hat, sigma_hat

def initial_state_module(ex):
    """Computes the initial state from the feature maps of a
    number of consequtive initial observations. 
    These feature maps are given in the batch dimension, meaning
    that the first dimension should be of size batch_size x nrof_observations.
    """
    with tf.variable_scope('initial_state_module'):
        e = tf.unstack(ex, axis=1)
        c = tf.concat(e, axis=-1)
        s = conv_stack(c, 1, 64, 3, 64, 3, 64)
    return s

def observation_decoder(s, z):
    with tf.variable_scope('observation_decoder_module'):
        if z is None:
            c = s
        else:
            c = tf.concat([s, z], axis=-1)
        cs1 = conv_stack(c, 1, 32, 5, 32, 3, 64)
        dts1 = tf.nn.depth_to_space(cs1, 2)
        cs2 = conv_stack(dts1, 3, 64, 3, 64, 1, 48)
        dts2 = tf.nn.depth_to_space(cs2, 4)
    return dts2

def kl_divergence_gaussians(q_mu, q_sigma, p_mu, p_sigma):
    # https://github.com/openai/baselines/blob/f2729693253c0ef4d4086231d36e0a4307ec1cb3/baselines/acktr/utils.py
    num = tf.square(q_mu - p_mu) + tf.square(q_sigma) - tf.square(p_sigma)
    den = 2 * tf.square(p_sigma) + 1e-8
    kl = tf.reduce_sum(num/den + tf.log(p_sigma) - tf.log(q_sigma), axis=[2,3,4])
    return kl
  
def kl_divergence_bernoulli(p, q):
    eps = 1e-6
    pc = tf.clip_by_value(p, eps, 1-eps)
    qc = tf.clip_by_value(q, eps, 1-eps)
    kl = tf.reduce_sum(pc*tf.log(pc/qc) + (1-pc)*tf.log((1-pc)/(1-qc)), axis=[2,3,4])
    return kl
  
def get_onehot_actions(actions, nrof_actions, state_shape):
    length = actions.get_shape()[1]
    _, height, width, _ = state_shape
    oh = tf.one_hot(tf.reshape(actions, [-1, length, 1, 1]), nrof_actions, axis=-1)
    onehot_actions = tf.tile(oh, multiples=(1, 1, height, width, 1))
    return onehot_actions
  
class EnvModel():
    
    def __init__(self, is_pdt, obs, actions, nrof_actions=None, nrof_init_time_steps=3, nrof_time_steps=None, nrof_free_nats=0.0):
        _, length, width, height, depth = obs.get_shape().as_list()
    
        self.obs = obs
        self.actions = actions
        
        # Encode observations
        obs_reshaped = tf.reshape(self.obs, [-1, width, height, depth])
        self.encoded_obs_reshaped = observation_encoder(obs_reshaped)
        
        shape = [-1,length]+self.encoded_obs_reshaped.get_shape().as_list()[1:]
        self.encoded_obs = tf.reshape(self.encoded_obs_reshaped, shape)
        
        # Initialize state
        self.encoded_obs_init = self.encoded_obs[:,:nrof_init_time_steps,:,:,:]
        self.initial_state = initial_state_module(self.encoded_obs_init)
        state = self.initial_state
        
        # Convert actions to one-hot
        onehot_actions = get_onehot_actions(self.actions, nrof_actions, state.get_shape().as_list())

        obs_hat_list = []
        next_state_list = []
        mu_list = []
        sigma_list = []
        mu_hat_list = []
        sigma_hat_list = []
        z_list = []
        for t in range(nrof_time_steps):
          
            if t > 0:
                # Variables are reused for time step 1 and onwards
                tf.get_variable_scope().reuse_variables()
          
            # Compute prior statistics
            mu, sigma = prior_module(state, onehot_actions[:,t,:,:,:])
            mu_list += [ mu ]
            sigma_list += [ sigma ]
            
            # Compute posterior statistics
            mu_hat, sigma_hat = posterior_module(mu, sigma, state, self.encoded_obs[:,t,:,:,:], onehot_actions[:,t,:,:,:])
            mu_hat_list += [ mu_hat ]
            sigma_hat_list += [ sigma_hat ]
            
            # Sample from z using the reparametrization trick
            eps = tf.random_normal(tf.shape(sigma), 0.0, 1.0, dtype=tf.float32)
            mu_x = tf.where(is_pdt[:,t], mu, mu_hat)
            sigma_x = tf.where(is_pdt[:,t], sigma, sigma_hat)
            z = mu_x + tf.multiply(sigma_x, eps)
            z_list += [ z ]
            
            # Calculate next state
            next_state = state_transition_module(onehot_actions[:,t,:,:,:], state, z)
            next_state_list += [ next_state ]
            
            # Calculate observation
            obs_hat = observation_decoder(next_state, z)
            obs_hat_list += [ obs_hat ]
            
            state = next_state
            
        # Stack lists of tensors
        self.mu = tf.stack(mu_list, axis=1)
        self.sigma = tf.stack(sigma_list, axis=1)
        self.mu_hat = tf.stack(mu_hat_list, axis=1)
        self.sigma_hat = tf.stack(sigma_hat_list, axis=1)
        self.z = tf.stack(z_list, axis=1)
        self.next_state = tf.stack(next_state_list, axis=1)
        self.obs_hat = tf.nn.sigmoid(tf.stack(obs_hat_list, axis=1))

        # Calculate loss
        f = nrof_free_nats * np.prod(self.mu.get_shape().as_list()[2:])
        print('Reg loss limit: %.3f' % f)
        self.regularization_loss = tf.maximum(tf.constant(f, tf.float32), kl_divergence_gaussians(self.mu, self.sigma, self.mu_hat, self.sigma_hat))
        self.reconstruction_loss = kl_divergence_bernoulli(self.obs[:,nrof_init_time_steps:,:,:,:], self.obs_hat)
        
def create_dataset(filelist, path, buffer_size=25, batch_size=10):
    def gen(filelist, path):
        for fn in filelist:
            data = np.float32(utils.load_pickle(os.path.join(path, fn)))
            data = np.expand_dims(data, 4)
            data = np.repeat(data, 3, axis=4)
            for i in range(data.shape[0]):
                yield data[i,:13,:,:,:], np.zeros((13,), dtype=np.int32)
          
    ds = Dataset.from_generator(lambda: gen(filelist, path), (tf.float32, tf.int32), (tf.TensorShape([13, 80, 80, 3]), tf.TensorShape([13,])))
    ds = ds.repeat(count=None)
    ds = ds.prefetch(buffer_size)
    ds = ds.batch(batch_size)
    return ds
  
def create_transition_type_matrix(batch_size, seq_length, training_scheme='75%PDT'):
    is_pdt = np.ones((batch_size, seq_length), np.bool)
    if training_scheme=='75%PDT':
        is_pdt[:,0::4] = False
    else:
        raise ValueError('Invalid training scheme "%s".' % training_scheme)
    return is_pdt
    
def main(args):

    src_path,_ = os.path.split(os.path.realpath(__file__))
    
    # Create result directory
    res_name = utils.gettime()
    res_dir = os.path.join(src_path, 'results', res_name)
    os.makedirs(res_dir, exist_ok=True)
    
    log_filename = os.path.join(res_dir, 'log.h5')
    model_filename = os.path.join(res_dir, res_name)
    
    # Store some git revision info in a text file in the log directory
    utils.store_revision_info(src_path, res_dir, ' '.join(sys.argv))
    
    utils.store_hdf(os.path.join(res_dir, 'parameters.h5'), vars(args))

    # Copy learning rate schedule file to result directory
    learning_rate_schedule = utils.copy_learning_rate_schedule_file(args.learning_rate_schedule, res_dir)


    with tf.Session() as sess:
      
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
      
        filelist = [ 'bouncing_balls_training_data_%03d.pkl' % i for i in range(20) ]
        dataset = create_dataset(filelist, 'data', buffer_size=20000, batch_size=args.batch_size)

        # Create an iterator over the dataset
        iterator = dataset.make_one_shot_iterator()
        obs, action = iterator.get_next()
        
        is_pdt_ph = tf.placeholder(tf.bool, [None, args.seq_length])
        is_pdt = create_transition_type_matrix(args.batch_size, args.seq_length)

        with tf.variable_scope('env_model'):
            env_model = EnvModel(is_pdt_ph, obs, action, 1, nrof_time_steps=args.seq_length, nrof_free_nats=args.nrof_free_nats)

        reg_loss = tf.reduce_mean(env_model.regularization_loss)
        rec_loss = tf.reduce_mean(env_model.reconstruction_loss)
        loss = reg_loss + rec_loss

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate_ph = tf.placeholder(tf.float32, ())
        train_op = tf.train.AdamOptimizer(learning_rate_ph).minimize(loss, global_step=global_step)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        
        stat = {
            'loss': np.zeros((args.max_nrof_steps,), np.float32),
            'rec_loss': np.zeros((args.max_nrof_steps,), np.float32),
            'reg_loss': np.zeros((args.max_nrof_steps,), np.float32),
            'learning_rate': np.zeros((args.max_nrof_steps,), np.float32),
            }


        try:
            print('Started training')
            rec_loss_tot, reg_loss_tot, loss_tot = (0.0, 0.0, 0.0)
            lr = None
            for i in range(1, args.max_nrof_steps+1):
                if not lr or i % 100 == 0:
                    lr = utils.get_learning_rate_from_file(learning_rate_schedule, i)
                    if lr < 0:
                        break
                stat['learning_rate'][i-1] = lr
                _, rec_loss_, reg_loss_, loss_ = sess.run([train_op, rec_loss, reg_loss, loss], feed_dict={is_pdt_ph: is_pdt, learning_rate_ph:lr})
                stat['loss'][i-1], stat['rec_loss'][i-1], stat['reg_loss'][i-1] = loss_, rec_loss_, reg_loss_
                rec_loss_tot += rec_loss_
                reg_loss_tot += reg_loss_
                loss_tot += loss_
                if i % 10 == 0:
                    print('step: %-5d  lr: %-12.6f  rec_loss: %-12.1f  reg_loss: %-12.1f  loss: %-12.1f' % (i, lr, rec_loss_tot/10, reg_loss_tot/10, loss_tot/10))
                    rec_loss_tot, reg_loss_tot, loss_tot = (0.0, 0.0, 0.0)
                if i % 5000 == 0 and i>0:
                    saver.save(sess, model_filename, i)
                if i % 100 == 0:
                    utils.store_hdf(log_filename, stat)

                
        except tf.errors.OutOfRangeError:
            pass
          
        print("Saving model...")
        saver.save(sess, model_filename, i)

        print('Done!')
        
        
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_nrof_steps', type=int,
        help='Number of steps to train for.', default=20000)
    parser.add_argument('--batch_size', type=int,
        help='The number of sequences to process in one batch.', default=16)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=42)
    parser.add_argument('--learning_rate_schedule', type=str,
        help='File containing the learning rate schedule.', default='learning_rate_schedule_bouncing_balls.txt')
    parser.add_argument('--seq_length', type=int,
        help='The length of each sequence (excluding warm-up).', default=10)
    parser.add_argument('--nrof_free_nats', type=float,
        help='The number of free nats per dimension.', default=0.05)

    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

        