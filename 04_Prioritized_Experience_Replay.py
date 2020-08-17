# Deep Q-Network Algorithm

# Import modules
import tensorflow as tf
import sys
import random
import numpy as np
import os
import time, datetime
from collections import deque
import cv2
import pickle

import pygame
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Import game
sys.path.append("DQN_GAMES/")

import pong as game

action_size = game.Return_Num_Action()

file_name =  sys.argv[0][:-3]

model_path = "save_model/" + game.ReturnName() + "/" + file_name
graph_path = "save_graph/" + game.ReturnName() + "/" + file_name

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

class DQN_agent:
    def __init__(self):

        # Get parameters
        self.progress = " "
        
        # get size of state and action
        self.action_size = action_size
        
        # train time define
        self.training_time = 5*60
        
        # Parameters for DQN_agent
        self.eps = 0.00001

        self.alpha = 0.6
        self.beta_init = 0.4
        self.beta = self.beta_init

        self.TD_list = np.array([])

        # Initial parameters
        self.n_exploitation = 100000
        
        # These are hyper parameters for the DQN
        self.learning_rate = 0.0001
        self.discount_factor = 0.99
        
        self.epsilon_max = 1.0
        # final value of epsilon
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.0001
        self.epsilon = self.epsilon_max
        
        self.step = 0
        self.score = 0
        self.episode = 0
        
        self.ep_trial_step = 5000
        
        # parameters for skipping and stacking
        self.state_set = []
        self.Num_skipping = 4
        self.Num_stacking = 4

        # Parameter for Experience Replay
        self.size_replay_memory = 5000
        self.batch_size = 64
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 200

        # Parameters for network
        self.img_rows , self.img_cols = 80, 80
        self.Num_colorChannel = 1

        self.first_conv   = [8,8,1 * 4,32]
        self.second_conv  = [4,4,32,64]
        self.third_conv   = [3,3,64,64]
        self.first_dense  = [10*10*64, 512]
        self.second_dense = [512, self.action_size]
        
        # Initialize Network
        self.input, self.output = self.build_model('network')
        self.tgt_input, self.tgt_output = self.build_model('target')
        self.train_step, self.action_tgt, self.y_tgt, self.Loss, self.w_is, self.TD_error = self.loss_and_train()
            
    def reset_env(self, game_state):
        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros([self.action_size])
        state, reward, done = game_state.frame_step(do_nothing)
        
        # state = self.preprocess(state)
        state_out = cv2.resize(state, (self.img_rows, self.img_cols))
        # if self.Num_colorChannel == 1:
        state_out = cv2.cvtColor(state_out, cv2.COLOR_BGR2GRAY)
        state_out = np.reshape(state_out, (self.img_rows, self.img_cols, 1))
        state = np.uint8(state_out)

        for i in range(self.Num_skipping * self.Num_stacking):
            self.state_set.append(state)

        return state

    def skip_and_stack_frame(self, state):
        self.state_set.append(state)

        state_in = np.zeros((self.img_rows, self.img_cols, self.Num_colorChannel * self.Num_stacking))

        # Stack the frame according to the number of skipping frame
        for stack_frame in range(self.Num_stacking):
            state_in[:,:, self.Num_colorChannel * stack_frame : self.Num_colorChannel * (stack_frame+1)] = self.state_set[-1 - (self.Num_skipping * stack_frame)]

        del self.state_set[0]

        state_in = np.uint8(state_in)
        return state_in

    # Resize and make input as grayscale
    def preprocess(self, state):
        state_out = cv2.resize(state, (self.img_rows, self.img_cols))
        state_out = cv2.cvtColor(state_out, cv2.COLOR_BGR2GRAY)
        state_out = np.reshape(state_out, (self.img_rows, self.img_cols, 1))
        state_out = np.uint8(state_out)
        return state_out

    # Convolution and pooling
    def conv2d(self, x, w, stride):
        return tf.nn.conv2d(x,w,strides=[1, stride, stride, 1], padding='SAME')

    # Get Variables
    def conv_weight_variable(self, name, shape):
        return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())

    def weight_variable(self, name, shape):
        return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

    def bias_variable(self, name, shape):
        return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self, network_name):
        # input layer
        x_image = tf.placeholder(tf.float32, shape = [None,
                                                      self.img_rows,
                                                      self.img_cols,
                                                      self.Num_stacking * self.Num_colorChannel])

        x_normalize = (x_image - (255.0/2)) / (255.0/2)

        with tf.variable_scope(network_name):
            # Convolution variables
            w_conv1 = self.conv_weight_variable('_w_conv1', self.first_conv)
            b_conv1 = self.bias_variable('_b_conv1',[self.first_conv[3]])

            w_conv2 = self.conv_weight_variable('_w_conv2',self.second_conv)
            b_conv2 = self.bias_variable('_b_conv2',[self.second_conv[3]])

            w_conv3 = self.conv_weight_variable('_w_conv3',self.third_conv)
            b_conv3 = self.bias_variable('_b_conv3',[self.third_conv[3]])

            # Densely connect layer variables
            w_fc1 = self.weight_variable('_w_fc1',self.first_dense)
            b_fc1 = self.bias_variable('_b_fc1',[self.first_dense[1]])

            w_fc2 = self.weight_variable('_w_fc2',self.second_dense)
            b_fc2 = self.bias_variable('_b_fc2',[self.second_dense[1]])

        # Network
        h_conv1 = tf.nn.relu(self.conv2d(x_normalize, w_conv1, 4) + b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)

        h_pool3_flat = tf.reshape(h_conv3, [-1, self.first_dense[0]])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1)

        # Q Value layer
        output = tf.matmul(h_fc1, w_fc2) + b_fc2
        return x_image, output

    def loss_and_train(self):
        # Loss function and Train
        action_tgt = tf.placeholder(tf.float32, shape = [None, self.action_size])
        y_tgt = tf.placeholder(tf.float32, shape = [None])

        y_prediction = tf.reduce_sum(tf.multiply(self.output, action_tgt), reduction_indices = 1)

        # ################################################## DQN_agent ############################################################
        w_is = tf.placeholder(tf.float32, shape = [None])
        TD_error_tf = tf.subtract(y_prediction, y_tgt)

        # Loss = tf.reduce_mean(tf.square(y_prediction - y_tgt))
        Loss = tf.reduce_sum(tf.multiply(w_is, tf.square(y_prediction - y_tgt)))
        ###################################################################################################################

        train_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate, epsilon = 1e-02).minimize(Loss)

        return train_step, action_tgt, y_tgt, Loss, w_is, TD_error_tf

    # get action from model using epsilon-greedy policy
    def get_action(self, stacked_state):
        # choose an action epsilon greedily
        action = np.zeros([self.action_size])
        action_index = 0
        
        if random.random() < self.epsilon:
            # print("----------Random Action----------")
            action_index = random.randint(0, self.action_size-1)
            action[action_index] = 1
        else:
            # Choose greedy action
            Q_value = self.output.eval(feed_dict= {self.input:[stacked_state]})[0]
            action_index = np.argmax(Q_value)
            action[action_index] = 1
            
        return action, action_index

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        # If Replay memory is longer than size_replay_memory, delete the oldest one
        if len(self.memory) >= self.size_replay_memory:
            del self.memory[0]
            self.TD_list = np.delete(self.TD_list, 0)

        if self.progress == 'Exploration':
            self.memory.append([state, action, reward, next_state, done])
            self.TD_list = np.append(self.TD_list, pow((abs(reward) + self.eps), self.alpha))
        elif self.progress == 'Training':
            self.memory.append([state, action, reward, next_state, done])
            # ################################################## DQN_agent ############################################################
            tgt_q_value_next = self.tgt_output.eval(feed_dict = {self.tgt_input: [next_state]})

            if done == True:
                y = [reward]
            else:
                y = [reward + self.discount_factor * np.max(tgt_q_value_next)]

            TD_error = self.TD_error.eval(feed_dict = {self.action_tgt: [action], self.y_tgt: y, self.input: [state]})[0]
            self.TD_list = np.append(self.TD_list, pow((abs(TD_error) + self.eps), self.alpha))
            # ###################################################################################################################

    def train_model(self, minibatch, w_array, batch_index):

        # Save the each batch data
        states      = [batch[0] for batch in minibatch]
        actions     = [batch[1] for batch in minibatch]
        rewards     = [batch[2] for batch in minibatch]
        next_states = [batch[3] for batch in minibatch]
        dones       = [batch[4] for batch in minibatch]

        # Get target values
        y_array = []
        # Selecting actions
        tgt_q_value_next = self.tgt_output.eval(feed_dict = {self.tgt_input: next_states})

        # Get target values
        for i in range(len(minibatch)):
            if dones[i] == True:
                y_array.append(rewards[i])
            else:
                y_array.append(rewards[i] + self.discount_factor * np.max(tgt_q_value_next[i]))

        # Training!! 
        feed_dict = {self.action_tgt: actions, self.y_tgt: y_array, self.input: states, self.w_is: w_array}
        _, self.loss, TD_error_array = self.sess.run([self.train_step, self.Loss, self.TD_error], feed_dict = feed_dict)
        # Update TD_list
        for i_array in range(len(batch_index)):
            self.TD_list[batch_index[i_array]] = pow((abs(TD_error_array[i_array]) + self.eps), self.alpha)

        # Update Beta
        self.beta = self.beta + (1 - self.beta_init) / self.n_exploitation
        
        # Decrease epsilon while training
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else :
            self.epsilon = self.epsilon_min

    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        # Get trainable variables
        trainable_variables = tf.trainable_variables()
        # network variables
        src_vars = [var for var in trainable_variables if var.name.startswith('network')]

        # target variables
        dest_vars = [var for var in trainable_variables if var.name.startswith('target')]

        for i in range(len(src_vars)):
            self.sess.run(tf.assign(dest_vars[i], src_vars[i]))

    ################################################## DQN_agent ############################################################
    def prioritized_minibatch(self):
        # Update TD_error list
        TD_normalized = self.TD_list / np.linalg.norm(self.TD_list, 1)
        TD_sum = np.cumsum(TD_normalized)

        # Get importance sampling weights
        weight_is = np.power((self.size_replay_memory * TD_normalized), - self.beta)
        weight_is = weight_is / np.max(weight_is)

        # Select mini batch and importance sampling weights
        minibatch = []
        batch_index = []
        w_array = []
        for i in range(self.batch_size):
            rand_array = random.random()
            TD_index = np.nonzero(TD_sum >= rand_array)[0][0]
            batch_index.append(TD_index)
            w_array.append(weight_is[TD_index])
            minibatch.append(self.memory[TD_index])

        return minibatch, w_array, batch_index
    ###################################################################################################################

    def save_model(self):
        # Save the variables to disk.
        save_path = self.saver.save(self.sess, model_path + "/model.ckpt")

        with open(model_path + '/append_sample.pickle', 'wb') as f:
            pickle.dump(self.memory, f)

        save_object = (self.epsilon, self.episode, self.TD_list, self.step) 
        with open(model_path + '/epsilon.pickle', 'wb') as ggg:
            pickle.dump(save_object, ggg)

        print("\n Model saved in file: %s" % save_path)

def main():
    
    agent = DQN_agent()
    
    # Initialize variables
    # Load the file if the saved file exists
    agent.sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    agent.saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        agent.saver.restore(agent.sess, ckpt.model_checkpoint_path)
        if os.path.isfile(model_path + '/append_sample.pickle'):  
            with open(model_path + '/append_sample.pickle', 'rb') as f:
                agent.memory = pickle.load(f)
            with open(model_path + '/epsilon.pickle', 'rb') as ggg:
                agent.epsilon, agent.episode, agent.TD_list, agent.step = pickle.load(ggg)

        print('\n\n Variables are restored!')

    else:
        agent.sess.run(init)
        print('\n\n Variables are initialized!')
        agent.epsilon = agent.epsilon_max
    
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # start training    
    # Step 3.2: run the game
    display_time = datetime.datetime.now()
    print("\n\n",file_name, "-game start at :",display_time,"\n")
    start_time = time.time()
    
    agent.Copy_Weights()
    
    while time.time() - start_time < agent.training_time:

        done = False
        agent.score = 0
        ep_step = 0
        
        # reset_env
        state = agent.reset_env(game_state)
        stacked_state = agent.skip_and_stack_frame(state)

        while not done and ep_step < agent.ep_trial_step:
            if len(agent.memory) < agent.size_replay_memory:
                agent.progress = "Exploration"
            else :
                agent.progress = "Training"

            ep_step += 1
            agent.step += 1

            # Select action
            action, action_index = agent.get_action(stacked_state)

            # run the selected action and observe next state and reward
            next_state, reward, done = game_state.frame_step(action)
            next_state = agent.preprocess(next_state)
            stacked_next_state = agent.skip_and_stack_frame(next_state)
            
            # store the transition in memory
            agent.append_sample(stacked_state, action, reward, stacked_next_state, done)
            
            # only train if done observing
            if agent.progress == "Training":
                # Make mini-batch with Prioritization
                minibatch, w_array, batch_index  = agent.prioritized_minibatch()
                # Training!
                agent.train_model(minibatch, w_array, batch_index)
                if agent.step % agent.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agent.Copy_Weights()
                    
            # update the old values
            stacked_state = stacked_next_state
            agent.score += reward
            
            # If game is over (done)
            if done or ep_step == agent.ep_trial_step:
                if agent.progress == "Training":
                    agent.episode += 1
                print('episode :{:>7,d}'.format(agent.episode),'/ ep step :{:>6,d}'.format(ep_step), \
                      '/ time step :{:>10,d}'.format(agent.step),'/ progress :',agent.progress, \
                      '/ epsilon :{:>1.5f}'.format(agent.epsilon),'/ score :{:> 5f}'.format(agent.score) )
                break
    # Save model
    agent.save_model()
    
    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()
                    
if __name__ == "__main__":
    main()
