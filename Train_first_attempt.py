from pentago_main_ML_AI_v1 import Board, DummyAI
import copy
from random import randint
import numpy as np
import tensorflow as tf
import os


board = Board()
player2 = DummyAI("Rando", -1)

def game_step(action):
    #turn action vector into move
    #get x, y, Quad, and direction from action vector
        #turn action into 6x6x4x2 and get indices of maximum value
    place_holder = np.zeros((288,1))
    place_holder[action]=100
    a = np.reshape(place_holder, (6,6,4,2))
    ind = np.unravel_index(np.argmax(a, axis = None), a.shape)
    x = ind[0]
    y = ind[1]
    Quad = ind[2] + 1
    direction = ind[3] + 1
    #check if x, y valid, reward if yes and if not give huge penalty
    if board.Get(x,y) == 0:   #if valid move
        board.AddPiece(x, y, 1)     #make the move
        reward = 0.3
    else:       #end game because NN_AI was dumb
        reward = -1.0
        obs = board
        done = True
        return obs, reward, done
    board.Rotate(Quad, direction)
    obs = board
    #check if NN_AI won
    gameOver = board.GameEnd()
    if gameOver == 1:
        done = True
        #this would only be true if NN_AI won
        reward = 1.0
        return obs, reward, done
    else:
        #check if draw
        if 0 not in board.boardmtx:     #if draw
            obs = board
            done = True
            reward = 0.1
            return obs, reward, done
        else:
            done = False
            #reward = -0.1  #should only happen after five moves though
    #DummyAI moves
    player2.play(board)
    obs = board  #new board position after second player has moved
    #check if DummyAI won
    gameOver = board.GameEnd()
    if gameOver == 1:
        done = True
        #this would only be true if Dummy_AI won
        reward = -0.7
    else:
        #check if draw
        if 0 not in board.boardmtx:     #if draw
            obs = board
            done = True
            reward = 0.1
        else: 
            done = False
    #print(board.boardmtx)
    return obs, reward, done

def preprocess_observation(obs):    
    return np.reshape(obs.boardmtx,(6,6,1))

#create DQN network to build two DQN (actor and critic)
#code based on A. Geron GitHub and O'Reilly book
input_height = 6
input_width = 6
input_channels = 1
conv_n_maps = [64, 64]
conv_kernel_sizes = [(2,2), (3,3)]
conv_strides = [1, 1]
conv_paddings = ["VALID"]*2 
conv_names = ["conv1","conv2"]
conv_activation = [tf.nn.relu]*2
n_hidden_in = 64 * 3 * 3  # conv1 has maps of 4x4 each leading to conv2 having maps of 3 x 3
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = 288  # 36 x4 x2 = 288 actions are available
initializer = tf.contrib.layers.variance_scaling_initializer()

def q_network(X_state, name):
    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        hidden2 = tf.layers.dense(hidden, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)

        outputs = tf.layers.dense(hidden2, n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

#create inpute placeholder, two DQNs, and copy critic to actor
X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
                                            input_channels])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

#from AGeron github
learning_rate = 0.001
momentum = 0.95

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1, keep_dims=True)
    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#deque list for replay memory
from collections import deque

replay_memory_size = 500000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

#epsilon-greedy policy for exploring game
eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2500

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action




#Inference loop
def Infer():
    done = False
    checkpoint_path = "./my_dqn.ckpt"
    n_max_steps = 10
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        board.reset()
        obs = board
        for step in range(n_max_steps):
            state = preprocess_observation(obs)
            # Online DQN evaluates what to do
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            action = np.argmax(q_values)
            # Online DQN plays
            obs, reward, done = game_step(action)
            print(board.boardmtx)
            if done:
                break

#training loop
def TrainNN():
    #training loop initial variables
    n_steps = 10000  # total number of training steps
    training_start = 2000  # start training after 100 game iterations
    training_interval = 4  # run a training step every 4 game iterations
    save_steps = 100  # save the model every 100 training steps
    copy_steps = 25  # copy online DQN to target DQN every 25 training steps
    discount_rate = 0.99
    batch_size = 50
    iteration = 0  # game iterations
    checkpoint_path = "./my_dqn.ckpt"
    done = True # env needs to be reset
        
    #tracking progress
    loss_val = np.infty
    game_length = 0
    total_max_q = 0
    mean_max_q = 0.0
    
    with tf.Session() as sess:
        init.run()
        copy_online_to_target.run()
        while True:
            step = global_step.eval()
            if step >= n_steps:
                break
            iteration += 1
            #print("\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   ".format(
            #    iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q), end="")
            if done: # game over, start again
                board.reset()
                obs = board
                reward = 0
                done = False
                state = preprocess_observation(obs)

            # Online DQN evaluates what to do
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            action = epsilon_greedy(q_values, step)
            old_state = state.copy()

            # Online DQN plays
            obs, reward, done = game_step(action)
            next_state = preprocess_observation(obs)

            # Let's memorize what happened
            replay_memory.append((old_state, action, reward, next_state, 1.0 - done))
            state = next_state

            # Compute statistics for tracking progress (not shown in the book)
            total_max_q += q_values.max()
            game_length += 1
            if done:
                mean_max_q = total_max_q / game_length
                total_max_q = 0.0
                game_length = 0

            if iteration < training_start or iteration % training_interval != 0:
                continue # only train after warmup period and at regular intervals
            
            # Sample memories and use the target DQN to produce the target Q-Value
            X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                sample_memories(batch_size))
            next_q_values = target_q_values.eval(
                feed_dict={X_state: X_next_state_val})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
            y_val = rewards + continues * discount_rate * max_next_q_values

            # Train the online DQN
            _, loss_val = sess.run([training_op, loss], feed_dict={
                X_state: X_state_val, X_action: X_action_val, y: y_val})

            # Regularly copy the online DQN to the target DQN
            if step % copy_steps == 0:
                copy_online_to_target.run()

            # And save regularly
            if step % save_steps == 0:
                saver.save(sess, checkpoint_path)

inference_mode = Falsejkhjhg

if inference_mode == False:
    TrainNN()
else:
    Infer()
