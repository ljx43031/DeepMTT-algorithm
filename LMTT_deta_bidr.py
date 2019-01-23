#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:34:11 2018

@author: ljx
"""

#==============================================================================
#train the network with bidirectional lstm, train the detas of trajectory
#==============================================================================
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from batchdata_derive3 import*
from maxout import max_out as MO     #maxout activation function

def Noisy_af(Xt):                    #noisy activation function
    p = 1
    c = 1
    h = tf.nn.relu(0.5*Xt+0.5)-tf.nn.relu(0.5*Xt-0.5) -0.5
#    h = tf.nn.relu(Xt+1)-tf.nn.relu(Xt-1)-1
    y = h + c*tf.square(tf.sigmoid(p*(h-Xt))-0.5) * tf.random_normal(tf.shape(Xt),mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
    return y

def piecewise(Xt):                  #piecewise activation function
#    y = (tf.nn.relu(0.5*Xt+0.5)-tf.nn.relu(0.5*Xt-0.5)-0.5) #piecewise activation function
    y = tf.nn.relu(Xt+1)-tf.nn.relu(Xt-1)-1
    return y

def fir_filter(x,w,b_size,t_size): #快速FIR滤波
    with tf.name_scope('FIR_filter'):
        #x,待滤波的数据,shape为[batch_size,time_size,output_size]
        #w,滤波网络,shape为[fir_size, output_size]
#        shape_x = x.get_shape().as_list()
        shape_w = w.get_shape().as_list()
        x_add = tf.constant(0, dtype=tf.float32, shape=[b_size,shape_w[0]-1,shape_w[1]], name='X_add')
        x = tf.concat([x_add,x],1)  #给前面不足长度的待滤波序列补全零
        x = tf.expand_dims(x,2)     #扩展一维，存储待滤波数据
        y = []
        for i in range(shape_w[0]):
            y.append(x[:,i:i+t_size,:,:])
        z = tf.concat(y,2)
        z = z*w
        return tf.reduce_sum(z,reduction_indices=2)
#==============================================================================
#==============================================================================
#建立lstm网络

from tensorflow.contrib import rnn
# Set CPU/GPF mode
#sess = tf.Session()  #CPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction=0.9
sess = tf.Session(config=config)

#超参数的定义-----------------------------------------
# Set RNN parameter
BN = 0.2
lr = 1e-5
#lr = 1e3
#处理数据的batch大小
_batch_size = np.array([int(100*BN)])
# The size of batch for learning，这是在图里的batch_size张量定义
batch_size = tf.placeholder(tf.int32,[1])
# 每个时刻的输入特征是4维的，就是每个时刻输入一行，一行有距离x,y和速度vx,vy
input_size = 4
# 时序持续长度
timestep_size = 50

## 隐含层的数量
#hidden_size = 64
# LSTM layer 的层数
layer_num = 3
#第一层隐层的节点数
hidden_size_1 = 128
#第二层隐层的节点数
hidden_size_2 = 256
#第三层隐层的节点数
hidden_size_3 = 256
#输出层maxout节点数
maxout_size = 64
#正则项系数
lambda1 = 0.003
#FIR滤波层滤波阶数
fir_size = 5

# 最后输出向量的维度
output_size = 4

#输入输出定义------------------------------------------------------------------
with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, [None, timestep_size, input_size], name='input_x')
    y = tf.placeholder(tf.float32, [None, timestep_size, output_size], name='input_y')
    Xtrac = tf.placeholder(tf.float32, [None, timestep_size, input_size], name='input_x')
    myd = tf.placeholder(tf.float32, [None, timestep_size-1, output_size], name='input_y')
    #定义一层FIR卷积层做滤波
    Fir_w = tf.Variable(tf.constant(1.0/fir_size,dtype=tf.float32,shape=[fir_size, input_size]), dtype=tf.float32, name='fir_w')
    tf.summary.histogram('fir_w',Fir_w)
    X_f = fir_filter(X, Fir_w, _batch_size, timestep_size)

#构建lstm网络，多层分开--------------------------------------------------------
with tf.variable_scope('lstm') as lstm_net:
    keep_prob = tf.placeholder(tf.float32)
    #第一层：
    with vs.variable_scope("lstm_cell1_fw") as lstm_cell1_fw:
    #forward:
        lstm_cell_1_fw = rnn.BasicLSTMCell(num_units=hidden_size_1, forget_bias=1.0, state_is_tuple=True, activation=Noisy_af)
        lstm_cell_1_fw = rnn.DropoutWrapper(cell=lstm_cell_1_fw, input_keep_prob=1.0, output_keep_prob=keep_prob)
        init_state_1_fw = lstm_cell_1_fw.zero_state(batch_size, dtype=tf.float32)
        outputs_l1_fw, _ = tf.nn.dynamic_rnn(lstm_cell_1_fw, inputs=X_f, initial_state=init_state_1_fw, time_major=False, scope=lstm_cell1_fw)
    with vs.variable_scope("lstm_cell1_bw") as lstm_cell1_bw:    
    #backward:
        lstm_cell_1_bw = rnn.BasicLSTMCell(num_units=hidden_size_1, forget_bias=1.0, state_is_tuple=True, activation=Noisy_af)
        lstm_cell_1_bw = rnn.DropoutWrapper(cell=lstm_cell_1_bw, input_keep_prob=1.0, output_keep_prob=keep_prob)
        init_state_1_bw = lstm_cell_1_bw.zero_state(batch_size, dtype=tf.float32)
        outputs_l1_bw, _ = tf.nn.dynamic_rnn(lstm_cell_1_bw, inputs=tf.reverse(X_f,[1]), initial_state=init_state_1_bw, time_major=False, scope=lstm_cell1_bw)
    #output:
    outputs_l1_bw = tf.reverse(outputs_l1_bw,[1])
    outputs_l1 = tf.concat([outputs_l1_fw,outputs_l1_bw],2)
    tf.summary.histogram('lstm1_outputs', outputs_l1)
    #第二层：
    with vs.variable_scope("lstm_cell2_fw") as lstm_cell2_fw:
    #forward:
        lstm_cell_2_fw = rnn.BasicLSTMCell(num_units=hidden_size_2, forget_bias=1.0, state_is_tuple=True, activation=Noisy_af)
        lstm_cell_2_fw = rnn.DropoutWrapper(cell=lstm_cell_2_fw, input_keep_prob=1.0, output_keep_prob=keep_prob)
        init_state_2_fw = lstm_cell_2_fw.zero_state(batch_size, dtype=tf.float32)
        outputs_l2_fw, _ = tf.nn.dynamic_rnn(lstm_cell_2_fw, inputs=outputs_l1, initial_state=init_state_2_fw, time_major=False, scope=lstm_cell2_fw)
    with vs.variable_scope("lstm_cell2_bw") as lstm_cell2_bw:
    #backward:
        lstm_cell_2_bw = rnn.BasicLSTMCell(num_units=hidden_size_2, forget_bias=1.0, state_is_tuple=True, activation=Noisy_af)
        lstm_cell_2_bw = rnn.DropoutWrapper(cell=lstm_cell_2_bw, input_keep_prob=1.0, output_keep_prob=keep_prob)
        init_state_2_bw = lstm_cell_2_bw.zero_state(batch_size, dtype=tf.float32)
        outputs_l2_bw, _ = tf.nn.dynamic_rnn(lstm_cell_2_bw, inputs=tf.reverse(outputs_l1,[1]), initial_state=init_state_2_bw, time_major=False, scope=lstm_cell2_bw)
    #output:
    outputs_l2_bw = tf.reverse(outputs_l2_bw,[1])
    outputs_l2 = tf.concat([outputs_l2_fw,outputs_l2_bw],2)
    tf.summary.histogram('lstm2_outputs', outputs_l2)
    #第三层：
    with vs.variable_scope("lstm_cell3_fw") as lstm_cell3_fw:
    #forward:
        lstm_cell_3_fw = rnn.BasicLSTMCell(num_units=hidden_size_3, forget_bias=1.0, state_is_tuple=True)
#        lstm_cell_3_fw = rnn.DropoutWrapper(cell=lstm_cell_3_fw, input_keep_prob=1.0, output_keep_prob=keep_prob)
        init_state_3_fw = lstm_cell_3_fw.zero_state(batch_size, dtype=tf.float32)
        outputs_l3_fw, _ = tf.nn.dynamic_rnn(lstm_cell_3_fw, inputs=outputs_l2, initial_state=init_state_3_fw, time_major=False, scope=lstm_cell3_fw)
    with vs.variable_scope("lstm_cell3_bw") as lstm_cell3_bw:
        #backward:
        lstm_cell_3_bw = rnn.BasicLSTMCell(num_units=hidden_size_3, forget_bias=1.0, state_is_tuple=True)
#        lstm_cell_3_bw = rnn.DropoutWrapper(cell=lstm_cell_3_bw, input_keep_prob=1.0, output_keep_prob=keep_prob)
        init_state_3_bw = lstm_cell_3_bw.zero_state(batch_size, dtype=tf.float32)
        outputs_l3_bw, _ = tf.nn.dynamic_rnn(lstm_cell_3_bw, inputs=tf.reverse(outputs_l2,[1]), initial_state=init_state_3_bw, time_major=False, scope=lstm_cell3_bw)    
    #output:
    outputs_l3_bw = tf.reverse(outputs_l3_bw,[1])
    outputs_l3 = tf.concat([outputs_l3_fw,outputs_l3_bw],2)
    tf.summary.histogram('lstm3_outputs', outputs_l3)

    outputs = outputs_l3
    hidden_size = hidden_size_3*2
    
    #网络参数提取
    lstm_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=lstm_net.name)

#输出层的定义------------------------------------------------------------------
with tf.name_scope('output_layer'):
    #maxout层
    maxout_w = tf.Variable(tf.truncated_normal([hidden_size, timestep_size, hidden_size], stddev=0.1), dtype=tf.float32, name='maxout_w')
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(maxout_w))
    tf.summary.histogram('maxout_w', maxout_w)
    maxout_b = tf.Variable(tf.constant(0.1,shape=[timestep_size, hidden_size]), dtype=tf.float32, name='maxout_b')
    tf.summary.histogram('maxout_b', maxout_b)
    T_o = tf.transpose(outputs,[1,0,2])
    T_w = tf.transpose(maxout_w,[1,0,2])
    maxout_o = tf.transpose(tf.matmul(T_o, T_w),[1,0,2]) + maxout_b
    mo_f = MO(maxout_o, maxout_size, axis=2)   #maxout
    #输出层
    W = tf.Variable(tf.truncated_normal([maxout_size, timestep_size, output_size], stddev=0.1), dtype=tf.float32, name='output_w')
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(W))
    tf.summary.histogram('output_w', W)
    bias = tf.Variable(tf.constant(0.1,shape=[timestep_size, output_size]), dtype=tf.float32, name='output_b')
    tf.summary.histogram('output_b',bias)
    #tensroflow的三维tensor相乘，第一维不管，后面两维相乘，所以这里把timestep_size先放到第一维
    tran_o = tf.transpose(mo_f,[1,0,2])
    tran_w = tf.transpose(W,[1,0,2])
    y_pre = tf.transpose(tf.matmul(tran_o, tran_w),[1,0,2]) + bias

    lstm_outputs={'maxout_w':maxout_w, 'maxout_b':maxout_b, 'output_w':W,'output_b':bias}  #把要保存的参数做成字典
    
    y_final = y_pre + Xtrac
    y_deta = y_final[:,1:,:] - y_final[:,:-1,:]

#损失和评估函数,RMSE最小-------------------------------------------------------
with tf.name_scope('my_sqr'):
    my_sqr = tf.square(y-y_pre)
    
with tf.name_scope('my_sqr_deta'):
    my_sqr_deta = tf.square(myd-y_deta)   

with tf.name_scope('RMSE'):
    RMSE = tf.sqrt(tf.reduce_mean(my_sqr))
    tf.summary.scalar('RMSE', RMSE)
    tf.add_to_collection('losses', RMSE)
    
with tf.name_scope('RMSE_deta'):
    RMSE_deta = tf.sqrt(tf.reduce_mean(my_sqr_deta))
    tf.summary.scalar('RMSE_deta', RMSE_deta)    

final_loss = tf.add_n(tf.get_collection('losses'))
with tf.name_scope('train_step'):
#    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss) # 使用梯度下降法，设置步长lr，来最小化损失
    train_step  = tf.train.AdamOptimizer(lr).minimize(RMSE)
#    train_step  = tf.train.AdamOptimizer(lr).minimize(final_loss)
    
with tf.name_scope('train_deta'):
    train_deta  = tf.train.AdamOptimizer(lr).minimize(RMSE_deta)
    
merged = tf.summary.merge_all()
## save the logs  
#writer = tf.summary.FileWriter("logs/LMTT1_deta_bidr", sess.graph)


sess.run(tf.global_variables_initializer())

#==============================================================================
#模型加载
saver = tf.train.Saver()
model_path = "/home/ljx/文档/OpenSources/DeepMTT/Models/LMTT.ckpt"
load_path = saver.restore(sess, model_path)
print ("Model restored from file: %s" % model_path)

##直接加载部分模型参数
##-----1,save the lstm network parameters
#saver_lstm = tf.train.Saver(lstm_variables)  
#model_path = "/home/ljx/proj/Maneuvering_Target_Tracking/model_save/LMTT1_deta/LMTT_vb.ckpt"
#saver_lstm.restore(sess, model_path)
##-----2,save the output layer
#saver_outputs = tf.train.Saver(lstm_outputs)
#model_path = "/home/ljx/proj/Maneuvering_Target_Tracking/model_save/LMTT1_deta/LMTT_op.ckpt"
#saver_outputs.restore(sess, model_path)

#迭代总次数
iter_time = 1000000
#显示准确率的相隔次数
accu_st = 10
save_st = 20
#准确率存储，初始化
accuracy_save = np.array([0.0 for ttt in range(int(iter_time/accu_st))], 'float64')
itertime_save = np.array([0.0 for ttt in range(int(iter_time/accu_st))], 'float64')
t_0 = 0

#数据预处理1----整个batch数据中的最大值作为归一化  
def Data_Pro1(data):
    weight = np.max(np.abs(data))
    results = data/weight
    return results, weight

#数据处理2-----batch里面每一个数据按照第一个值的最大值进行归一化
def Data_Pro2(data):
    weight = np.max(np.abs(data[:,0,:]), axis=1)
    weight = np.transpose(np.array([[weight]]),[2,1,0])
    results = data/weight
    return results, weight

#==============================================================================
#训练：========================================================================
for i in range(iter_time):    
#    _,_,_, batch_input, output_results= creat_batch(_batch_size, timestep_size)
    ori_traj,_,tra_traj,output_results = creat_batch3(0,100000-1,30,3,BN)
#    ori_traj_c, ori_traj_f = data_change(ori_traj)
#    tra_traj_c, _ = data_change(tra_traj)
#    #数据处理1----整个batch数据中的最大值作为归一化   
#    my_batch, _ = Data_Pro1(tra_traj)
    
    #数据处理2-----batch里面每一个数据按照第一个值的最大值进行衰减，线性衰减
    my_batch, _ = Data_Pro2(tra_traj)
    detain = ori_traj[:,1:,:]-ori_traj[:,:-1,:]
#    output_results = ori_traj_c - tra_traj_c
    
    if (i+1)%accu_st == 0:
        my_results_t = sess.run(y_pre, feed_dict={
            X:my_batch, y: output_results, keep_prob: 1.0, batch_size: _batch_size})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
        print ("step %d, Tracking RMSE X: %g, Y: %g" % ((i+1), np.mean(np.abs(output_results-my_results_t)[:,:,0]), np.mean(np.abs(output_results-my_results_t)[:,:,1])))
        accuracy_save[t_0] = np.mean(np.abs(output_results-my_results_t)[:,:,0:2])
        itertime_save[t_0] = i+1
        t_0 = t_0 +1

#        my_tensor = sess.run(merged, feed_dict={
#            X:my_batch, y: output_results, Xtrac:tra_traj, myd:detain, keep_prob: 1.0, batch_size: _batch_size})
#        writer.add_summary(my_tensor, i)
        
        myaccuracy = {'accuracy_save':accuracy_save,'itertime_save':itertime_save}
        scio.savemat('my_accuracy_deta_bidr', myaccuracy)
        
    if (i+1)%save_st == 0:
        #继续保存模型
        saver = tf.train.Saver()
        model_path = "/home/ljx/文档/OpenSources/DeepMTT/Models/LMTT.ckpt"
#        save_path = saver.save(sess, model_path, global_step=i)
        save_path = saver.save(sess, model_path)
#        print "Model saved in file: %s" % save_path
        
        
#        #直接保存部分模型参数
#        #-----1,save the lstm network parameters
#        saver_lstm = tf.train.Saver(lstm_variables)  
#        model_path = "/home/ljx/proj/Maneuvering_Target_Tracking/model_save/LMTT1/LMTT_lstm.ckpt"
#        saver_lstm.save(sess, model_path)
#        #-----2,save the output layer
#        saver_outputs = tf.train.Saver(lstm_outputs)
#        model_path = "/home/ljx/proj/Maneuvering_Target_Tracking/model_save/LMTT1/LMTT_op.ckpt"
#        saver_outputs.save(sess, model_path)
    
    _ = sess.run(train_step, feed_dict={X:my_batch, y: output_results, keep_prob: 1.0, batch_size: _batch_size})
#    _ = sess.run(train_deta, feed_dict={X:my_batch, y: output_results, Xtrac:tra_traj, myd:detain, keep_prob: 1.0, batch_size: _batch_size})

## 计算测试数据的准确率
#Traj_r,_, my_tracking_results,my_real_deta = creat_batch()
##my_tracking_results_c, my_tracking_results_f = data_change(my_tracking_results)
##Traj_r_c,_ = data_change(Traj_r)
##my_real_deta = Traj_r_c - my_tracking_results_c
##数据处理1/2-----batch里面每一个数据按照第一个值的最大值进行归一化
#my_tracking_inputs,_ = Data_Pro2(my_tracking_results)
#my_result = sess.run(y_pre, feed_dict={
#    X: my_tracking_inputs, y: my_real_deta, keep_prob: 1.0, batch_size: _batch_size})
##my_result_return = my_result + my_tracking_results_c
##my_traj_final = my_result_return * my_tracking_results_f
#print "test x orignal RMSE %g"% (np.mean(np.abs(my_real_deta)[:,:,0]))
#print "test x orignal RMSE %g"% (np.mean(np.abs(my_real_deta)[:,:,1]))
#print "test x RMSE %g"% (np.mean(np.abs(my_real_deta-my_result)[:,:,0]))
#print "test x RMSE %g"% (np.mean(np.abs(my_real_deta-my_result)[:,:,1]))
##画图
#import matplotlib.pyplot as plt
##plt.figure(1) # 创建图表1
##plt.plot(itertime_save, accuracy_save)
##plt.xlabel('Iteration Times')# make axis labels3
##plt.ylabel('Accuracy')
#my_traj_pre = my_tracking_results + my_result
#for i in range(10):
#    plt.figure(i) # 创建图表1
#    plt.plot(Traj_r[i,:,0], Traj_r[i,:,1], ".-")
#    plt.plot(my_traj_pre[i,:,0], my_traj_pre[i,:,1], "x-")
#    plt.plot(my_tracking_results[i,:,0], my_tracking_results[i,:,1], "^-")
##==============================================================================

##继续保存模型
#saver = tf.train.Saver()
#model_path = "/home/ljx/文档/Python_Proj/v1_0/model_save/TT2.ckpt"
#save_path = saver.save(sess, model_path)
#print "Model saved in file: %s" % save_path

##直接保存部分模型参数
##-----1,save the lstm network parameters
#saver_lstm = tf.train.Saver(lstm_variables)  
#model_path = "/home/ljx/文档/Python_Proj/v1_0/model_save/TT_ny_lstm.ckpt"
#saver_lstm.save(sess, model_path)
##-----2,save the output layer
#saver_outputs = tf.train.Saver(lstm_outputs)
#model_path = "/home/ljx/文档/Python_Proj/v1_0/model_save/TT_ny_outputs.ckpt"
#saver_outputs.save(sess, model_path)
