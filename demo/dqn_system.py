import numpy as np
import argparse
import pickle
import gym
import pdb
import os

from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam, SGD
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

from env.drl_env_vsb import ViewScaleBrightnessSearchEnv
# from env.drl_env_vs import ViewScaleSearchEnv
from env.dqn import DQNAgent
from env.policy import BoltzmannQPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument("--config-file", default="configs/fcos/fcos_imprv_R_50_FPN_1x_DRL.yaml", metavar="FILE",)
    parser.add_argument("--weights", default="training_dir/fcos_imprv_R_50_FPN_1x_car/model_final.pth", metavar="FILE")
    parser.add_argument("--drl-weights", default='training_dir/drlnet/dqn_weights_final.h5f', metavar="FILE")
    parser.add_argument("--pickle-dir", default='search', metavar="FILE")
    parser.add_argument("--min-image-size", type=int, default=800)
    parser.add_argument("--double", action="store_true", default=True)
    parser.add_argument("--dueling", action="store_true", default=False)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

class AODSystem(object):
    def __init__(self, config_file, weights, drl_weights, pickle_dir):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True
        self.session = tf.compat.v1.Session(config=config)
        KTF.set_session(self.session)

        args = parse_args()
        args.config_file = config_file
        args.weights = weights
        args.drl_weights = drl_weights
        args.pickle_dir = pickle_dir
    
        self.env = ViewScaleBrightnessSearchEnv(args, is_train=False)
        # self.env = ViewScaleSearchEnv(args, is_train=False)
        model = self.get_model(self.env)
    
        memory = SequentialMemory(limit=20000, window_length=1)
        policy = GreedyQPolicy()
        self.dqn = DQNAgent(model=model, nb_actions=self.env.action_space.n, policy=policy, memory=memory,
                       nb_steps_warmup=5000, gamma=.99, target_model_update=5000, env=self.env,
                       enable_double_dqn=args.double, enable_dueling_network=args.dueling)
        self.dqn.compile(Adam(lr=0.001), metrics=['mae'])
    
        weights_filename = args.drl_weights
        self.dqn.load_weights(weights_filename)

    def Conv2d_BN(self, x, nb_filter, kernel_size, strides=(1,1), padding='same',name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        # x = BatchNormalization(axis=3, name=bn_name)(x)
        return x
     
    def Conv_Block(self, inpt, nb_filter, kernel_size, strides=(1,1), with_conv_shortcut=False):
        x = self.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = self.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = self.Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x
    
    def get_model(self, env):
        inputs1 = Input(shape=env.feature_length1)
        x1 = self.Conv2d_BN(inputs1, nb_filter=64, kernel_size=(3, 3), padding='same')
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        x1 = self.Conv_Block(x1, nb_filter=64, kernel_size=(3, 3))
        x1 = self.Conv_Block(x1, nb_filter=64, kernel_size=(3, 3))
        x1 = self.Conv_Block(x1, nb_filter=64, kernel_size=(3, 3))
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        x1 = self.Conv_Block(x1, nb_filter=64, kernel_size=(3, 3))
        x1 = self.Conv_Block(x1, nb_filter=64, kernel_size=(3, 3))
        x1 = self.Conv_Block(x1, nb_filter=64, kernel_size=(3, 3))
        x1 = Flatten()(x1)
        output1 = Dense(32, activation='relu')(x1)
        model1 = Model(input=inputs1, output=output1)
    
        inputs2 = Input(shape=env.feature_length2)
        x2 = self.Conv2d_BN(inputs2, nb_filter=64, kernel_size=(3, 3), padding='same')
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        x2 = self.Conv_Block(x2, nb_filter=64, kernel_size=(3, 3))
        x2 = self.Conv_Block(x2, nb_filter=64, kernel_size=(3, 3))
        x2 = self.Conv_Block(x2, nb_filter=64, kernel_size=(3, 3))
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        x2 = self.Conv_Block(x2, nb_filter=64, kernel_size=(3, 3))
        x2 = self.Conv_Block(x2, nb_filter=64, kernel_size=(3, 3))
        x2 = self.Conv_Block(x2, nb_filter=64, kernel_size=(3, 3))
        x2 = Flatten()(x2)
        output2 = Dense(32, activation='relu')(x2)
        model2 = Model(input=inputs2, output=output2)
    
        inputs3 = Input(shape=env.feature_length3)
        x3 = self.Conv2d_BN(inputs3, nb_filter=64, kernel_size=(3, 3), padding='same')
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        x3 = self.Conv_Block(x3, nb_filter=64, kernel_size=(3, 3))
        x3 = self.Conv_Block(x3, nb_filter=64, kernel_size=(3, 3))
        x3 = self.Conv_Block(x3, nb_filter=64, kernel_size=(3, 3))
        x3 = Flatten()(x3)
        output3 = Dense(32, activation='relu')(x3)
        model3 = Model(input=inputs3, output=output3)
    
        inputs4 = Input(shape=env.feature_length4)
        x4 = self.Conv2d_BN(inputs4, nb_filter=64, kernel_size=(3, 3), padding='same')
        x4 = self.Conv_Block(x4, nb_filter=64, kernel_size=(3, 3))
        x4 = self.Conv_Block(x4, nb_filter=64, kernel_size=(3, 3))
        x4 = self.Conv_Block(x4, nb_filter=64, kernel_size=(3, 3))
        x4 = Flatten()(x4)
        output4 = Dense(32, activation='relu')(x4)
        model4 = Model(input=inputs4, output=output4)
    
        inputs5 = Input(shape=env.feature_length5)
        x5 = self.Conv2d_BN(inputs5, nb_filter=64, kernel_size=(3, 3), padding='same')
        x5 = self.Conv_Block(x5, nb_filter=64, kernel_size=(3, 3))
        x5 = self.Conv_Block(x5, nb_filter=64, kernel_size=(3, 3))
        x5 = self.Conv_Block(x5, nb_filter=64, kernel_size=(3, 3))
        x5 = Flatten()(x5)
        output5 = Dense(32, activation='relu')(x5)
        model5 = Model(input=inputs5, output=output5)
    
        inputs6 = Input(shape=(1,) + env.feature_length6)
        x6 = Flatten()(inputs6)
        output6 = Dense(32, activation='relu')(x6)
        model6 = Model(input=inputs6, output=output6)
    
        inputs7 = Input(shape=(1,) + env.feature_length7)
        x7 = Flatten()(inputs7)
        output7 = Dense(32, activation='relu')(x7)
        model7 = Model(input=inputs7, output=output7)
    
        combined = concatenate([model1.output, model2.output, model3.output, model4.output, model5.output, model6.output, model7.output])
        x = Dense(128, activation='relu')(combined)
        output = Dense(env.action_space.n, activation='linear')(x)
    
        model = Model(input=[model1.input, model2.input, model3.input, model4.input, model5.input, model6.input, model7.input], output=output)
        print(model.summary())
    
        return model