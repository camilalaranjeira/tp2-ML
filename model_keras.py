from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras import regularizers
#from theano import tensor as T
#import theano
import cv2, numpy as np
import pickle
import sys


def rot90(W):
     for i in range(W.shape[0]):
         for j in range(W.shape[1]):
             W[i, j] = np.rot90(W[i, j], 2)
     return W

def get_caffe_weights():
	caffe_root = '/home/laranjeira/caffe-segnet-cudnn5/'
	sys.path.insert(0, caffe_root + 'python')
	import caffe

	root_path = '/home/laranjeira/projects/Segnet/'
	model	   = root_path + 'models/segnet_sun.prototxt'
	weights    = root_path + 'models/segnet_sun.caffemodel'

	caffe.set_mode_cpu()
	net  = caffe.Net(model, weights, caffe.TEST)

	w = {}
	b  = {}

	keys = net.params.keys()
	for n, k in enumerate(keys):

		#print k, len(net.params[k]), np.asarray(net.params[k][0].data[...]).shape, np.asarray(net.params[k][1].data[...]).shape
		w[k] = net.params[k][0].data[...]
		b[k]  = net.params[k][1].data[...]

	return w, b

def VGG_16(weights_path=None):
	model = Sequential()

	#conv1_1
	model.add(ZeroPadding2D((1,1),input_shape=(360,480,3), batch_size=1, data_format="channels_last", name='pad_conv1_1'))
	model.add(Convolution2D(64, 3, name='conv1_1'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv1_1_bn'))
	model.add(Activation('relu'))
	
	#conv1_2
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, name='conv1_2'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv1_2_bn'))
	model.add(Activation('relu'))

	#pool1
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	#conv2_1
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, name='conv2_1'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv2_1_bn'))
	model.add(Activation('relu'))	
	
	#conv2_2
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, name='conv2_2'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv2_2_bn'))
	model.add(Activation('relu'))

	#pool2
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	#conv3_1
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, name='conv3_1'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv3_1_bn'))
	model.add(Activation('relu'))
    
	#conv3_2
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, name='conv3_2'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv3_2_bn'))
	model.add(Activation('relu'))
    
	#conv3_3
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, name='conv3_3'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv3_3_bn'))
	model.add(Activation('relu'))

	#pool3
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	#encdrop3
	model.add(Dropout(0.5))
	
	#conv4_1
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, name='conv4_1'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv4_1_bn'))
	model.add(Activation('relu'))

	#conv4_2
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, name='conv4_2'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv4_2_bn'))
	model.add(Activation('relu'))    
	
	#conv4_3
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, name='conv4_3'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv4_3_bn'))
	model.add(Activation('relu'))

	model.add(ZeroPadding2D((1,0)))
	#pool4
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	#encdrop4
	model.add(Dropout(0.5))

	#conv5_1
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, name='conv5_1'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv5_1_bn'))
	model.add(Activation('relu'))

	#conv5_2
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, name='conv5_2'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv5_2_bn'))
	model.add(Activation('relu'))
    
	#conv5_3
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, name='conv5_3'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv5_3_bn'))
	model.add(Activation('relu'))

	model.add(ZeroPadding2D((1,0)))
	#pool5
	model.add(MaxPooling2D((2,2), strides=(2,2), name='pool5'))

	#encdrop5
	model.add(Dropout(0.5))

	if weights_path:
		model.load_weights(weights_path)

	return model

def Segnet(weights_path=None):
	model = VGG_16()

	#upsample5
	model.add(UpSampling2D(size=(2, 2)))
	
	#conv5_3_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, name='conv5_3_D'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv5_3_D_bn'))
	model.add(Activation('relu'))

	#conv5_2_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, name='conv5_2_D'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv5_2_D_bn'))
	model.add(Activation('relu'))

	#conv5_1_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, name='conv5_1_D'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv5_1_D_bn'))
	model.add(Activation('relu'))

	#decdrop5
	model.add(Dropout(0.5))
	
	#upsample4
	model.add(UpSampling2D(size=(2, 2)))

	#conv4_3_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, name='conv4_3_D'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv4_3_D_bn'))
	model.add(Activation('relu'))

	#conv4_2_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, name='conv4_2_D'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv4_2_D_bn'))
	model.add(Activation('relu'))

	#conv4_1_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, name='conv4_1_D'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv4_1_D_bn'))
	model.add(Activation('relu'))

	#decdrop4
	model.add(Dropout(0.5))
	
	#upsample3
	model.add(UpSampling2D(size=(2, 2)))

	#conv3_3_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, name='conv3_3_D'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv3_3_D_bn'))
	model.add(Activation('relu'))

	#conv3_2_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, name='conv3_2_D'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv3_2_D_bn'))
	model.add(Activation('relu'))

	#conv3_1_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, name='conv3_1_D'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv3_1_D_bn'))
	model.add(Activation('relu'))

	#decdrop3
	model.add(Dropout(0.5))

	#upsample2
	model.add(UpSampling2D(size=(2, 2)))

	#conv2_2_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, name='conv2_2_D'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv2_2_D_bn'))
	model.add(Activation('relu'))

	#conv2_1_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, name='conv2_1_D'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv2_1_D_bn'))
	model.add(Activation('relu'))

	#upsample1
	model.add(UpSampling2D(size=(2, 2)))

	#conv1_2_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, name='conv1_2_D'))
	model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', center=False, scale=False, name='conv1_2_D_bn'))
	model.add(Activation('relu'))

	#conv1_1_D
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(38, 3, name='conv1_1_D'))

	if weights_path:
#		model.load_weights_from_hdf5_group(weights_path['model_weights'])
		model.load_weights(weights_path)
	
	return model

def set_weights(model):

	with open('/home/laranjeira/projects/Segnet/weights/weights.pkl') as f:
		weights = pickle.load(f)
	with open('/home/laranjeira/projects/Segnet/weights/biases.pkl') as f:
		biases = pickle.load(f)

	for layer in model.layers:
		a = layer.get_weights()
		if len(a) > 0:
			if layer.name[-2:] == 'bn':
				w = weights[layer.name].flatten()
				b = biases[layer.name].flatten()				
			else:
				w = weights[layer.name].T
				b = biases[layer.name]

			#print layer.name, np.asarray(layer.get_weights()[0]).shape, np.asarray(layer.get_weights()[1]).shape, w.shape, b.shape
			new_weights = [w, b]
			layer.set_weights(new_weights)

	return model


def add_dense_layers(model):
	#FC6
	model.add(Flatten())
		
	model.add(Dropout(0.5))

	#FC7
	model.add(Dense(397, activation='softmax', name='fc7'))

	return model
