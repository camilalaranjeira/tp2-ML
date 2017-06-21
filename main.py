import numpy as np
from scipy import misc
import time
import model_keras
import os
#import theano

def freeze_weights(model):
	# Freeze conv weights. Train only dense layers
	for layer in model.layers:
		if not layer.name[:2] == 'fc':
			layer.trainable = False

	return model

'''
def kullback_leibler(y_pred,y_true):
    eps = 1e-5
    results, updates = theano.scan(lambda y_true,y_pred: (y_true+eps)*(T.log(y_true+eps)-T.log(y_pred+eps)), sequences = [y_true,y_pred])
    return(T.sum(results, axis= - 1))

def get_output_layer(input_, model, layer_name):
	layer = model.get_layer(layer_name)
	get_output = theano.function([model.input], layer.output) 

	output = get_output(input_)
	print output.shape
'''

if __name__=='__main__':
	
	model = model_keras.VGG_16()
	model = model_keras.set_weights(model)
	#model = model_keras.add_dense_layers(model)

	#model = freeze_weights(model)
	all_feats = []
	dataset_path = '/home/laranjeira/datasets/Scene15/'
	for class_ in os.listdir(dataset_path):
		class_path = os.path.join(dataset_path, class_)
		for sample in os.listdir(class_path):

			img_path = os.path.join(class_path, sample)
			if not os.path.isfile(img_path):
				continue
			img = misc.imread(img_path, mode='RGB')
			img = misc.imresize(img, (360, 480, 3))
			img = np.expand_dims(img, axis=0)

			#feat = get_output_layer(img, model, 'pool5')
			feat = model.predict(img)
			feat = np.transpose(feat, (0,3,1,2))
			feat = feat.flatten()
			print img_path, feat.shape

			all_feats.append(feat)

	save_path = '/home/laranjeira/projects/Segnet/features/scene15/all_feats.npy'
	with open(save_path, 'w') as f:
		np.savez_compressed(f, all_feats)


#	print model.summary()	
	#get optimization parameters from solve file
	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(optimizer = sgd)