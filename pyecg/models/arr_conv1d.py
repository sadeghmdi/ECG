import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *





def print_layer(layer):
	'''Prints layer output dim and its name or class name'''
	l_out_shape = tf.keras.backend.shape(layer)._inferred_value
	l_name = layer.name
	#l_in_shape = layer.input_shape
	#l_out_shape = layer.output_shape
	#print('\nLayer: {} --> Input shape: {}, Output shape: {}'.
	#		format(str(l_name), str(l_in_shape) , str(l_out_shape))) 
	print('\nLayer: {} -->  Output shape: {}'.
			format(str(l_name).upper(), str(l_out_shape))) 


def reg():
	return tf.keras.regularizers.l2(l=0.01)


def conv1d_block(inp, name=None, filters=64, kernel_size=64, strides=1, 
							bn=True, drate=0.30, pool_size=0,flatten=True,regularizer=None):

	#print('{}:'.format(name))
	output = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, 
									padding='valid', activation=None, 
									kernel_regularizer = regularizer)(inp)
	if bn:
		output = BatchNormalization(axis=-1)(output)

	output = relu(output)

	if drate>0:
		output = Dropout(drate)(output)
	if pool_size>0:
		output = MaxPool1D(pool_size=pool_size)(output)
	if flatten:
		output = Flatten()(output)
	
	#print(tf.keras.backend.shape(output)._inferred_value)
	return output 


def model_arch(params_model):

	
	return tf.keras.Model(inputs=input_layer, outputs=out, name='Model_Conv1d_ARR')






