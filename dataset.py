from glob import glob
import numpy as np
import os

ACTION_TYPES = [
	"boxing",
	"handclapping",
	"handwaving",
	"jogging",
	"running",
	"walking"
]

def one_hot(y_):
    """
    Function to encode output labels from number indexes.

    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    n_values = 6
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def get_data(part):
	actions_x_data = []
	actions_y_data = []
	for i, action in enumerate(ACTION_TYPES):
		files = glob(part+'*_'+action+'_*')
		files.sort()
		list1 = []
		for file in files:
			vec = np.load(file)[:44]
			list1.append(vec)
		save_array = np.array(list1)
		m,n,_,_ = save_array.shape
		y_label = np.zeros([m,1])
		y_label[:,0] = i
		y_label = one_hot(y_label)
		save_array = save_array.reshape((m,n,13*2))
		actions_x_data.append(save_array)
		actions_y_data.append(y_label)
	x_data = np.vstack(actions_x_data)
	y_data = np.vstack(actions_y_data)
	j, m, n = x_data.shape
	x_data = x_data.reshape(j,m*n)
	# shuffle data
	data = np.hstack((x_data, y_data))
	np.random.shuffle(data)
	x = data[:,:-6]
	x = x.reshape(j, m, n)
	y = data[:,-6:]
	return x, y

def get_dataset():
	train_x, train_y = get_data('train/')
	test_x, test_y = get_data('test/')
	return train_x,train_y,test_x,test_y
if __name__ == '__main__':
	train_x,train_y,test_x,test_y = get_dataset()
	print train_x.shape,train_y.shape,test_x.shape,test_y.shape