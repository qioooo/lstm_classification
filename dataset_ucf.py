from glob import glob
import numpy as np
import os

# ACTION_TYPES = [
# 	"boxing",
# 	"handclapping",
# 	"handwaving",
# 	"jogging",
# 	"running",
# 	"walking"
# ]
ratio = 0.6
def get_action_types():
	files = os.listdir('vectors_ucfsport/')
	list_files = []
	for file in files:
		list_files.append(file.split('_')[0]) 
	actions = list(set(list_files))
	actions.sort()
	return actions
def one_hot(y_):
    """
    Function to encode output labels from number indexes.

    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    n_values = 12
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def get_data(part):
	global ratio
	actions_x_data = []
	actions_y_data = []
	ACTION_TYPES = get_action_types()
	for i, action in enumerate(ACTION_TYPES):
		files = glob('vectors_ucfsport/'+action+'_*')
		files.sort()
		if part == 'train/':
			files = files[:int(len(files)*ratio)]
		if part == 'test/':
			files = files[int(len(files)*ratio):]
		list1 = []
		print len(files)
		for file in files:
			vec = np.load(file)[:22]
			list1.append(vec)
		save_array = np.array(list1)
		m,n,_,_ = save_array.shape
		print save_array.shape
		y_label = np.zeros([m,1])
		y_label[:,0] = i
		y_label = one_hot(y_label)
		save_array = save_array.reshape((m,n,13*2))
		actions_x_data.append(save_array)
		actions_y_data.append(y_label)
	x_data = np.vstack(actions_x_data)
	y_data = np.vstack(actions_y_data)
	print x_data.shape
	print y_data.shape
	j, m, n = x_data.shape
	x_data = x_data.reshape(j,m*n)
	# shuffle data
	data = np.hstack((x_data, y_data))
	np.random.shuffle(data)
	x = data[:,:-12]
	x = x.reshape(j, m, n)
	y = data[:,-12:]
	return x, y

def get_dataset():
	train_x, train_y = get_data('train/')
	test_x, test_y = get_data('test/')
	return train_x,train_y,test_x,test_y
if __name__ == '__main__':
	train_x,train_y,test_x,test_y = get_dataset()
	print train_x.shape,train_y.shape,test_x.shape,test_y.shape