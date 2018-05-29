import numpy as np
from scipy.io import loadmat

from aux_code.ops import randomly_split_data
from aux_code.ops import one_hot_encoder



def load_data(dataset_name, seq_len=200):
    '''
    Returns:
    x - a n_samples long list containing arrays of shape (sequence_length,
                                                          n_features)
    y - an array of the labels with shape (n_samples, n_classes)
    '''
    print("Loading " + dataset_name + " dataset ...")

    if dataset_name == 'test':
        n_data_points = 5000
        sequence_length = 100
        n_features = 1
        x = list(np.random.rand(n_data_points, sequence_length, n_features))
        n_classes = 4
        y = np.random.randint(low=0, high=n_classes, size=n_data_points)

    if dataset_name == 'mnist':
        return get_mnist(permute=False)

    if dataset_name == 'pmnist':
        return get_mnist(permute=True)

    if dataset_name == 'emg':
        return get_emg()

    if dataset_name == 'add':
        x, y = get_add(n_data=150000, seq_len=seq_len)

    if dataset_name == 'copy':
        return get_copy(n_data=150000, seq_len=seq_len)

    train_idx, valid_idx, test_idx = randomly_split_data(
        y, test_frac=0.2, valid_frac=0.1)

    x_train = [x[i] for i in train_idx]
    y_train = y[train_idx]
    x_valid = [x[i] for i in valid_idx]
    y_valid = y[valid_idx]
    x_test = [x[i] for i in test_idx]
    y_test = y[test_idx]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_emg():
    combined = loadmat('combined.mat')
    combined = np.array((combined['combined']))
    labels = combined[:, 10]
    emg = combined[:,0:10]
    no_of_classes = int(np.max(labels)) + 1
    print("no_of_classes without zero is " + str(no_of_classes-1))
    all_x = list()
    x_train = list()
    x_valid = list()
    x_test = list()
    y_train = np.zeros((80,10))
    y_valid = np.zeros((40,10))
    y_test = np.zeros((30,10))
    for gesture in range(0, no_of_classes-1):
        all_x = list()
        itemIndex = np.where(labels == gesture+1)[0]
        last = 0
        for i in range(0, len(itemIndex)-1):
            if itemIndex[i] + 1 != itemIndex[i+1] or itemIndex[i] + 1 != itemIndex[i+1]:
                begining = itemIndex[i]-(i-last)
                end = itemIndex[i]
                last = i
                all_x.append(emg[begining+1:begining+511])
        all_x.append(emg[itemIndex[last+1]:itemIndex[last+1]+511])
        if len(all_x) == 12*(no_of_classes-1):
            print("length of the list is equal to 12*no_of_classes, which is correct")


        x_train.extend(all_x[:8])
        print(len(x_train))
        #  y_train = np.append(y_train, np.full((8, 1), gesture))
        #  y_train_hot = one_hot_encoder(y_train.astype(int)[:,None], no_of_classes-1)
        y_train_hot = one_hot_encoder(np.ones(8)*gesture, n_classes=10)
        y_train[gesture*8:(gesture+1)*8,:] = y_train_hot

        x_valid.extend(all_x[8:12])
        #  y_valid = np.append(y_valid, np.full((1, 1), gesture))
        #  y_valid_hot = one_hot_encoder(y_valid.astype(int)[:,None], no_of_classes-1)
        y_valid_hot = one_hot_encoder(np.ones(4)*gesture, n_classes=10)
        y_valid[gesture*4:(gesture+1)*4,:] = y_valid_hot

        x_test.extend(all_x[9::])
        #  y_test = np.append(y_test, np.full((3, 1), gesture))
        #  y_test_hot = one_hot_encoder(y_test.astype(int)[:,None], no_of_classes-1)
        y_test_hot = one_hot_encoder(np.ones(3)*gesture, n_classes=10)
        y_test[gesture*3:(gesture+1)*3,:] = y_test_hot

        #print((np.asarray(x_train)).shape, y_train_hot.shape)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_add(n_data, seq_len):
    x = np.zeros((n_data, seq_len, 2))
    x[:,:,0] = np.random.uniform(low=0., high=1., size=(n_data, seq_len))
    inds = np.random.randint(seq_len/2, size=(n_data, 2))
    inds[:,1] += seq_len//2
    for i in range(n_data):
        x[i,inds[i,0],1] = 1.0
        x[i,inds[i,1],1] = 1.0

    y = (x[:,:,0] * x[:,:,1]).sum(axis=1)
    y = np.reshape(y, (n_data, 1))
    return x, y


def get_copy(n_data, seq_len):
    x = np.zeros((n_data, seq_len+1+2*10))
    info = np.random.randint(1, high=9, size=(n_data, 10))

    x[:,:10] = info
    x[:,seq_len+10] = 9*np.ones(n_data)

    y = np.zeros_like(x)
    y[:,-10:] = info

    x = one_hot_sequence(x)
    y = one_hot_sequence(y)

    n_train, n_valid, n_test = [100000, 10000, 40000]
    x_train = list(x[:n_train])
    y_train = y[:n_train]
    x_valid = list(x[n_train:n_train+n_valid])
    y_valid = y[n_train:n_train+n_valid]
    x_test = list(x[-n_test:])
    y_test = y[-n_test:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_mnist(permute=False):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST", one_hot=True)

    if permute:
        perm_mask = np.load('misc/pmnist_permutation_mask.npy')
    else:
        perm_mask = np.arange(784)

    x_train = list(np.expand_dims(mnist.train.images[:,perm_mask],-1))
    y_train = mnist.train.labels
    x_valid = list(np.expand_dims(mnist.validation.images[:,perm_mask],-1))
    y_valid = mnist.validation.labels
    x_test = list(np.expand_dims(mnist.test.images[:,perm_mask], -1))
    y_test = mnist.test.labels

    print("Train:Validation:Testing - %d:%d:%d" % (len(y_train), len(y_valid),
                                                   len(y_test)))

    return x_train, y_train, x_valid, y_valid, x_test, y_test
