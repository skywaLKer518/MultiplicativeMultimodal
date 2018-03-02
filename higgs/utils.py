import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import pandas as pd


class TextLoader():
    def __init__(self, data_dir, batch_size, encoding='utf-8', test='', 
        test2 = '', timestamp='1127', overlap=1):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.encoding = encoding
        self.overlap = overlap
        # only debug use; comment please
        # test2 = test

        input_file_name = os.path.join(data_dir, 'HIGGS2.csv')

        tensor_file = os.path.join(data_dir, 'HIGGS' + timestamp + test + '.pkl')

        if not os.path.exists(tensor_file):
            print("reading input files; \ndata normalize; save preprocessed files")
            self.preprocess(input_file_name, tensor_file, test)
        else:
            print("loading preprocessed files {}".format(tensor_file))
            self.load_preprocessed(tensor_file)
        
        # # self.create_batches()
        self.reset_batch_pointer()
        return 

    def preprocess(self, input_file_name, saved_file_name, test):
        f = input_file_name
        print('loading file {}'.format(f))
        data0 = pd.read_csv(f, delimiter='\t', header=None).values
        n = len(data0)
        n_te = 500000
        n_tr = n - n_te
        print('number of total training instances : {}'.format(n_tr))
        
        data_te = data0[-n_te:, :]
        print('number of test instances : {}'.format(len(data_te)))

        if test == '':
            sample = 0
        elif test.startswith('.sample'):
            sample = int(test.split('.sample')[1])
        else:
            print('error! test argument has to start with .sample but is {}'.fomat(test))

        x_mean = data0[:, 1:22].mean(0)
        x_std = data0[:, 1:22].std(0)
        z_mean = data0[:, 22:29].mean(0)
        z_std = data0[:, 22:29].std(0)

        print('std ')
        print(x_std)
        print(z_std)

        if sample == 0:
            inds = range(n_tr)
        else:
            inds = range(0, n_tr, sample)
        data_tr = data0[inds, :]
        n_tr = len(data_tr)

        print('number of sampled training instances : {}'.format(len(data_tr)))


        self.y_tr, self.y_te = data_tr[:, 0], data_te[:, 0]
        self.y_tr = np.concatenate([self.y_tr.reshape(n_tr, 1), 
            (1.0 - self.y_tr).reshape(n_tr, 1)], 1)
        self.y_te = np.concatenate([self.y_te.reshape(n_te, 1), 
            (1.0 - self.y_te).reshape(n_te, 1)], 1)
        
        self.x_tr, self.x_te = data_tr[:, 1:22], data_te[:, 1:22]
        
        self.x_tr = (self.x_tr - x_mean) / x_std
        self.x_te = (self.x_te - x_mean) / x_std

        self.z_tr, self.z_te = data_tr[:, 22:29], data_te[:, 22:29]

        self.z_tr = (self.z_tr - z_mean) / z_std
        self.z_te = (self.z_te - z_mean) / z_std

        self.v_tr, self.v_te = None, None

        self.num_batches = int(1.0 * len(self.y_tr) / self.batch_size)
        self.num_batches_te = int(1.0 * len(self.y_te) / self.batch_size)

        data = (self.x_tr, self.y_tr, self.z_tr, self.v_tr, self.x_te, 
            self.y_te, self.z_te, self.v_te)
        with open(saved_file_name, 'wb') as f:
            cPickle.dump(data, f)
        return

    def load_preprocessed(self, tensor_file):
        with open(tensor_file, 'rb') as f:
            data = cPickle.load(f)
        (self.x_tr, self.y_tr, self.z_tr, self.v_tr, 
            self.x_te, self.y_te, self.z_te, self.v_te) = data
        print('low level feature means')
        print(self.x_te.mean(0))
        self.num_batches = int(1.0 * len(self.y_tr) / self.batch_size)
        self.num_batches_te = int(1.0 * len(self.y_te) / self.batch_size)
        return 

    def next_batch(self, eval=False, split='train'):
        if split == 'train':
            x, y, z, v = self.x_tr, self.y_tr, self.z_tr, self.v_tr
            l = len(self.y_tr)
        else:
            x, y, z, v = self.x_te, self.y_te, self.z_te, self.v_te
            l = len(self.y_te)
        # if self.tile > 0:
        #     img_ids = self.img_ids if split == 'train' else self.img_ids_te
        # if self.use_uid != 0:
        #     user_tokens = self.user_tokens_tr if split == 'train' else self.user_tokens_te

        if eval:
            inds = range(self.pointer, self.pointer + self.batch_size)
            self.pointer += self.batch_size
        else:
            self.pointer = np.random.choice(l - self.batch_size)
            inds = range(self.pointer, self.pointer + self.batch_size)
        y_batch = [y[i] for i in inds]
        x_batch = [x[i] for i in inds]
        z_batch = [z[i] for i in inds]
        v_batch = []
        # v_batch = [v[i] for i in inds]

        return x_batch, y_batch, z_batch, v_batch

    def reset_batch_pointer(self):
        self.pointer = 0

