import pickle
import numpy as np
from pathlib import Path

class data:

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            _dict = pickle.load(fo, encoding='latin1')
        return _dict
    
    def load_cifar_labels(self):
        file_name = self.cifar_dataset + "/batches.meta"
        labels = self.unpickle(file_name)
        self.labels = labels["label_names"]
        self.num_train_elements = labels["num_cases_per_batch"]
    
    def load_cifar_test(self):
        file_name = self.cifar_dataset + "/test_batch"
        test = self.unpickle(file_name)
        self.test_data = test["data"]
        self.test_data = self.test_data.reshape((len(self.test_data), 3, 32, 32)).transpose(0, 2, 3, 1)
        self.test_data = np.reshape(self.test_data, (self.test_data.shape[0], -1))
        self.test_labels = test["labels"]
    
    def load_cifar_train(self, ind):
        # load training data
        train_data_names = ["/data_batch_1", "/data_batch_2", "/data_batch_3", "/data_batch_4", "/data_batch_5"]
        file_name = self.cifar_dataset + train_data_names[ind]
        train = self.unpickle(file_name)
        self.train_data = train["data"]
        self.train_data = self.train_data.reshape((len(self.train_data), 3, 32, 32)).transpose(0, 2, 3, 1)
        self.train_data = np.reshape(self.train_data, (self.train_data.shape[0], -1))
        self.train_labels = train["labels"]

    def __init__(self):
        self.root = str(Path(__file__).parent.parent)
        self.dataset_directory = self.root + "/datasets"
        self.cifar_dataset = self.dataset_directory + "/cifar"
