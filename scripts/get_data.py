import pickle
import numpy as np
from pathlib import Path
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

class data:

    def savePickle(self, filename, data):
        with open(filename, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    def loadPickle(self, filename):
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)
        return b

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            _dict = pickle.load(fo, encoding='latin1')
        return _dict
    
    def load_cifar_labels(self):
        labelfile = self.cifar_dataset + "/labelingdata_processed"
        # load labeling data
        if self.load:
            label_data = self.loadPickle(labelfile)
            self.labels = label_data["names"]
            self.num_train_elements = label_data["num"]
            return

        file_name = self.cifar_dataset + "/batches.meta"
        labels = self.unpickle(file_name)
        self.labels = labels["label_names"]
        self.num_train_elements = labels["num_cases_per_batch"]

        label_data = {}
        label_data["names"] = self.labels
        label_data["num"] = self.num_train_elements
        self.savePickle(labelfile, label_data)
    
    def load_cifar_test(self):
        testfile = self.cifar_dataset + "/testingdata_processed"
        # load testing data
        if self.load:
            test_data = self.loadPickle(testfile)
            self.test_data = test_data["data"]
            self.test_labels = test_data["labels"]
            return

        file_name = self.cifar_dataset + "/test_batch"
        test = self.unpickle(file_name)
        test_data = test["data"]
        test_data = test_data.reshape((len(test_data), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_data = test_data.astype('float32')
        test_data = test_data / 255.0
        self.test_data = []
        ## convert to grayscale
        for im_ind in range(len(test_data)):
            self.test_data.append(rgb2gray(test_data[im_ind]))
        self.test_data = np.array(self.test_data)

        self.test_data = np.reshape(self.test_data, (self.test_data.shape[0], -1))
        self.test_labels = test["labels"]

        test_data = {}
        test_data["data"] = self.test_data
        test_data["labels"] = self.test_labels
        self.savePickle(testfile, test_data)
    
    def load_cifar_train(self, ind):
        trainfile = self.cifar_dataset + "/trainingdata_processed"
        # load training data
        if self.load:
            train_data = self.loadPickle(trainfile)
            self.train_data = train_data["data"]
            self.train_labels = train_data["labels"]
            return

        train_data_names = ["/data_batch_1", "/data_batch_2", "/data_batch_3", "/data_batch_4", "/data_batch_5"]
        self.train_data = []
        self.train_labels = []
        for name in train_data_names:
            file_name = self.cifar_dataset + name
            train = self.unpickle(file_name)
            train_data = train["data"]
            train_data = train_data.reshape((len(train_data), 3, 32, 32)).transpose(0, 2, 3, 1)
            train_data = train_data.astype('float32')
            train_data = train_data / 255.0
            ## convert to grayscale
            for im_ind in range(len(train_data)):
                self.train_data.append(rgb2gray(train_data[im_ind]))
            self.train_labels.extend(train["labels"])
        
        self.train_data = np.array(self.train_data)
        self.train_data = np.reshape(self.train_data, (self.train_data.shape[0], -1))
        
        train_data = {}
        train_data["data"] = self.train_data
        train_data["labels"] = self.train_labels
        self.savePickle(trainfile, train_data)

    def __init__(self, load):
        self.root = str(Path(__file__).parent.parent)
        self.dataset_directory = self.root + "/datasets"
        self.cifar_dataset = self.dataset_directory + "/cifar"
        self.load = load
