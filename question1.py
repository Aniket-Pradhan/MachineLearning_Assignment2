import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# get_ipython().run_line_magic('ma  tplotlib', 'inline')
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from scripts import get_data

def savePickle(self, filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(self, filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def svc_param_selection(X, Y, numFolds):
    Cs = [0.001, 0.01, 0.1]
    param_grid = {'C': Cs}
    grid_search = GridSearchCV(LinearSVC(random_state=0, tol=1e-5), param_grid, cv=nfolds, verbose=10)
    grid_search.fit(X, Y)
    # print(grid_search.best_params_)
    return grid_search.best_params_

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--preprocess", required=False, help="Just preprocess the train and test data.", action="store_true")
    ap.add_argument("-lt", "--loadtraintestdata", required=False, help="Use cached/stored train test data.", action="store_true")
    ap.add_argument("-n", "--num-folds", required=False, default=3, help="Number of folds to find the best param.")
    args = vars(ap.parse_args())

    preprocess = args["preprocess"]
    loadtraintestdata = args["loadtraintestdata"]
    numFolds = args["numFolds"]

    data = get_data.data(loadtraintestdata)
    data.load_cifar_labels()
    data.load_cifar_train(0)
    data.load_cifar_test()

    if preprocess:
        print("preprocessed and stored the dataset")
        exit(0)
    if loadtraintestdata:
        print("loaded train_test data")

    print("Finding best params for the linear SVM")
    start_time = time.time()
    best_params = svc_param_selection(data.train_data, data.train_labels, numFolds)
    print("Time taken: %s seconds ---" % (time.time() - start_time))
