import os
import time
import pickle
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# get_ipython().run_line_magic('ma  tplotlib', 'inline')
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from scripts import get_data

def savePickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def checkandcreatedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def storemodel(model, name):
    root = str(Path(__file__).parent.parent)
    modeldir = root + "/models"
    checkandcreatedir(modeldir)
    filepath = modeldir + "/" + name
    savePickle(filepath, model)

def loadmodel(filename):
    root = str(Path(__file__).parent.parent)
    modeldir = root + "/models"
    filename = modeldir + "/" + filename
    try:
        model = loadPickle(filename)
        return model
    except:
        raise Exception("Model not found: " + filename )

def svc_param_selection(X, Y, numFolds):
    Cs = [0.001, 0.01, 0.1]
    param_grid = {'C': Cs}
    grid_search = GridSearchCV(svm.LinearSVC(random_state=0, tol=1e-5), param_grid, cv=numFolds, verbose=10, n_jobs=1)
    grid_search.fit(X, Y)
    storemodel(grid_search, "gridsearch")
    # print(grid_search.best_params_)
    return grid_search.best_params_

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--preprocess", required=False, help="Just preprocess the train and test data.", action="store_true")
    ap.add_argument("-lt", "--loadtraintestdata", required=False, help="Use cached/stored train test data.", action="store_true")
    ap.add_argument("-lg", "--loadgridsearch", required=False, help="Use cached/stored grid_search data.", action="store_true")
    ap.add_argument("-lm", "--loadtrainmodel", required=False, help="Use cached/stored model.", action="store_true")
    ap.add_argument("-lmn", "--loadtrainmodelnew", required=False, help="Use cached/stored updated model (trained only on support vecs)).", action="store_true")
    ap.add_argument("-n", "--num-folds", required=False, default=3, help="Number of folds to find the best param.")
    args = vars(ap.parse_args())

    preprocess = args["preprocess"]
    loadtraintestdata = args["loadtraintestdata"]
    numFolds = args["num_folds"]
    loadgridsearch = args["loadgridsearch"]
    loadtrainmodel = args["loadtrainmodel"]
    loadnewtrainmodel = args["loadtrainmodelnew"]

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
    if loadgridsearch:
        gridsearch = loadmodel("gridsearch")
    else:
        best_params = svc_param_selection(data.train_data, data.train_labels, numFolds)
    print("Time taken: %s seconds ---" % (time.time() - start_time))
    print("Best params:", gridsearch.best_params_)
    best_params = gridsearch.best_params_

    #train
    print("Training model on best params")
    start_time = time.time()
    if loadtrainmodel:
        linsvm_model = loadmodel("trained_model")
    else:
        linsvm_model = svm.SVC(kernel='linear', random_state=0, tol=1e-5, C=best_params["C"], verbose=10)
        linsvm_model.fit(data.train_data, data.train_labels)
        storemodel(linsvm_model, "trained_model")
    print("Time taken: %s seconds ---" % (time.time() - start_time))
    # print("Training accuracy:", linsvm_model.score(data.train_data, data.train_labels))
    # Test
    # print("Testing accuracy:", linsvm_model.score(data.test_data, data.test_labels))
    
    # Find support vectors
    supportvecindices = linsvm_model.support_
    new_training_set = []
    new_training_labels = []
    for supportvecindex in supportvecindices:
        new_training_set.append(data.train_data[supportvecindex])
        new_training_labels.append(data.train_labels[supportvecindex])
    new_training_set = np.array(new_training_set)
    new_training_labels = np.array(new_training_labels)
    
    print("Training model on best params and on support vectors")
    start_time = time.time()
    if loadnewtrainmodel:
        linsvm_model_new = loadmodel("trained_model_new")
    else:
        linsvm_model_new = svm.SVC(kernel='linear', random_state=0, tol=1e-5, C=best_params["C"], verbose=10)
        linsvm_model_new.fit(new_training_set, new_training_labels)
        storemodel(linsvm_model_new, "trained_model_new")
    print("Time taken: %s seconds ---" % (time.time() - start_time))
    print("Training accuracy:", linsvm_model_new.score(new_training_set, new_training_labels))
    # Test
    print("Testing accuracy:", linsvm_model_new.score(data.test_data, data.test_labels))
