import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def checkandcreatedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def savePickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--plotdata", required=False, help="Plot the wine dataset.", action="store_true")
    ap.add_argument("-nb", "--naivebayes", required=False, help="Do the Naive-Bayes classification.", action="store_true")
    ap.add_argument("-lnb", "--loadnaivebayes", required=False, help="Load the Naive-Bayes classification cached model.", action="store_true")
    ap.add_argument("-dt", "--dectree", required=False, help="Do the descision tree classification.", action="store_true")
    ap.add_argument("-ldt", "--loaddectree", required=False, help="Load the descision tree classification cached model.", action="store_true")
    
    args = vars(ap.parse_args())
    
    plotdata = args["plotdata"]
    nb = args["naivebayes"]
    lnb = args["loadnaivebayes"]
    dt = args["dectree"]
    ldt = args["loaddectree"]

    raw_data = load_wine()

    wine_data = raw_data["data"]
    wine_index = [i+1 for i in range(len(wine_data))]
    wine_columns = raw_data["feature_names"]

    min_max_scaler = preprocessing.MinMaxScaler()

    X = pd.DataFrame(data=wine_data, index=wine_index, columns=wine_columns)
    # normalizing X
    X = pd.DataFrame(data=min_max_scaler.fit_transform(X), columns=X.columns, index=X.index)
    Y = pd.DataFrame(data=raw_data["target"], index=wine_index, columns=["Target"])

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    # convert to np arrays
    train_X = train_X.values
    train_Y = train_Y.values
    test_X = test_X.values
    test_Y = test_Y.values

    if plotdata:
        print("Plotting")
        # print(X.columns)
        g = sns.pairplot(X, palette="husl")
        plt.show()
    
    if nb or lnb:
        start_time = time.time()
        if lnb:
            print("Loading nb model")
            gauss_nb_model = loadmodel("naivebayes")
        else:
            print("Fitting nb model")
            gauss_nb_model = GaussianNB()
            gauss_nb_model.fit(train_X, train_Y.ravel())
            storemodel(gauss_nb_model, "naivebayes")
        print("Time taken: %s seconds ---" % (time.time() - start_time))
        print()
        # Accuracy
        print("Training accuracy:", gauss_nb_model.score(train_X, train_Y))
        print("Testing accuracy:", gauss_nb_model.score(test_X, test_Y))
        # F1 Score
        train_Y_pred = gauss_nb_model.predict(train_X)
        test_Y_pred = gauss_nb_model.predict(test_X)
        # Calculate metrics globally by counting the total true positives, false negatives and false positives.
        print("Training F1-Score", f1_score(train_Y, train_Y_pred, average='micro') )
        print("Testing F1-Score", f1_score(test_Y, test_Y_pred, average='micro') )
    
    if dt or ldt:
        start_time = time.time()
        if ldt:
            print("Loading dt model")
            dectree_model = loadmodel("dectree")
        else:
            print("Fitting dt model")
            dectree_model = tree.DecisionTreeClassifier()
            dectree_model.fit(train_X, train_Y.ravel())
            storemodel(dectree_model, "dectree")
        print("Time taken: %s seconds ---" % (time.time() - start_time))
        # Accuracy
        print("Training accuracy:", dectree_model.score(train_X, train_Y))
        print("Testing accuracy:", dectree_model.score(test_X, test_Y))
        # F1 Score
        train_Y_pred = dectree_model.predict(train_X)
        test_Y_pred = dectree_model.predict(test_X)
        # Calculate metrics globally by counting the total true positives, false negatives and false positives.
        print("Training F1-Score", f1_score(train_Y, train_Y_pred, average='micro') )
        print("Testing F1-Score", f1_score(test_Y, test_Y_pred, average='micro') )
