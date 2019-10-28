import statistics
import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class onevsone:

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self):
        for classifier_ind in range(len(self.classifiers)):
            self.classifiers[classifier_ind].fit(self.X[classifier_ind], self.Y[classifier_ind])
            # print(self.Y[classifier_ind])
            self.train_accuracy.append(self.classifiers[classifier_ind].score(self.X[classifier_ind], self.Y[classifier_ind]))
    
    def split_data(self, X, Y):
        Y_temp = []
        X_temp = []
        for i in range(self.num_classes):
            Y_temp.append([])
            X_temp.append([])

        # 01 02 12
        for i in range(len(Y)):
            if Y[i] == 0:
                Y_temp[0].append(0)
                Y_temp[1].append(0)
                X_temp[0].append(X[i])
                X_temp[1].append(X[i])
            elif Y[i] == 1:
                Y_temp[0].append(1)
                Y_temp[2].append(1)
                X_temp[0].append(X[i])
                X_temp[2].append(X[i])
            else:
                Y_temp[1].append(2)
                Y_temp[2].append(2)
                X_temp[1].append(X[i])
                X_temp[2].append(X[i])
        return X_temp, Y_temp


    def score(self, X, Y):
        # X, Y = self.split_data(X, Y)
        fin_preds = []
        for sample_ind in range(len(X)):
            sample_X = X[sample_ind].reshape(1, -1)
            preds = []
            for classifier_ind in range(len(self.classifiers)):
                preds.append(self.classifiers[classifier_ind].predict(sample_X)[0])
            fin_preds.append(max(set(preds), key=preds.count))
        return accuracy_score(fin_preds, Y.ravel()), fin_preds

    def __init__(self, num_classes, traindata, trainlabels):
        self.num_classes = num_classes
        self.classifiers = [svm.LinearSVC(random_state=0, tol=1e-5, C=1.0, max_iter=2000) for i in range(num_classes)] 
        self.train_accuracy = []
        # 3 -> 01, 02, 12

        X = traindata
        y = trainlabels.ravel()
        # y = label_binarize(y, classes=[0, 1, 2])

        # get traindata samples with labels as 12, 13, and 23
        self.X, self.Y = self.split_data(X, y)

        self.Y = np.array(self.Y)
        self.X = np.array(self.X)
        for i in range(num_classes):
            self.X[i] = np.array(self.X[i])
            self.Y[i] = np.array(self.Y[i])
        # for i in self.Y:
        #     print(i)
