import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class onevsrest:

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self):
        for classifier_ind in range(len(self.classifiers)):
            # print(self.X)
            self.classifiers[classifier_ind].fit(self.X, self.Y[classifier_ind])
            # print(self.Y[classifier_ind])
            self.train_accuracy.append(self.classifiers[classifier_ind].score(self.X, self.Y[classifier_ind]))

    def score(self, X, Y):
        pass
        # X, Y = self.split_data(X, Y)
        fin_preds = []
        for sample_ind in range(len(X)):
            sample_X = X[sample_ind].reshape(1, -1)
            preds = []
            scores = []
            for classifier_ind in range(len(self.classifiers)):
                if classifier_ind == 0:
                    score = (self.classifiers[classifier_ind].score(sample_X, [1]))
                    if score == 1:
                        preds.append(0)
                    else:
                        preds.extend([1, 2])
                if classifier_ind == 1:
                    score = (self.classifiers[classifier_ind].score(sample_X, [1]))
                    if score == 1:
                        preds.append(1)
                    else:
                        preds.extend([0, 2])
                if classifier_ind == 2:
                    score = (self.classifiers[classifier_ind].score(sample_X, [1]))
                    if score == 1:
                        preds.append(2)
                    else:
                        preds.extend([1, 0])

            fin_preds.append(max(set(preds), key=preds.count))
        return accuracy_score(fin_preds, Y.ravel()), fin_preds

    def __init__(self, num_classes, traindata, trainlabels):
        self.num_classes = num_classes
        self.classifiers = [svm.LinearSVC(random_state=0, tol=1e-5, C=1.0, max_iter=2000) for i in range(num_classes)] 
        self.train_accuracy = []
        # 3 -> 0, 1, 2

        self.X = traindata
        y = trainlabels.ravel()
        # get traindata samples with labels as 1, 2, and 3
        self.Y = label_binarize(y, classes=[0, 1, 2]).T

        # self.Y = np.array(self.Y)
        # self.X = np.array(self.X)
        # for i in range(num_classes):
        #     self.X[i] = np.array(self.X[i])
        #     self.Y[i] = np.array(self.Y[i])
        # for i in self.Y:
        #     print(i)
