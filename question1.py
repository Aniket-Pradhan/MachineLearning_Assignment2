import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# get_ipython().run_line_magic('ma  tplotlib', 'inline')
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from scripts import get_data


data = get_data.data()
data.load_cifar_labels()
data.load_cifar_train(0)
data.load_cifar_test()
# print(data.train_data[0].shape)
# plt.imshow(x[0])

def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

print("in")
start_time = time.time()
print(svc_param_selection(data.train_data, data.train_labels, 5))
print("out")
print("--- %s seconds ---" % (time.time() - start_time))
