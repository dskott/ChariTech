# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:04:45 2017

@author: Georgios
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def analysis(X,y):
    X1=X[:,:2]
    X_2d = X[:, :2]
    X_2d = X_2d[y > 0]
    y_2d = y[y > 0]
    y_2d -= 1

    # It is usually a good idea to scale the data for SVM training.
    # We are cheating a bit in this example in scaling all of the data,
    # instead of fitting the transformation on the training set and
    # just applying it on the test set.
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_2d = scaler.fit_transform(X_2d)
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

    # Now we need to fit a classifier for all parameters in the 2d version
    # (we use a smaller set of parameters here because it takes a while to train)

    C_2d_range = [1e-2, 1, 1e2]
    gamma_2d_range = [1e-1, 1, 1e1]
    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(C=C, gamma=gamma)
            clf.fit(X_2d, y_2d)
            classifiers.append((C, gamma, clf))
    plt.figure(figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    for (k, (C, gamma, clf)) in enumerate(classifiers):
        # evaluate decision function in a grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # visualize decision function for these parameters
        plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
        plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
                  size='medium')

        # visualize parameter's effect on decision function
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.
    
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()
    
    h = .02  # step size in the mesh
    X=X1
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=C).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
                            plt.subplot(2, 2, i + 1)
                            plt.subplots_adjust(wspace=0.4, hspace=0.4)
                            
                            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                            
                            # Put the result into a color plot
                            Z = Z.reshape(xx.shape)
                            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
                            
                            # Plot also the training points
                            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
                            plt.xlabel('Sepal length')
                            plt.ylabel('Sepal width')
                            plt.xlim(xx.min(), xx.max())
                            plt.ylim(yy.min(), yy.max())
                            plt.xticks(())
                            plt.yticks(())
                            plt.title(titles[i])
                            
iris = load_iris()
X = iris.data
y = iris.target
analysis(X,y)