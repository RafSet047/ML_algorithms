import numpy as np
import progressbar

# you need Regresssion trees, so use either your implementation or sklearn's
from sklearn.tree import DecisionTreeRegressor 

widgets = ['Model Training: ', progressbar.Percentage(), ' ',
            progressbar.Bar(marker="-", left="[", right="]"),
            ' ', progressbar.ETA()]

def square_loss(y, y_pred): return (y - y_pred)**2


def square_loss_gradient(y, y_pred): return - 2 * (y - y_pred)

class GradientBoostingRegressor:
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                min_impurity=1e-7, max_depth=4):      
      self.n_estimators = n_estimators
      self.learning_rate = learning_rate
      self.min_samples_split = min_samples_split
      self.min_impurity = min_impurity
      self.max_depth = max_depth
      self.bar = progressbar.ProgressBar(widgets=widgets)
      self.loss = square_loss
      self.loss_gradient = square_loss_gradient

    def fit(self, X, y):
      self.trees = [] # we will store the regression trees per iteration
      self.train_loss = [] # we will store the loss values per iteration

      # initialize the predictions (f(x) in the lectures)
      # with the mean values of y
      # hint: you may want to use the np.full function here
      self.mean_y = np.mean(y)
      y_pred = np.full(y.shape, self.mean_y)
      for i in self.bar(range(self.n_estimators)):
        tree = DecisionTreeRegressor(
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity,
                max_depth=self.max_depth) # this is h(x) from our lectures
        # get the loss when comparing y_pred with true y
        # and store the values in self.train_loss
        self.train_loss.append(self.loss(y, y_pred))

        # get the pseudo residuals
        residuals = - self.loss_gradient(y, y_pred)

        tree.fit(X, residuals) # fit the tree on the residuals
        # update the predictions y_pred using the tree predictions on X
        y_pred += self.learning_rate * tree.predict(X)

        self.trees.append(tree) # stor the tree model

    def predict(self, X):
      # start with initial predictions as vector of
      # the mean values of y_train (self.mean_y)
      y_pred = np.full(X.shape[0], self.mean_y)
      # iterate over the regression trees and apply the same gradient updates
      # as in the fitting process, but using test instances
      for tree in self.trees:
          y_pred +=  self.learning_rate * tree.predict(X)
      return y_pred

