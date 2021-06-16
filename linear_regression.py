import numpy as np

class MyLinearRegression:

    def __init__(self, regularization=None, lam=0, learning_rate=1e-3, tol=0.05):
      """
      This class implements linear regression models
      Params:
      --------
      regularization - None for no regularization
                      'l2' for ridge regression
                      'l1' for lasso regression

      lam - lambda parameter for regularization in case of
          Lasso and Ridge

      learning_rate - learning rate for gradient descent algorithm,
                      used in case of Lasso

      tol - tolerance level for weight change in gradient descent
      """

      self.regularization = regularization
      self.lam = lam
      self.learning_rate = learning_rate
      self.tol = tol
      self.weights = None
  
    def fit(self, X, y):
    
      X = np.array(X)
      # first insert a column with all 1s in the beginning
      # hint: you can use the function np.insert
      X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

      if self.regularization is None:
        # the case when we don't apply regularization
        self.weights = np.linalg.inv(X.T @ X) @ (X.T @ y)
      elif self.regularization == 'l2':
        # the case of Ridge regression
        self.weights = np.linalg.inv(X.T @ X + self.lam * np.eye(X.shape[0])) @ (X.T @ y)
      elif self.regularization == 'l1':
        # in case of Lasso regression we use gradient descent
        # to find the optimal combination of weights that minimize the
        # objective function in this case (slide 37)

        # initialize random weights, for example normally distributed
        self.weights = np.random.randn(X.shape[1])

        converged = False
        # we can store the loss values to see how fast the algorithm converges
        self.loss = []
        # just a counter of algorithm steps
        i = 0
        while (not converged):
          i += 1
          # calculate the predictions in case of the weights in this stage
          y_pred = X @ self.weights
          # calculate the mean squared error (loss) for the predictions
          # obtained above
          self.loss.append((1 / X.shape[0]) * np.sum((y_pred - y)**2))
          # calculate the gradient of the objective function with respect to w
          # for the second component \sum|w_i| use np.sign(w_i) as it's derivative
          grad = -2 * X.T @ (y - y_pred) + self.lam * np.sign(self.weights)
          new_weights = self.weights - self.learning_rate * grad
          # check whether the weights have changed a lot after this iteration
          # compute the norm of difference between old and new weights
          # and compare with the pre-defined tolerance level, if the norm
          # is smaller than the tolerance level then we consider convergence
          # of the algorithm
          learned = np.linalg.norm(self.weights - new_weights)
          if learned >= self.tol:
              converged = False
          else:
              converged = True
          self.weights = new_weights
        print(f'Converged in {i} steps')

    def predict(self, X):
      X = np.array(X)
      # don't forget to add the feature of 1s in the beginning
      X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
      # predict using the obtained weights
      return X @ self.weights
  


