import numpy as np


class PCA:

    def __init__(self, nr_components):
      self.nr_components = nr_components

      # we will store the PC coordinates here
      self.components = None
      # how much variance is explained with the PCs
      self.explained_variance = None
      # how much variance is explained with the PCs among the total variance
      self.explained_variance_ratio = None

    def fit(self, X):
      # this method is used to compute the PC components (projection matrix)
      nr_components = self.nr_components

      # compute the covariance matrix of the given dataset
      # note that we are interested in covariance in terms of the
      # features (columns) of our dataset
      covariance_matrix = (1 / X.shape[0]) * (X - X.mean()).T @ (X - X.mean())

      # get the eigenvalues and eigenvectors of the covariance matrix
      eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

      # get the indices of the first nr_components eigenvalues
      idx = eigenvalues.argsort()[::-1]
      self.explained_variance = sum(eigenvalues[idx][:nr_components])
      self.explained_variance_ratio = self.explained_variance / sum(eigenvalues)

      # select the first nr_components eigenvectors as the projection matrix
      self.components = eigenvectors[:, idx][:,:nr_components]

    def transform(self, X):
      # this method will project the initial data to the new subspace
      # spanned with the principal components, here you will need self.components
      return X @ self.components

    def fit_transform(self, X):
      self.fit(X)
      return self.transform(X)