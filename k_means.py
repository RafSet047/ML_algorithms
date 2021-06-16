import numpy as np

class KMeans:
    def __init__(self, k=2, max_iterations=500, tol=0.5):
      # number of clusters
      self.k = k
      # maximum number of iterations to perform
      # for updating the centroids
      self.max_iterations = max_iterations
      # tolerance level for centroid change after each iteration
      self.tol = tol
      # we will store the computed centroids
      self.centroids = None

    def init_centroids(self, X):
      # this function initializes the centroids
      # by choosing self.k points from the dataset
      # Hint: you may want to use the np.random.choice function
      idx = np.random.choice(range(X.shape[0]), self.k, replace=False)
      centroids = X[idx, :]
      return centroids

    def closest_centroid(self, X):
      # this function computes the distance (euclidean) between
      # each point in the dataset from the centroids filling the values
      # in a distance matrix (dist_matrix) of size n x k
      # Hint: you may want to remember how we solved the warm-up exercise
      # in Programming module (Python_Numpy2 file)
      dist_matrix = np.zeros((X.shape[0], self.k))
      for i in range(self.k):
          dist_matrix[:, i] = np.sqrt(np.sum((self.centroids[i] - X)**2, axis=1))
      # after constructing the distance matrix, you should return
      # the index of minimal value per row
      # Hint: you may want to use np.argmin function
      return np.argmin(dist_matrix, axis=1)

    def update_centroids(self, X, label_ids):
      # this function updates the centroids (there are k centroids)
      # by taking the average over the values of X for each label (cluster)
      # here label_ids are the indices returned by closest_centroid function
      new_centroids = np.zeros((self.k, X.shape[1]))
      for i in range(self.k):
          new_centroids[i] = X[np.where(label_ids==i)].mean(axis=0)
      return new_centroids

    def fit(self, X):
      # this is the main method of this class
      X = np.array(X)
      # we start by random centroids from our data
      self.centroids = self.init_centroids(X)

      not_converged = True
      i = 1 # keeping track of the iterations
      while not_converged and (i < self.max_iterations):
        current_labels = self.closest_centroid(X)
        new_centroids = self.update_centroids(X, current_labels)

        # count the norm between new_centroids and self.centroids
        # to measure the amount of change between
        # old cetroids and updated centroids
        norm = np.linalg.norm(self.centroids - new_centroids)
        not_converged = norm > self.tol
        self.centroids = new_centroids
        i += 1
      self.labels = current_labels
      print(f'Converged in {i} steps')

    def predict(self, X):
      # we can also have a method, which takes a new instance (instances)
      # and assigns a cluster to it, by calculating the distance
      # between that instance and the fitted centroids
      # returns the index (indices) of of the cluster labels for each instance
      X = np.array(X)
      dist_mat = np.zeros((X.shape[0], self.k))
      for i in range(self.k):
          dist_mat[:, i] = np.sqrt(np.sum((self.centroids[i] - X)**2, axis=1))
      return np.argmin(dist_mat, axis=1)