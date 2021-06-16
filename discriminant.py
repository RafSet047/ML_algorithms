import numpy as np

class BinaryLDA:
     
    def __init__(self):
        self.w = None
        self.t = None

    def get_covariance_matrix(self, X):
        """
        Calculate the covariance matrix for the dataset X
        """

        nr_samples = X.shape[0]
        return (1 / (nr_samples - 2)) * (X - X.mean(axis=0)).T @ (X - X.mean(axis=0))

    def fit(self, X, y):
      # Separate data by class for convenience
      X1 = X[y == 0]
      X2 = X[y == 1]

      # Calculate the covariance matrices of the two datasets
      cov1 = self.get_covariance_matrix(X1)
      cov2 = self.get_covariance_matrix(X2)
      
      # cov1 and cov2 should already be normalized,
      # therefore we just add them to get sigma
      sigma = cov1 + cov2   

      # Calculate the mean of the two datasets (mu_k in our lecture slides)
      mean1 = X1.mean(0)
      mean2 = X2.mean(0)
      mean_diff = mean1 - mean2
      mean_sum = mean1 + mean2

      # Calculate the class priors
      p1 = sum(y == 0) / len(y)
      p2 = sum(y == 1) / len(y)

      # Get the inverse of sigma
      sigma_inv = np.linalg.pinv(sigma)

      # determine the decision boundary w*x=t


      self.t = -np.log(p1/p2) + 1/2 * (mean_sum).T @ sigma_inv @ (mean_diff)
      self.w = sigma_inv @ mean_diff

    def predict(self, X):
      y_pred = ((X @ self.w - self.t) < 0) * 1
      return y_pred

class LDA:

    def __init__(self):
        self.delta = None
        self.nr_labels = None
        self.nr_data = None
        self.labels = None

	
    def get_covariance_matrix(self, X):

        """Calculate the covariance matrix for the dataset X """
        # It is recommended that you try to compute the covariance matrix
        # by performing the needed operations by hand, instead of using np.cov()
        # function, this way you will have better idea of the covariance matrix

        # Also take into account that the random variables for which you want to
        # compute the covariance matrix are the columns of X, not the rows!

        # don't forget the 1/(N-K) term in this case
        # YOUR CODE HERE

        return (1 / (self.nr_data - self.nr_labels)) * (X - X.mean(axis=0)).T @ (X - X.mean(axis=0))
        
        
    def fit(self, X, y):
        # compute means (mu_k), priors (p_k) for each class and
        # sigma by adding the class covariance matrices
        self.labels, counts = np.unique(y, return_counts=True)
        self.nr_labels = len(self.labels)
        self.nr_data = X.shape[0]
        mu = []
        p = []
        sigma = np.zeros((X.shape[1], X.shape[1]))
        for i in range(self.nr_labels):
            mu.append(X[y == self.labels[i]].mean(0))
            p.append(counts[i] / sum(counts))
            sigma += self.get_covariance_matrix(X[y == self.labels[i]])
        mu = np.array(mu)
        p = np.array(p)


        # get the inverse of sigma
        sigma_inv = np.linalg.pinv(sigma)

        # remember that we need to compute the values of the
        # discriminant functions (deltas) for each class, so we will need
        # the respective coefficients to use in the 'predict' method

        # you can store those coefficients in some data structure
        delta = []
        for label_ind in range(self.nr_labels):
            w = sigma_inv @ mu[label_ind]
            t = mu[label_ind] @ w / 2 + np.log(p[label_ind])
            delta.append([w, t])



        self.delta = delta
    
    def get_values(self, x):
    
        delta = self.delta
        values = []
        for i in range(self.nr_labels):
           values.append(x.T @ delta[i][0] - delta[i][1])
        return values

    def predict(self, X):
        # use the coefficients in self.deltas to compute delta_k per class
        # for each instance from X and select the class
        # which has the highest delta_k
        delta = self.delta
        y_pred = []
        for i in range(X.shape[0]):
           deltas = self.get_values(X[i])
           y_pred.append(self.labels[np.argmax(deltas)])
        return np.array(y_pred)


class QDA:


    def __init__(self):
        self.delta = None
        self.nr_labels = None
        self.nr_data = None
        self.labels = None
	

    def get_covariance_matrix(self, X):
        """ Calculate the covariance matrix for the dataset X """
        # what should be instead of K in this term 1/(N-K) in this case ?!
        covariance_matrix =  (1 / (self.nr_data - 1)) * (X - X.mean(axis=0)).T @ (X - X.mean(axis=0))
        return covariance_matrix

    def fit(self, X, y):

        # compute means (mu_k), priors (p_k) and sigma_k for each class
        # you will also need the determinant and inverse of each sigma_k




        # discriminant functions (deltas) for each class, so we will need
        # the respective coefficients to use in the 'predict' method

        # you can store those coefficients in some data structure
        self.labels, counts = np.unique(y, return_counts=True)
        self.nr_labels = len(self.labels)
        self.nr_data = X.shape[0]

        delta = []
        for i in range(self.nr_labels):
            mu = (X[y == self.labels[i]]).mean(0)
            p = (counts[i] / sum(counts))
            sigma = (self.get_covariance_matrix(X[y == self.labels[i]]))

            sigma_det = np.linalg.det(sigma)
            sigma_inv = np.linalg.pinv(sigma)

            w = sigma_inv @ mu
            t = mu.T @ w / 2 - np.log(p) + 0.5 * np.log(sigma_det)
            delta.append([w, t, sigma_inv])

        self.delta = delta
    
  
  

    def get_values(self, x):
        delta = self.delta
        values = []
        for i in range(self.nr_labels):
            values.append(x.T @ delta[i][0] - 0.5 * x.T @ delta[i][2] @ x - delta[i][1])
        return values

    def predict(self, X):
        # the coefficients in self.deltas to compute delta_k per class
        # for each instance from X and select the class
        # which has the highest delta_k
        y_pred = []
        for i in range(X.shape[0]):
            deltas = self.get_values(X[i])
            y_pred.append(self.labels[np.argmax(deltas)])

        return y_pred



