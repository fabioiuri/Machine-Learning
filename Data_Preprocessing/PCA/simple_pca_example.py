from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import Perceptron

iris = datasets.load_iris()
X = iris.data
y = iris.target

### Try using raw data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
ppn = Perceptron(max_iter=1000, tol=1e-5, eta0=0.01, random_state=1)
ppn.fit(X_train, y_train)
print("============================================")
print("WITHOUT PCA")
print("--> Accuracy on training set:",  ppn.score(X_train, y_train))
print("--> Accuracy on test set:",  ppn.score(X_test, y_test))
print("============================================")

### Now try using PCA first and then train the perceptron
X = StandardScaler().fit_transform(X) # Normalize data to zero mean and 1 variance

cov_mat = np.cov(X.T)
# Compute the covariance matrix to see how each feature is
# correlated to each other .
# 0 -> no correlation
# 1 -> positive correlation, when one increases the other increases
# -1 -> negative correlation, when one increases the other decreases (and vice versa)
#print(cov_mat)

# Compute the eigen values and vectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
#print("eig_pairs", eig_pairs)


# Only keep a certain number of eigen vectors based on 
# the "explained variance percentage" which tells us how 
# much information (variance) can be attributed to each 
# of the principal components
exp_var_percentage = 97 # Threshold of 97% explained variance

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in map(lambda x: x[0], eig_pairs)]
cum_var_exp = np.cumsum(var_exp)
print("Explained var ratio:", var_exp)
print("Acummulative explained var ratio:", cum_var_exp)

num_vec_to_keep = 0

for index, percentage in enumerate(cum_var_exp):
    if percentage > exp_var_percentage:
        num_vec_to_keep = index + 1
        break

# Compute the projection matrix based on the top eigen vectors
num_features = X.shape[1]
proj_mat = eig_pairs[0][1].reshape(num_features,1)
for eig_vec_idx in range(1, num_vec_to_keep):
  proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features,1)))

# Project the data 
pca_data = X.dot(proj_mat)
#print("Projected data", pca_data)

# Now learn perceptron on projected data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
ppn = Perceptron(max_iter=1000, tol=1e-5, eta0=0.01, random_state=1)
ppn.fit(X_train, y_train)
print("============================================")
print("WITH PCA")
print("--> From", num_features, "dimensions to", num_vec_to_keep)
print("--> Accuracy on training set:",  ppn.score(X_train, y_train))
print("--> Accuracy on test set:",  ppn.score(X_test, y_test))
print("============================================")