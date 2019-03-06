import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

exp_var_percentage = 90 # Threshold of 90% explained variance

###### NON-LINEAR DATA
X1 = np.array([a**30.0 for a in range(1,100, 3)])
X2 = np.array([b for b in range(1, 100, 3)])
X = np.concatenate((np.matrix(X1), np.matrix(X2)), axis=0).reshape(X1.size, 2)
Y = np.array(X1 + X2)
print("============================================")
print("PCA ON NON-LINEAR DATA")
print("============================================")

fig, ax = plt.subplots()
plt.title("Non linear dataset")
ax.scatter(X1, X2, c=Y, cmap="bwr")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()

X = StandardScaler().fit_transform(X) # Normalize data to zero mean and 1 variance

cov_mat = np.cov(X.T)
print(cov_mat)

# Compute the eigen values and vectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
print("eig_pairs", eig_pairs)

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in map(lambda x: x[0], eig_pairs)]
cum_var_exp = np.cumsum(var_exp)
print("Explained var ratio:", var_exp)
print("Acummulative explained var ratio:", cum_var_exp)

num_vec_to_keep = 1
for index, percentage in enumerate(cum_var_exp):
    if percentage > exp_var_percentage:
        num_vec_to_keep = index + 1
        break

# Compute the projection matrix based on the top eigen vectors
num_features = X.shape[1]
for eig_vec_idx in range(0, num_vec_to_keep):
    print("-> keep", eig_pairs[eig_vec_idx])



###### LINEAR DATA
X1 = np.array([float(a) for a in range(1,100, 3)])
X2 = np.array([b for b in range(1, 100, 3)])
X = np.concatenate((np.matrix(X1), np.matrix(X2)), axis=0).reshape(X1.size, 2)
Y = np.array(X1 + X2)
print("============================================")
print("PCA ON LINEAR DATA")
print("============================================")

fig, ax = plt.subplots()
plt.title("Linear dataset")
ax.scatter(X1, X2, c=Y, cmap="bwr")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()

X = StandardScaler().fit_transform(X) # Normalize data to zero mean and 1 variance

cov_mat = np.cov(X.T)
print(cov_mat)

# Compute the eigen values and vectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
print("eig_pairs", eig_pairs)

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in map(lambda x: x[0], eig_pairs)]
cum_var_exp = np.cumsum(var_exp)
print("Explained var ratio:", var_exp)
print("Acummulative explained var ratio:", cum_var_exp)

num_vec_to_keep = 1
for index, percentage in enumerate(cum_var_exp):
    if percentage > exp_var_percentage:
        num_vec_to_keep = index + 1
        break

# Compute the projection matrix based on the top eigen vectors
num_features = X.shape[1]
for eig_vec_idx in range(0, num_vec_to_keep):
    print("-> keep", eig_pairs[eig_vec_idx])
