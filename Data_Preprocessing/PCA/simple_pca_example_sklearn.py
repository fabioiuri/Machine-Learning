import numpy as np
from sklearn import decomposition, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
ppn = Perceptron(max_iter=1000, tol=1e-5, eta0=0.01, random_state=1)
ppn.fit(X_train, y_train)
print("============================================")
print("WITHOUT PCA")
print("--> Accuracy on training set:",  ppn.score(X_train, y_train))
print("--> Accuracy on test set:",  ppn.score(X_test, y_test))
print("============================================")

# first 3 features contain the highest singular values
# so drop the last one
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

print("Explained var ratio:", pca.explained_variance_ratio_)  
print("Singular values:", pca.singular_values_)

# Now learn perceptron on projected data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
ppn = Perceptron(max_iter=1000, tol=1e-5, eta0=0.01, random_state=1)
ppn.fit(X_train, y_train)
print("============================================")
print("WITH PCA")
print("--> From",  iris.data[0].size, "dimensions to", X[0].size)
print("--> Accuracy on training set:",  ppn.score(X_train, y_train))
print("--> Accuracy on test set:",  ppn.score(X_test, y_test))
print("============================================")