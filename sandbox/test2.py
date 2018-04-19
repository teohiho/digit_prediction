from sklearn import svm
from sklearn.externals import cross_val_score
iris = datasets.load_iris()
X, y = iris.data, iris.target
print(X)
print(y)
