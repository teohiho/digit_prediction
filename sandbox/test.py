from sklearn import svm
from sklearn import datasets
clf = svm.SVC()				
iris = datasets.load_iris()
X, y = iris.data, iris.target
# print(X) 					#X la mang 2 chieu
# print(y) 					#y la list

clf.fit(X,y) 				#fit cho may hoc
							# X la n bai tap, y la tap dap an
# clf2 = pickle.loads(s)
# test = clf.predict(X[0:1])		
# print(test)	

#doan cho vui =))
# sample =  [4.6 3.1 1.5 0.2]
# predict_class= clf.predict([sample])	#doan
# print(predict_class)


# cross validation
#----Mot hoc cach khac
kq = clf.score(X, y)		#X de thi, y dap an	
print(kq)
