import csv as csv
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv") 
# print(train)


from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
# from sklearn import svm
# clf = svm.SVC(kernel= 'poly')


# def accuracy(predictions):
#     count = 0.0
#     for i in range(len(predictions)):
#         if predictions[i] == train["label"][i]:
#             count = count + 1.0
#     print("Count",count)
            
#     accuracy = count/len(predictions)
#     print ("--- Accuracy value is " + str(accuracy))
#     return accuracy

predictors = []
for i in range(784):
	string = "pixel" + str(i)
	predictors.append(string)

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=2, min_samples_leaf=1)
# print(alg)

# scores = cross_validation.cross_val_score(alg, train[predictors], train["label"], cv=3)

alg.fit(train[predictors], train["label"])

# predictions = alg.predict_proba(train[predictors]).astype(float)
# print(predictions)		#pham tram [0,0,0.11,]..[1,1.999,1]
# print("----")			
# predictions = predictions.argmax(axis=1)
# print(predictions)		# label cac hang [1,0,...]

# accuracyV = accuracy(predictions)
# print(len(predictions))

# submission = pd.DataFrame({
#         "true value": train["label"],
#         "label": predictions
#     })


# filename = str('%0.5f' %accuracyVs) + "_test_mnist.csv"
##-----------
predictions = alg.predict_proba(test[predictors]).astype(float)

predictions = predictions.argmax(axis=1)

ImageId = []
for i in range(1, 28001):
	ImageId.append(i)

submission = pd.DataFrame({
        "ImageId": ImageId,
        "Label": predictions
    })
submission.to_csv("sample_submission.csv", index=False)









# print(predictions)


# print(train['pixel1'])
# print(train[predictors])
# print(scores)
# print("Cross validation scores = " + str(scores.mean()))


#----
# alg.fit(train[predictors], train["label"])
# kq = alg.score(test[predictors], train["label"])	
# print(kq)