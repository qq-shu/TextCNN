from sklearn.ensemble import RandomForestClassifier as RF
# from sklearn import cross_validation
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import pandas as pd

subtrainLabel = pd.read_csv('subtrainLabels.csv')
subtrainfeature = pd.read_csv("3gramfeature.csv")
subtrain = pd.merge(subtrainLabel,subtrainfeature,on='Id')
labels = subtrain.Class
subtrain.drop(["Class","Id"], axis=1, inplace=True)
subtrain = subtrain.values


# X_train, X_test, y_train, y_test = model_selection.train_test_split(subtrain,labels,test_size=0.4)

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold,KFold

srf = RF(n_estimators=200, n_jobs=-1)
kfolder = KFold(n_splits=10,random_state=1)
scores4=cross_val_score(srf, subtrain, labels,cv=kfolder)
print(scores4)
print(scores4.mean())

sfolder = StratifiedKFold(n_splits=4,random_state=0)
sfolder = StratifiedKFold(n_splits=4,random_state=0)
scores3=cross_val_score(srf, subtrain, labels,cv=sfolder)
print(scores3)
print(scores3.mean())


clf = KNeighborsClassifier()
kfolder = KFold(n_splits=10,random_state=1)
scores=cross_val_score(clf, subtrain, labels,cv=kfolder)
print(scores)
print(scores.mean())


from sklearn.svm import SVC
clf2 = SVC(kernel='rbf', probability=True)
sfolder = StratifiedKFold(n_splits=4,random_state=0)
scores2=cross_val_score(clf2, subtrain, labels,cv=sfolder)
print(scores2)
print(scores2.mean())



# srf = RF(n_estimators=200, n_jobs=-1)
# srf.fit(X_train,y_train)
# print (srf.score(X_test,y_test))
# y_pred = srf.predict(X_test)
# print (confusion_matrix(y_test, y_pred))