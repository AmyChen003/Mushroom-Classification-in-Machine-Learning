import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import collections
import pydotplus
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

data = pd.read_csv("mushrooms.csv")
#print data.shape
# data=data.drop(["veil-type"],axis=1)
# print data.shape
label1=data["class"]
#print label1

lb=LabelEncoder()
for i in data.columns:
    data[i] = lb.fit_transform(data[i])
#print data.head()

feature=data.iloc[:,1:23]
# print feature.columns
label=data.iloc[:,0]
#print label.values


x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size = 0.2)

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train, y_train)
#print "clf:"+str(clf)
#

vis_data = export_graphviz(clf, out_file=None,
                         feature_names=feature.columns,
                         filled=True, rounded=True)

graph = pydotplus.graph_from_dot_data(vis_data)
# colors = ('brown', 'darkgreen')
# edges = collections.defaultdict(list)
#
# for edge in graph.get_edge_list():
#     edges[edge.get_source()].append(int(edge.get_destination()))
#
# for edge in edges:
#     edges[edge].sort()
#     for i in range(2):
#         dest = graph.get_node(str(edges[edge][i]))[0]
#         dest.set_fillcolor(colors[i])
#
graph.write_png('tree.png')
y_pred=clf.predict(x_test)
y_test=y_test.values
#
print "Accuracy is ", accuracy_score(y_test,y_pred)
#
feature=feature.values
label=label.values

kf1 = KFold(n_splits=10,shuffle = True)
for train_index, test_index in kf1.split(feature):
    X_train, X_test = feature[train_index], feature[test_index]
    Y_train, Y_test = label[train_index], label[test_index]
    clf = DecisionTreeClassifier(criterion='gini')
    clf.fit(X_train, Y_train)
    y_pred=clf.predict(X_test)
    print "Accuracy is ", accuracy_score(Y_test, y_pred)




    # cm = confusion_matrix(y_test, y_pred)
    # TPR = cm[1][1] /float(cm[1][1] + cm[1][0])
    # acc = float(cm[0][0] + cm[1][1]) / np.sum(cm)
    # print "Accuracy is ",acc


df1 = pd.DataFrame(cvAccuracy)

df1.columns = ['10-fold cv Accuracy']
df = df1.reindex(range(1, 101))
df.plot()
plt.title("Decision Tree - 10-fold Cross Validation Accuracy vs Depth of tree")
plt.xlabel("Depth of tree")
plt.ylabel("Accuracy")
plt.ylim([0.8, 1])
plt.xlim([0, 100])
plt.show()



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
#print roc_auc



plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()





clf_GNB = GaussianNB()
clf_GNB = clf_GNB.fit(x_train, y_train)
label_pred_GNB=clf_GNB.predict(x_test)
auc_GNB=classification_report(y_test, label_pred_GNB)
print auc_GNB

print (y_test!= label_pred_GNB).sum()
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, label_pred_GNB)
roc_auc = auc(false_positive_rate, true_positive_rate)
print roc_auc

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
