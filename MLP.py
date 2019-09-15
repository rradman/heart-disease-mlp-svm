# -*- coding: utf-8 -*-
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix


dataset = genfromtxt('D:\Python\processed.cleveland.data',delimiter=',')
X = dataset[:,0:13]  # Ulazni podaci
Y = dataset[:,13]    # Izlazni podaci

# Prebacivanje izlaznih vrijednosti u binarni sustav (0 i 1)
for index, item in enumerate(Y):   
	if not (item == 0.0):
		Y[index] = 1
print("\nIzlazne vrijednosti nakon prebacivanja u binarni sustav:\n\n" + str(Y))
target_names = ['0','1']


# Primjena PCA - skaliranje ulaza na drugu dimenziju
pca = PCA(n_components=2, whiten=True).fit(X)   
X_new = pca.transform(X)
# Plotanje u 2D

# Podjela ulaznih podataka na setove za treniranje i testiranje u omjeru 80% : 20%
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X_new, Y, test_size = 0.2, train_size=0.8, random_state=0)
    
    
# ---------------------------------- MLP -----------------------------------
print("\n---------------------------------- MLP -----------------------------------")

hidden_layer_sizes = [(5,), (10,10), (30,30,30)]
activation = ["logistic", "identity", "tanh"]
solver = ["sgd", "adam"]
alpha = [0.0001, 0.00001]

clf = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes[0],\
                    activation = activation[0],\
                    max_iter = 10000,\
                    solver = solver[1],\
                    alpha = alpha[1])

#Treniranje/Učenje klasifikatora s podacima za treniranje 
clf.fit(X_new,Y)

#Predvidanje
predict = clf.predict(X_new)
print ("\nPredikcije:\n\n" + str(predict))
score = clf.score(X_new, Y, sample_weight=None)
print ("\nScore = " + str(score))

# Evaluacija modela
print("\nConfusion matrix:\n")
print(confusion_matrix(Y,predict))
print("\nClassification report:\n")
print(classification_report(Y,predict))

def plot_decision_boundary(X,Y, clf, title = None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z,alpha=0.5)
    
    Y_color = []
    for yc in Y:
        if yc == 0:
            Y_color.append('red')  
        else:
            Y_color.append('green') 
    Y_color = np.asarray(Y_color)
    
    plt.scatter(X[:, 0], X[:, 1], c=Y_color, alpha=0.8)
    
    if title is not None:
        plt.title(title) 
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
plot_decision_boundary(X_new,Y, clf)  
 

    

