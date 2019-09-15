from numpy import genfromtxt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from itertools import cycle
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


dataset = genfromtxt('D:\Python\processed.cleveland.data',delimiter=',')
X = dataset[:,0:13]  # Ulazni podaci
Y = dataset[:,13]    # Izlazni podaci

# Prebacivanje izlaznih vrijednosti u binarni sustav (0 i 1)
for index, item in enumerate(Y):   
	if not (item == 0.0):
		Y[index] = 1
#print("\nIzlazne vrijednosti nakon prebacivanja u binarni sustav:\n\n" + str(Y))
target_names = ['0','1']

# Funkcija za plotanje u 2D nakon PCA
def plot_2D(data,target,target_names):
	colors = cycle('gmykw')
	target_ids = range(len(target_names))
	plt.figure()
	for i,c, label in zip(target_ids, colors, target_names):
		plt.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label)
	plt.legend()
	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(7.5, 5.5)
	plt.show()

# Funkcija za plotanje rezultata SVM-a
def plotRBF(X_new, Y, model, title):
    X_min, X_max = X_new[:,0].min() - 1, X_new[:,0].max() + 1
    Y_min, Y_max = X_new[:,1].min() - 1, X_new[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(X_min, X_max,0.2),
    	                 np.arange(Y_min, Y_max,0.2))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z, alpha=0.1)
    
    # Dio koda koji stvara polje s vrijednostima 'red' i 'blue' koje odgovaraju
    # jedinicama i nulama iz "Y" respektivno
    Y_color = []
    for yc in Y:
        if yc == 0:
            Y_color.append('red')
        else:
            Y_color.append('blue')    
    Y_color = np.asarray(Y_color)   # prebacivanje liste u polje
         
    plt.scatter(X_new[:,0], X_new[:,1], c=Y_color)  # odabir stupaca iz X_new te boje (parametar "c")
    plt.xlabel("PCA1")   # naziv x osi
    plt.ylabel("PCA2")   # naziv y osi
    plt.xlim(xx.min(),xx.max())  # postavljanje granica x osi
    plt.ylim(yy.min(),yy.max())  # postavljanje granica y osi
    plt.xticks(())      # micanje oznaka sa osi
    plt.yticks(())      # micanje oznaka sa osi
    plt.title(title)   # postavljanje naslova
    plt.show()  # prikazi sliku

    
# Primjena PCA - skaliranje ulaza na drugu dimenziju
pca = PCA(n_components=2, whiten=True).fit(X)   
X_new = pca.transform(X)
# Plotanje u 2D
#print("\n2D prikaz reduciranih dimenzija:\n")
#plot_2D(X_new, Y, target_names)

# Podjela ulaznih podataka na setove za treniranje i testiranje u omjeru 80% : 20%
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X_new, Y, test_size = 0.2, train_size=0.8, random_state=0)


#Predvidanje i evaluacija bez podjele podataka .............................
gamma = [0.01, 0.1, 1]
kernel = ["linear", "poly", "rbf", "sigmoid"]
C = [0.1, 1.0]

modelSVMraw = SVC(C = C[0],\
           kernel=kernel[0],\
           gamma=gamma[0])
modelSVMraw = modelSVMraw.fit(X_new,Y) # model bez odvanjana podataka za testiranje

modelSVMsplit = SVC(C = C[1],\
           kernel=kernel[3],\
           gamma=gamma[2])
modelSVMsplit = modelSVMsplit.fit(X_train, Y_train) # model s podacima za testiranje



#print("\n-------------------- BEZ PODJELE (CIJELI SET) ------------------------")

predict = modelSVMraw.predict(X_new)
#print ("\nPredikcije testnog seta bez podjele podataka:\n\n" + str(predict))
"""
# Evaluacija modela
print("\nConfusion matrix:\n")
print(confusion_matrix(Y,predict))
print("\nClassification report:\n")
print(classification_report(Y, predict))

print("SVM score without split:")
print(modelSVMraw.score(X_new, Y))
"""
#plotRBF(X_new,Y, modelSVMraw, "SVM - RAW")



"""
print("\n-------------------- S PODJELOM NA TRAIN I TEST ----------------------")

predict = modelSVMsplit.predict(X_test)
print ("\nPredikcije testnog seta nakon podjele podataka:\n\n" + str(predict))

# Evaluacija modela
print("\nConfusion matrix:\n")
print(confusion_matrix(Y_test,predict))
print("\nClassification report:\n")
print(classification_report(Y_test, predict))
"""
print("SVM score with split:")
print(modelSVMsplit.score(X_test, Y_test))

#plotRBF(X_test,Y_test, modelSVMsplit, "SVM - SPLIT") 



