# Projeto de conclusao de curso de Bacharelado em Ciencia da Computacao
# Codigo detector de gatos siameses em imagens
# Desenvolvido por Lucas Augusto Cavenaghi
# Orientador Alexandre Ferreira Mello


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from imutils import paths
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import imutils
import cv2
import os


#-----------------------------------------------------------------------------------------------------------#
#----------------------------------------FUNCAO DE EXTRACAO-------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
# criacao de histograma
def extract_color_histogram(image):
  # separando em cores
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  #                   imagem, B  G  R,mascara, 10 features,  procurar de 0 a tanto
  hist = cv2.calcHist([hsv], [0, 1, 2], None, (10, 10, 10) ,[0, 180, 0, 256, 0, 256])

  # histograma escala de tons
  #plt.hist(image.ravel(),256,[0,256]); plt.show()

  # histograma colorido  
  color = ('b','g','r')
  for i,col in enumerate(color):
      histr = cv2.calcHist([image],[i],None,[256],[0,256])
      plt.plot(histr,color = col)
      plt.xlim([0,256])
  #plt.show()

  #normalizacao opencv
  if imutils.is_cv2():
    hist = cv2.normalize(hist)
  else:
    cv2.normalize(hist, hist)


  return hist.flatten()
#-----------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------#
#------------------------------------------------- TREINO --------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
# pegando imagens na pasta datasetTreino
print("---------- Iniciando extracao features para treino")
imagePaths = list(paths.list_images("datasetTreino"))
features = []
labels = []

# pegando as imagens
for (i, imagePath) in enumerate(imagePaths):
  print("---------- Processando:{} ---------- {}/{}".format(imagePaths[i],i+1, len(imagePaths)))
  # carregar imagem e colocar label, utilizar formato: /datasetTreino/{class}.{image_num}.jpg
  image = cv2.imread(imagePath)
  label = imagePath.split(os.path.sep)[-1].split(".")[0]

  # pegando histograma das cores
  hist = extract_color_histogram(image)  
 
  # atualizando dados
  features.append(hist)
  labels.append(label)
 

# transformando em numpy arrays
features = np.array(features)
labels = np.array(labels)
print("---------- Matriz de Features: {:.2f}MB".format(features.nbytes / (1024 * 1000.0)))
#-----------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------#
#------------------------------------------------- TESTES --------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
# pegando imagens na pasta datasetTestes
print("---------- Iniciando extracao features para testes")
imagePaths = list(paths.list_images("datasetTestes"))
featuresTest = []
labelsTest = []

# pegando as imagens
for (i, imagePath) in enumerate(imagePaths):
  print("---------- Processando:{} ---------- {}/{}".format(imagePaths[i],i+1, len(imagePaths)))
  # carregar imagem e colocar label, utilizar formato: /datasetTestes/{class}.{image_num}.jpg
  image = cv2.imread(imagePath)
  label = imagePath.split(os.path.sep)[-1].split(".")[0]

  # pegando histograma das cores
  hist = extract_color_histogram(image)
 
  # atualizando dados
  featuresTest.append(hist)
  labelsTest.append(label)
 

# transformando em numpy arrays
featuresTest = np.array(featuresTest)
labelsTest = np.array(labelsTest)
print("---------- Matriz de Features: {:.2f}MB".format(featuresTest.nbytes / (1024 * 1000.0)))
#-----------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------#
#------------------------------------------- RANDOM FOREST -------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
print("---------- RandomForest")
# n_jobs = -1, roda em todos os nucleos
# estimators = numero de arvores do randomForest
clf = RandomForestClassifier(n_estimators = 30, n_jobs=-1, random_state=0)
clf.fit(features, labels)

# Predicoes
labelsPredicted = clf.predict(featuresTest)
print("---------- Acuracia total: {:.2f}%".format(100*clf.score(featuresTest,labelsTest)))

#Validacao Cruzada
scores = cross_val_score(clf, features, labels, cv=5)
print("---------- Acuracia CrossValidation:" )
print("---------- Menor: %0.2f" % (scores.min()*100))
print("---------- Maior: %0.2f" % (scores.max()*100))
print("---------- Medio: %0.2f" % (scores.mean()*100))

# Matriz de Confusao
print(pd.crosstab(labelsTest, labelsPredicted, rownames=['Verdadeiros'], colnames=['Predicoes']))

# 10 predictions em porcentagem
#print(clf.predict_proba(featuresTest)[0:10])
#-----------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------#
#------------------------------------------- GradientBoosting-----------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
print("---------- GradientBoosting")
# n_jobs = -1, roda em todos os nucleos
# estimators = numero de arvores do randomForest
clf = GradientBoostingClassifier()
clf.fit(features, labels)

# Predicoes
labelsPredicted = clf.predict(featuresTest)

print("---------- Acuracia total: {:.2f}%".format(100*clf.score(featuresTest,labelsTest)))

# Matriz de Confusao
print(pd.crosstab(labelsTest, labelsPredicted, rownames=['Verdadeiros'], colnames=['Predicoes']))

# 10 predictions em porcentagem
#print(clf.predict_proba(featuresTest)[0:10])
#-----------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------#
#---------------------------------------------- ExtraTrees  ------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
print("---------- ExtraTrees")
# n_jobs = -1, roda em todos os nucleos
# estimators = numero de arvores do randomForest
clf = ExtraTreesClassifier()
clf.fit(features, labels)

# Predicoes
labelsPredicted = clf.predict(featuresTest)

print("---------- Acuracia total: {:.2f}%".format(100*clf.score(featuresTest,labelsTest)))

# Matriz de Confusao
print(pd.crosstab(labelsTest, labelsPredicted, rownames=['Verdadeiros'], colnames=['Predicoes']))

# 10 predictions em porcentagem
#print(clf.predict_proba(featuresTest)[0:10])
#-----------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------#
#----------------------------------------- Vizinho mais proximo --------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
print("---------- Vizinho mais proximo")
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(features, labels)

# Predicoes
labelsPredicted = clf.predict(featuresTest)

acc = clf.score(featuresTest, labelsTest)
print("---------- Acuracia total: {:.2f}%".format(acc * 100))

# Matriz de Confusao
print(pd.crosstab(labelsTest, labelsPredicted, rownames=['Verdadeiros'], colnames=['Predicoes']))
#-----------------------------------------------------------------------------------------------------------#