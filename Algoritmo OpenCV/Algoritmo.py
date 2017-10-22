# Projeto de conclusao de curso de Bacharelado em Ciencia da Computacao
# Codigo detector de gatos siameses em imagens
# Desenvolvido por Lucas Augusto Cavenaghi
# Orientador Alexandre Ferreira Mello

import cv2
import os.path
import numpy as np
import time
 
def _criarListaPositivos(quantidade):
	print("----------Criando Lista de Positivos")
	arquivo = open('positivos.dat', 'w')
	for i in range(1,quantidade+1):
		arquivo.write("bdsaida/img%d.jpg 1 0 0 100 100\n" %i)
	print("----------Lista de Positivos criada")
	arquivo.close()

def _criarListaNegativos(quantidade):
	print("----------Criando Lista de Negativos")
	arquivo = open('negativos.dat', 'w')
	for i in range(1,quantidade+1):
		arquivo.write("bdnegativos/img%d.jpg\n" %i)
	print("----------Lista de Negativos criada")
	arquivo.close()

# utiliza haarcascadefrontalcatface que vem do OPENCV para processar imagem e retornar somente a face do gato
def _processa(quantidade):
	for contador in range(1,quantidade+1):
		if os.path.isfile("bdentrada/a (%d).jpg" % contador):
			# transforma em cinza
			imagemNormal = cv2.imread("bdentrada/a (%d).jpg" %contador)
			imagemCinza = cv2.cvtColor(imagemNormal, cv2.COLOR_BGR2GRAY)

			# cria imagem cortada
			imagemCortada = imagemCinza

			# detectando
			detector = cv2.CascadeClassifier("util/haarcascade_frontalcatface.xml")
			rects = detector.detectMultiScale(imagemCinza, scaleFactor=1.3,
				minNeighbors=10, minSize=(75, 75))

			# salvar imagem
			for (i, (x, y, w, h)) in enumerate(rects):
				cv2.rectangle(imagemCinza, (x, y), (x + w, y + h), (0, 0, 255), 2)
				imagemCortada = imagemCinza [y:y+h, x:x+w]
				#cv2.imshow("imagem",imagemCinza)
				imagemCortada = cv2.resize(imagemCortada, (100, 100)) 
				cv2.imwrite("bdsaida/img%d.jpg" % contador,imagemCortada)
				#cv2.imwrite("bdsaida/img%d.jpg" % contador,imagemNormal)
				cv2.waitKey(0)

			print("----------Imagem %d ajustada" %contador)

# cria arquivo anotation e as imagens positivas
def _criarAnnotations(quantidade, quantidadePorFoto):
	for contador in range(1,quantidade+1):
		executar = 'opencv_createsamples -img bdsaida/img%d.jpg -bg negativos.dat -info info/annotations%d.lst -maxxangle 0.1 -maxyangle 0.1 -maxzangle 0.1 -num %d' % (contador,contador,quantidadePorFoto)
		os.system(executar)
		time.sleep(1)
		print("----------Annotation %d criado" %contador)

def _juntarAnnotations(quantidade):
	print("----------Anexando Annotations")
	annotations = open("info/annotations.lst", "w")
	for contador in range(1,quantidade+1):
		arqtemp = open("info/annotations%d.lst" % contador, "r")
		annotations.writelines(arqtemp)
		print("----------Anexando Annotation%d" %contador)
		arqtemp.close() 
	print("----------Annotations anexados")
	annotations.close()

def _criarVetor(quantidade):
	print("----------Criando Vetor")
	os.system('opencv_createsamples -info info/Annotations.lst -vec positivos.vec -w 20 -h 20 -num %d' %quantidade)
	print("----------Vetor de Positivos Criado")

def _treinar(quantidade1, quantidade2):
	print("----------Iniciando Treinamento")
	#os.system('opencv_traincascade -data data -vec positivos.vec -bg negativos.dat -numPos %d -numNeg %d -numStages 10 -minHitRate 0.8 -w 20 -h 20' % (quantidade, quantidade/2))
	#os.system('opencv_traincascade -data data -vec positivos.vec -bg negativos.dat -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos %d -numNeg %d -w 20 -h 20 -mode ALL -precalcValBufSize 1024  -precalcIdxBufSize 1024' % (quantidade, quantidade*0.6))
	#os.system('opencv_traincascade -data data -vec positivos.vec -bg negativos.dat -numPos %d -numNeg %d-numStages 15 -featureType HAAR -minHitRate 0.999 -maxFalseAlarmRate 0.5 -w 20 -h 20' % (quantidade1, quantidade2))
	os.system('opencv_traincascade -data data -vec positivos.vec -bg negativos.dat -numPos %d -numNeg %d -numStages 18  -featureType HAAR -minHitRate 0.999 -maxFalseAlarmRate 0.5 -w 20 -h 20' % (quantidade1, quantidade2))
	print("----------Treinamento concluido")

def _mostrarVetor():
	os.system('opencv_createsamples -vec positivos.vec -show -w 20 -h 20')

# teste com camera - ESC sai da funcao
def _camera():
	cap = cv2.VideoCapture(0)

	while 1:
	    ret, img = cap.read()
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    detector =cv2.CascadeClassifier('haarcascade_gatoSiames.xml')
	    gato = detector.detectMultiScale(gray, scaleFactor=1.3,
		minNeighbors=10, minSize=(75, 75))
	    for (x,y,w,h) in gato:
	        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

	    # atualizar a imagem
	    cv2.imshow('img',img)
	    k = cv2.waitKey(30) & 0xff
	    if k == 27:
	        break

	cap.release()
	cv2.destroyAllWindows()

# testa a imagem teste.jpg na pasta raiz do projeto
def _testaFace():
	# transforma em cinza
	imagemNormal = cv2.imread("teste.jpg")
	imagemCinza = cv2.cvtColor(imagemNormal, cv2.COLOR_BGR2GRAY)

	# cria imagem cortada
	imagemCortada = imagemCinza

	# detectando
	detector = cv2.CascadeClassifier("util/haarcascade_frontalcatface.xml")
	gato = detector.detectMultiScale(imagemCinza, scaleFactor=1.3,
		minNeighbors=10, minSize=(75, 75))

	# retangular imagem
	for (i, (x, y, w, h)) in enumerate(gato):
		cv2.rectangle(imagemCinza, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# mostra e salva imagem
	cv2.imshow("Face", imagemCinza)
	cv2.waitKey(0)


# testa a imagem teste.jpg na pasta raiz do projeto
def _testaBanco(quantidade):
	for contador in range(1,quantidade+1):
		if os.path.isfile("bdtestes/a (%d).jpg" % contador):
			# transforma em cinza
			imagemNormal = cv2.imread("bdtestes/a (%d).jpg" %contador)
			imagemCinza = cv2.cvtColor(imagemNormal, cv2.COLOR_BGR2GRAY)

			# cria imagem cortada
			imagemCortada = imagemCinza

			# detectando
			detector = cv2.CascadeClassifier("haarcascade_gatoSiames.xml")
			rects = detector.detectMultiScale(imagemCinza, scaleFactor=1.3,
				minNeighbors=10, minSize=(75, 75))

			# salvar imagem
			for (i, (x, y, w, h)) in enumerate(rects):
				cv2.rectangle(imagemCinza, (x, y), (x + w, y + h), (0, 0, 255), 2)
				imagemCortada = imagemCinza [y:y+h, x:x+w]
				#cv2.imshow("imagem",imagemCinza)
				cv2.imwrite("bdresultado/img%d.jpg" % contador,imagemCortada)
				#cv2.imwrite("bdsalvo/img%d.jpg" % contador,imagemNormal)
				cv2.waitKey(0)


###chamada de funcoes comeca aqui###
## Configuracao de Imagens
numImgPositivas	= 500
numImgGerar		= 10
numImgPosGerada = numImgPositivas * numImgGerar - 1000
numImgNegativas = numImgPosGerada / 2

## inicio
#_criarListaPositivos(numImgGerar)
#_criarListaNegativos(numImgNegativas)

## algoritmo
_processa(numImgPositivas)
_criarAnnotations(numImgPositivas, numImgGerar)
_juntarAnnotations(numImgPositivas)
_criarVetor(numImgPosGerada)
_treinar(numImgPosGerada, numImgNegativas)

## testes
#_mostrarVetor()
#_camera()
#_testaFace()
#_testaBanco(100)

###fim###