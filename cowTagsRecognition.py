import numpy as np
import argparse
import imutils
from imutils import contours
import cv2
import pandas as pd
import ipywidgets as widgets
import os
import sys
import skimage.io
import scipy
import json
import matplotlib as plt
from PIL import Image

ap = argparse.ArgumentParser() 
ap.add_argument("-i", "--image", required=True, help="path de la imagen que ingresa") #--image es el path de la imagen que quiero reconocer con OCR.
ap.add_argument("-r", "--reference", required=True, help="path a la imagen de referencia") #--reference es el path a la imagen que tiene los digitos 0-9 .

ejemplo = os.path.abspath('C:\\Users\\Equipo\\Desktop\\Vacas\\Tags\\Train\\5044.png')
reference = os.path.abspath('C:\\Users\\Equipo\\Desktop\\Vacas\\referencea.png') #la imagen de referencia contiene digitos de 0 al 9 en arial black

ref = cv2.imread(reference) #cargar la imagen de referencia con los digitos en Arial Black
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) #transforma el color de la foto a escala de grises
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY)[1] #hago thresholding para segmentar la imagen en lo que me interesa
cv2.imwrite('ref.jpg',ref)

refConts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #guardo el contorno de los digitos en la imagen de referencia para hacer analisis luego
#RETR_EXTERNAL metodo que devuelve solo los contornos de los extremos exteriores
#CHAIN_APPROX_SIMPLE metodo que devuelve solo los endpoints necesarios para dibujar el contorno

refConts = imutils.grab_contours(refConts) #recorre el contorno y devuelve el num de pixeles
refConts = contours.sort_contours(refConts, method="left-to-right")[0] #ordena los contornos
digits = {}

# itero en cada digito de la imagen de referencia
for (i, c) in enumerate(refConts): #i es el digito, c es el contorno
	# computar la bounding box para el digito, extraerlo y redimensionarlo a un tamaño fijo
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w] #roi es region of interest
	roi = cv2.resize(roi, (57, 88))
	# actualizar el diccionario de digitos, mapeando el nombre del digito a la region de interes
	digits[i] = roi

# para estructurar el kernel, inicializar un rectangulo de mayor ancho que alto, y un cuadrado
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# subir la input image, redimensionarla y aplicarle un grayscale
image = cv2.imread(ejemplo) #ARREGLAR CUANDO ENTIENDA LA CELDA 13: ESTO TIENE QUE SER args["image"]
image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, rectKernel)
cv2.imwrite('close.png', close)
#remueve pixeles de fondo que matchean la estructura del elemento. Sirve para cerrar huecos en la imagen.

# calculo el gradiente de Scharr en la imagen con la op. morfologica, y escalo lo demas al rango [0, 255]
gradX = cv2.Sobel(close, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8") #porque uint8 tiene un rango de [0, 255]
cv2.imwrite('GradX1.png', gradX)

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# aplico otra closing operation en la imagen binaria para ayudar a cerrar huecos en la imagen
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv2.imwrite('thresh.png', thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #busco los contornos de los digitos
cnts = imutils.grab_contours(cnts) #los guardo en una lista
locs = [] #inicializar la lista con la ubicacion de cada digito

for (i, c) in enumerate(cnts): #itero a traves de los contornos
    (x, y, w, h) = cv2.boundingRect(c) #calculo el bounding box de los contornos
    ar = w / float(h) #ar es el aspect ratio
    if ar > 2.5 and ar < 4.0: #esto implicaria que la bb contiene una imagen mas ancha que alta
            if (w > 135 and w < 145) and  (h > 45 and h < 55):
                locs.append((x, y, w, h))

locs = sorted(locs, key=lambda x:x[0]) #ordeno los digitos de izq a der
output = [] #inicializo la lista de digitos clasificados

groupOutput = [] #inicizalizo una lista vacia para los outputs de mi grupo de digitos
group = gray[y:y + h, x:x + w]
group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #encuentro los contornos del grupo de digitos
digitCnts = imutils.grab_contours(digitCnts)
digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0] #ordeno los contornos de izquierda a derecha

for c in digitCnts:
    (x, y, w, h) = cv2.boundingRect(c) #armar la bounding box del digito individual
    roi = group[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88)) #ajusto el tamaño para que matchee el tamaño de los digitos de referencia
    scores = []
    for (digit, digitROI) in digits.items(): #aplico un correlation-based matching template para ver con que digito de la referencia matchea cada digito de la input image
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result) 
                scores.append(score)
    groupOutput.append(str(np.argmax(scores)))# me voy a quedar con el digito que tenga la mayor puntuacion


# para dibujar la clasificacion de cada digito:
cv2.rectangle(thresh, (x - 5, y - 5),
	(x + w + 5, y + h + 5), (0, 0, 255), 2)
cv2.putText(thresh, "".join(groupOutput), (x, y - 15),
cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
	# actualizamos la lista con el output de los numeros predecidos
output.extend(groupOutput)
