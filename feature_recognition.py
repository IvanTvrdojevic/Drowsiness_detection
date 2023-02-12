import numpy as np
import cv2
from tqdm import tqdm
import tflearn
from matplotlib import pyplot as plt
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import os

from params import tf_img_size, tf_learning_rate
from params import dms_scaleFactor, dms_minNeighbors, dms_minSize, dms_flags
from params import rect_stroke, rect_enable, face_rect_color
from params import lbl_font, lbl_fontScale, lbl_fontColor, lbl_thickness, lbl_lineType
from params import DBG_OUT

# funkcija za detekciju koristeci haar kaskade
# Ulazni parametri:
#   frame: ulazni frame/slika
#   cascade: haar kaskada
# Izlaz:
#   rects: pravokutnici od prepoznatih featura na frameu
def detect(frame, cascade):
    rects = cascade.detectMultiScale(frame, 
                                     scaleFactor    = dms_scaleFactor, 
                                     minNeighbors   = dms_minNeighbors, 
                                     minSize        = dms_minSize,
                                     flags          = dms_flags)
                                     
    if len(rects) == 0:
        return []

    return rects

# funkcija za crtanje pravokutnika
# Ulazni parametri:
#   frame: ulazni frame/slika
#   rects: pravokutnici od prepoznatih featura na frameu
#   color: boja obruba pravokutnika
# Izlaz:
def draw_rects(frame, rects, color):
    if rect_enable != 1:
        return

    for x, y, w, h in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, rect_stroke)

# funkcija za pisanje na sliku
# Ulazni parametri:
#   frame: ulazni frame/slika
#   label: text/oznaka koja se pise na sliku
#   x,y: pozicija texta na slici
# Izlaz:
def write_label(frame, label, x, y):

    cv2.putText(frame, label, (x, y), 
                lbl_font, 
                lbl_fontScale,
                lbl_fontColor,
                lbl_thickness,
                lbl_lineType)

# funkcija za detekciju i prepoznavanje stanja oka
# Ulazni parametri:
#   trenirani model za prepoznavanje stanja
#   ulaznu sliku
#   ulaznu sliku u gray formatu
#   pravokutnik lica
#   naznaku dali se radi o lijevo oku - 1 za da, 0 za ne (desno oko)
# Izlaz:
#   -1 - detekcija neuspjesna (spanje oka se nezna)
#   0 - zatvoreno
#   1 - otvoreno
def check_eye(model, frame, gray, rect, is_left, eye_rect_color = face_rect_color):
    # nije prosljedjen pravokutnik lica
    if len(rect) == 0:
        write_label(frame, 'Lice nije detektirano', 10, frame.shape[0] - 10)
        return -1

    # kaskadni model za oci - svako posebno, preciznije je prepoznavanje
    if is_left == 1:
        eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    else:
        eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

    for (x, y, w, h) in rect:
        # postavi podrucje za detekciju
        half = int(w/2)
        eye_frame = frame[y:y + h, x + is_left*half:x + (is_left + 1)*half] 
        eye_grey = gray[y:y + h, x + is_left*half:x + (is_left + 1)*half] 
        # detektiraj oko
        eye_rect = detect(eye_grey, eye_cascade)

        # nije pronadjen pravokutnik oka
        if len(eye_rect) == 0:
            write_label(eye_frame, 'Oko nije detektirano', 0, eye_frame.shape[0])
            return -1   

        # pronadjeno vise od jednog pravokutnika oka
        # (napomena: slucaj koji se nebi trebao dogodit, ali ako se dogodi nebi bilo lose da se
        # odabere npr najveci pravokutnik - za buducu implentaciju)
        if len(eye_rect) > 1:
            write_label(eye_frame, 'Vise ociju detektirano', 0, eye_frame.shape[0])
            return -1   

        draw_rects(eye_frame, eye_rect, eye_rect_color)  
        for (xe, ye, we, he) in eye_rect:
            detected_eye_grey = eye_grey[ye:ye + he, xe:xe + we] 
            # detektiraj stanje oka
            detected_eye_grey = cv2.resize(detected_eye_grey, (tf_img_size, tf_img_size))
            data = np.array(detected_eye_grey).reshape(tf_img_size, tf_img_size,1)
            model_out = model.predict([data])[0]

            # zapisi prepoznato stanje na sliku
            if np.argmax(model_out) == 1:
                write_label(eye_frame, 'Otvoreno', 0, he)
            else:
                write_label(eye_frame, 'Zatvoreno', 0, he)

            if DBG_OUT == 1:
                cv2.imshow("eye_grey" + str(is_left), detected_eye_grey)


# pomocna funkcija za detekciju i prepoznavanje stanja usta
# Ulazni parametri:
#   trenirani model za prepoznavanje stanja
#   ulaznu sliku
#   ulaznu sliku u gray formatu
#   pravokutnik lica
#   naznaku dali se radi o lijevo oku - 1 za da, 0 za ne (desno oko)
# Izlaz:
#   -1 - detekcija neuspjesna (stanje usta se nezna)
#   0 - zatvoreno
#   1 - otvoreno
def check_mouth(model, frame, gray, rect, mouth_rect_color = face_rect_color):
 # nije prosljedjen pravokutnik lica
    if len(rect) == 0:
        write_label(frame, 'Lice nije detektirano', 10, frame.shape[0] - 10)
        return -1

    # kaskadni model za usta
    mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

    for (x, y, w, h) in rect:
        # postavi podrucje za detekciju
        half = int(h/2)
        mouth_frame = frame[y + half:y + h, x:x + w] 
        mouth_grey = gray[y + half:y + h, x:x + w] 
        # detektiraj oko
        mouth_rect = detect(mouth_grey, mouth_cascade)

        # nije pronadjen pravokutnik oka
        if len(mouth_rect) == 0:
            write_label(mouth_frame, 'Usta nisu detektirana', 0, mouth_frame.shape[0])
            return -1   

        # pronadjeno vise od jednog pravokutnika usta
        # (napomena: slucaj koji se nebi trebao dogodit, ali ako se dogodi nebi bilo lose da se
        # odabere npr najveci pravokutnik - za buducu implentaciju)
        if len(mouth_rect) > 1:
            write_label(mouth_frame, 'Vise usta detektirano', 0, mouth_frame.shape[0])
            return -1   

        draw_rects(mouth_frame, mouth_rect, mouth_rect_color)  
        for (xe, ye, we, he) in mouth_rect:
            detected_mouth_grey = mouth_grey[ye:ye + he, xe:xe + we] 
            # detektiraj stanje usta
            detected_mouth_grey = cv2.resize(detected_mouth_grey, (tf_img_size, tf_img_size))
            data = np.array(detected_mouth_grey).reshape(tf_img_size, tf_img_size,1)
            model_out = model.predict([data])[0]


            # zapisi prepoznato stanje na sliku
            if np.argmax(model_out) == 1:
                write_label(mouth_frame, 'Ne Zijeva', 0, he)
            else:
                write_label(mouth_frame, 'Zijeva', 0, he)

            if DBG_OUT == 1:
                cv2.imshow("mouth_grey", detected_mouth_grey)                

