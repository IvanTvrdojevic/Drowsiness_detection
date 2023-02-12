import sys
import argparse
import cv2

from classes import eye_classes, mouth_classes
import classification as cf
import feature_recognition as fr

from params import reye_rect_color, leye_rect_color, mouth_rect_color
from params import tf_num_of_epochs


# dohvati parametre sa komandne linije
parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train", action = "store_true", help = "treniraj model")
parser.add_argument("--test", action = "store_true", help = "testiraj model")
parser.add_argument("--broj_epoha", help = "broj epoha treniranja")
args = parser.parse_args()
config = vars(args)

train = config["train"]
test = config["test"]
broj_epoha = config["broj_epoha"]
if broj_epoha is not None:
    tf_num_of_epochs = broj_epoha


# kreiraj konvolucijsku mrezu
convnet = cf.get_convnet()
# kreiraj modele
model_eyes = cf.get_model(convnet, 'eyes', eye_classes, train, test)
model_mouth = cf.get_model(convnet, 'mouth', mouth_classes, train, test)

# kaskadni model za lice
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# ostali kaskadni modeli za lice trenutno ne rade (model ne prepoznaje nista!!!)
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')

# ulazni stream (za ulaz s video kamere staviti 0 za parametar)
stream = cv2.VideoCapture('archive/Video/video.mp4')

# glavna petlja koja procesira stream
while True:
    # procitaj frame
    (ret, frame) = stream.read()
    # konvertiraj u greyscale (za lakse prepoznavanje - model se trenira takdjer na sivim slikama)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # detektiraj pravokutnik lica i nacrtaj ga na frameu
    face_rect = fr.detect(gray, face_cascade)   
    fr.draw_rects(frame, face_rect, fr.face_rect_color)

    # ocekujemo tocno jedno lice u slici, preskoci detekciju ako to nije slucaj
    if len(face_rect) != 1:
        continue

    # provjeri oci
    fr.check_eye(model_eyes, frame, gray, face_rect, 0, reye_rect_color)
    fr.check_eye(model_eyes, frame, gray, face_rect, 1, leye_rect_color)

    # provjeri usta
    fr.check_mouth(model_mouth, frame, gray, face_rect, mouth_rect_color)
    
    # prikazi frame
    cv2.imshow("frame", frame)
    
    # tipka za prekid glavne petlje
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):   
        break      
    
cv2.destroyAllWindows()