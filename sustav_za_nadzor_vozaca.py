import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import models, layers, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet
from keras.applications.resnet import preprocess_input
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from tqdm import tqdm, notebook
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import os
import random
import pickle


#################### konstante i parametri ####################
#### parametri modela za ucenje ####
# velicina slike
IMG_SIZE = 50
# Learning rate, korišten pri učenju mreže. Jako dobar learning rate za ADAM
LR = 3e-4

#### parametri kaskadnog modela ####
dms_scaleFactor = 1.3
dms_minNeighbors = 4
dms_minSize = (30, 30)
dms_flags = cv2.CASCADE_SCALE_IMAGE

#### prikaz ####
# prikazi pravokutnike
rect_enable = 1
# debljina crte pravokutnika
rect_stroke = 3
# boja pravokutnika 
face_rect_color = (255, 0, 0)
leye_rect_color = (0, 255, 0)
reye_rect_color = (0, 0, 255)
mouth_rect_color= (0, 255, 255)

# font labela
lbl_font        = cv2.FONT_HERSHEY_SIMPLEX
lbl_fontScale   = 1
lbl_fontColor   = (255,255,255)
lbl_thickness   = 1
lbl_lineType    = 2

#### ostalo ####
DBG_OUT = 1


#################### pomocne funkcije ####################
# pomocna funkcija za pripremu slike
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Učita sliku i transformira ju u grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # Mijenja veličinu slike 
    return new_array.reshape(IMG_SIZE, IMG_SIZE, 1)

# pomocna funkcija za detekciju
def detect(frame, cascade):
    rects = cascade.detectMultiScale(frame, 
                                     scaleFactor=dms_scaleFactor, 
                                     minNeighbors=dms_minNeighbors, 
                                     minSize=dms_minSize,
                                     flags=dms_flags)
                                     
    if len(rects) == 0:
        return []

    return rects

# pomocna funkcija za crtanje pravokutnika
def draw_rects(frame, rects, color):
    if rect_enable != 1:
        return

    for x, y, w, h in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, rect_stroke)

# pomocna funkcija za pisanje na sliku
def write_label(frame, label, x, y):

    cv2.putText(frame, label, (x, y), 
                lbl_font, 
                lbl_fontScale,
                lbl_fontColor,
                lbl_thickness,
                lbl_lineType)

# pomocna funkcija za detekciju i prepoznavanje stanja oka
# prima:
#   trenirani model za prepoznavanje stanja
#   ulaznu sliku
#   ulaznu sliku u gray formatu
#   pravokutnik lica
#   naznaku dali se radi o lijevo oku - 1 za da, 0 za ne (desno oko)
# vraca slijedece:
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
            detected_eye_grey = cv2.resize(detected_eye_grey, (IMG_SIZE, IMG_SIZE))
            data = np.array(detected_eye_grey).reshape(IMG_SIZE,IMG_SIZE,1)
            model_out = model.predict([data])[0]

            # zapisi prepoznato stanje na sliku
            if np.argmax(model_out) == 1:
                write_label(eye_frame, 'Otvoreno', 0, he)
            else:
                write_label(eye_frame, 'Zatvoreno', 0, he)

            if DBG_OUT == 1:
                cv2.imshow("eye_grey" + str(is_left), detected_eye_grey)


# pomocna funkcija za detekciju i prepoznavanje stanja usta
# prima:
#   trenirani model za prepoznavanje stanja
#   ulaznu sliku
#   ulaznu sliku u gray formatu
#   pravokutnik lica
#   naznaku dali se radi o lijevo oku - 1 za da, 0 za ne (desno oko)
# vraca slijedece:
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
            detected_mouth_grey = cv2.resize(detected_mouth_grey, (IMG_SIZE, IMG_SIZE))
            data = np.array(detected_mouth_grey).reshape(IMG_SIZE,IMG_SIZE,1)
            model_out = model.predict([data])[0]


            # zapisi prepoznato stanje na sliku
            if np.argmax(model_out) == 1:
                write_label(mouth_frame, 'Ne Zijeva', 0, he)
            else:
                write_label(mouth_frame, 'Zijeva', 0, he)

            if DBG_OUT == 1:
                cv2.imshow("mouth_grey", detected_mouth_grey)                


###################################### MODEL ZA OCI ######################################
#################### TRENIRANJE MODELA ####################
######training data
##with faces
#train_dir_closed = r'archive/TrainingData/WithFaces/Models/Closed'
#train_dir_open = r'archive/TrainingData/WithFaces/Models/Open'
#train_dir_closed = r'archive/TrainingData/WithFaces/Custom/Karlo/Closed'
#train_dir_open = r'archive/TrainingData/WithFaces/Custom/Karlo/Open'

##eyesonly
train_dir_closed = r'archive/TrainingData/EyesOnly/Models/Closed'
train_dir_open = r'archive/TrainingData/EyesOnly/Models/Open'

######test data
##with faces
#test_dir_closed = r'archive/TestData/WithFaces/Models/Closed'
#test_dir_open = r'archive/TestData/WithFaces/Models/Open'
#test_dir_closed = r'archive/TestData/WithFaces/Custom/Karlo/Closed'
#test_dir_open = r'archive/TestData/WithFaces/Custom/Karlo/Open'

##eyesonly
#test_dir_closed = r'archive/TestData/EyesOnly/Models/Closed'
#test_dir_open = r'archive/TestData/EyesOnly/Models/Open'
test_dir_closed = r'archive/TestData/EyesOnly/Custom/Karlo/Closed'
test_dir_open = r'archive/TestData/EyesOnly/Custom/Karlo/Open'


#Zatvorene oci imaju label [1,0]
training_data = []
#tqdm koristimo za progress bar
for img in tqdm(os.listdir(train_dir_closed)): 
    label = [1,0]
    #Putanja do slike
    path = os.path.join(train_dir_closed, img)
    #Učitavanje slike kao grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        #Mijenjanje velicine slike na traženu veličinu
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        #Dodavanje slike i label-a u training_data (spajanje slike sa labelom)
        training_data.append([np.array(img), np.array(label)])

#Otvorene oci imaju label [0,1]
#tqdm koristimo za progress bar
for img in tqdm(os.listdir(train_dir_open)):
    label = [0,1]
    #Putanja do slike
    path = os.path.join(train_dir_open, img)
    #Učitavanje slike kao grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        #Mijenjanje velicine slike na traženu veličinu
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        #Dodavanje slike i label-a u training_data (spajanje slike sa labelom)
        training_data.append([np.array(img), np.array(label)])

#Zatvorene oci imaju label [1,0]
test_data = []
#tqdm koristimo za progress bar
for img in tqdm(os.listdir(test_dir_closed)):
    label = [1,0]
    #Putanja do slike
    path = os.path.join(test_dir_closed, img)
    #Učitavanje slike kao grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        #Mijenjanje velicine slike na traženu veličinu
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        #Dodavanje slike i label-a u training_data (spajanje slike sa labelom)
        test_data.append([np.array(img), np.array(label)])

#Otvorene oci imaju label [0,1]
#tqdm koristimo za progress bar
if test_dir_open != '':
    for img in tqdm(os.listdir(test_dir_open)):
        label = [0,1]
        #Putanja do slike
        path = os.path.join(test_dir_open, img)
        #Učitavanje slike kao grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            #Mijenjanje velicine slike na traženu veličinu
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            #Dodavanje slike i label-a u training_data (spajanje slike sa labelom)
            test_data.append([np.array(img), np.array(label)])

#Kreiranje neuronske mreže
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='ulaz')

#Konvolucija, 32 filtera, 5x5, ReLu aktivacija
convnet = conv_2d(convnet, 32, 5, activation='relu')
#Pooling size 2
convnet = max_pool_2d(convnet, 2)

#Konvolucija, 64 filtera, 5x5, ReLu aktivacija
convnet = conv_2d(convnet, 64, 5, activation='relu')
#Pooling size 2
convnet = max_pool_2d(convnet, 2)

#Konvolucija, 128 filtera, 5x5, ReLu aktivacija
convnet = conv_2d(convnet, 128, 5, activation='relu')
#Pooling size 2
convnet = max_pool_2d(convnet, 2)

#Konvolucija, 256 filtera, 5x5, ReLu aktivacija
convnet = conv_2d(convnet, 256, 5, activation='relu')
#Pooling size 2
convnet = max_pool_2d(convnet, 2)

#Konvolucija, 128 filtera, 5x5, ReLu aktivacija
convnet = conv_2d(convnet, 128, 5, activation='relu')
#Pooling size 2
convnet = max_pool_2d(convnet, 2)

#Konvolucija, 64 filtera, 5x5, ReLu aktivacija
convnet = conv_2d(convnet, 64, 5, activation='relu')
#Pooling size 2
convnet = max_pool_2d(convnet, 2)

#Konvolucija, 32 filtera, 5x5, ReLu aktivacija
convnet = conv_2d(convnet, 32, 5, activation='relu')
#Pooling size 2
convnet = max_pool_2d(convnet, 2)

#Input je 4d Tensor (convnet), 1024 outputa, aktivacija je ReLu
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, #Optimizer je adam (Adam optimization algorithm, adaptive moment estimation)
                     loss='categorical_crossentropy', name='izlaz')

model = tflearn.DNN(convnet, tensorboard_dir='log')

train = training_data
test = test_data

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'ulaz': X}, {'izlaz': y}, n_epoch=5,
          validation_set=(0.3),
          snapshot_step=500, show_metric=True, run_id = 'EyeDet.model')
model.save('ClosedEyes-CNN.model')



#################### TESTIRANJE MODELA ####################
fig=plt.figure()

test = random.shuffle(test_data)

miss = 0
all = 0
for num,data in enumerate(test_data[:25]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(5,5,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Open'
    else: str_label='Closed'

    title_obj = plt.title(str_label)

    if np.argmax(model_out) != img_num[1]:
        miss += 1
        plt.setp(title_obj, color='r')
        
    y.imshow(orig,cmap='gray')
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

    all += 1

print('-------Oci-------')
print('Promaseno:', miss)
print('Postotak tocnosti:',(all - miss)/all*100)

plt.show()



###################################### MODEL ZA ZIJEVANJE ######################################
#################### TRENIRANJE MODELA ####################
train_dir = r'archive/TrainingData/MouthOnly/Models'
test_dir = r'archive/TestData/MouthOnly/Models'

model_yawn = tflearn.DNN(convnet, tensorboard_dir='log')

def label_train_img(img):
    img = img.strip(".png")
    if int(img) <= 2024:
        return [1,0]  # Otvorena usta
    else : return [0,1] #Zatvorena usta

def label_test_img(img):
    img = img.strip(".png")
    if int(img) <= 504:
        return [1,0]  # Otvorena usta
    else : return [0,1] #Zatvorena usta

def create_training_data():
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_train_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    
    return training_data

train_data = create_training_data()

def create_testing_data():
    testing_data = []
    for img in tqdm(os.listdir(test_dir)):
        label = label_test_img(img)
        path = os.path.join(test_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), np.array(label)])
    
    return testing_data

test_data = create_testing_data()


train = train_data
test = test_data

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model_yawn.fit({'ulaz': X}, {'izlaz': y}, n_epoch=1,
          validation_set=({'ulaz': test_x}, {'izlaz': test_y}),
          snapshot_step=500, show_metric=True)
model_yawn.save('Yawn_Model-CNN.model')


#################### TESTIRANJE MODELA ####################
fig=plt.figure()

test = random.shuffle(test_data)

miss = 0
all = 0
for num,data in enumerate(test_data[:12]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model_yawn.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Ne Zijeva'
    else: str_label='Zijeva'

    if np.argmax(model_out) != img_num[1]:
        miss += 1
        plt.setp(title_obj, color='r')
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

    all += 1

print('-------Zijevanje-------')
print('Promaseno:', miss)
print('Postotak tocnosti:',(all - miss)/all*100)
plt.show()




#################### KORISTENJE MODELA ####################
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
    face_rect = detect(gray, face_cascade)   
    draw_rects(frame, face_rect, face_rect_color)

    # ocekujemo tocno jedno lice u slici, preskoci detekciju ako to nije slucaj
    if len(face_rect) != 1:
        continue

    # provjeri oci
    check_eye(model, frame, gray, face_rect, 0, reye_rect_color)
    check_eye(model, frame, gray, face_rect, 1, leye_rect_color)

    # provjeri usta
    check_mouth(model_yawn, frame, gray, face_rect, mouth_rect_color)
    
    # prikazi frame
    cv2.imshow("frame", frame)
    
    # tipka za prekid glavne petlje
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):   
        break      
    
cv2.destroyAllWindows()