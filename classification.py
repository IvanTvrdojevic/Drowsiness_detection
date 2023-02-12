import numpy as np
import cv2
from tqdm import tqdm
import tflearn
from tensorflow import keras
from matplotlib import pyplot as plt
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import os

from params import tf_img_size, tf_learning_rate
from params import plot_num_x, plot_num_y
from params import tf_img_size, tf_num_of_epochs, tf_validation_set, tf_snapshot_step, tf_optimizer, tf_conv_2d_act, tf_full_act


# Ucitaj slike iz zadanog direktorija i priprema istih za ulaz u model
# Ulazni parametri:
#   dir: direktorij sa slikama
#   size: potrebna velicina slike za ulaz u model
# Izlaz:
#   4d array (broj slika * size * size * 1) 
def load_images(dir, size = tf_img_size):
    images = []
    for filename in tqdm(os.listdir(dir)): 
        # Putanja do slike
        path = os.path.join(dir, filename)
        # Učitavanje slike kao grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Mijenjanje velicine slike na traženu veličinu
            img = cv2.resize(img, (size, size))
            # Takodjer, za potrebe tensor flowa, moramo 2D grey scale
            # image array trenutne slike reshapeat u 3D
            img = img.reshape(size, size, 1)
            # Dodavanje slike 
            images += [img]

    return images


# Stvara podatke za treniranje/testiranje na temelju opisa
# klasa za prepoznavanje i pripadajucih slika za neki objekt (npr oci ili usta)
# Drugim rjecima, pridodaje slikama odgovarajuce oznake
# Ulazni parametri:
#   classes: opis recognition clasa sa pripadajucim slikama(tj direktorijam do istih)
#   getTestData: po defaultu false, znaci da dohvacamo trening datu, ako je true onda test datu
# Izlaz:
#   images: array sa slikama
#   labels: array sa oznakama za svaku sliku
def create_data(classes, getTestData = False):
    images = []
    labels = []

    # pridruzi pripadajuci index za direktorije 
    dirIdx = 1
    if getTestData:
        dirIdx = 2

    # za svaku recognition klasu
    for idx, cls in enumerate(classes):
        # napravi label za tu klasu
        # label je array velicine broj_klasa koji na indexu
        # trenutne klase ima 1, drugdje 0
        # npr za ukupno 4 klase i klasu 3, oznaka ce bit [0, 0, 1, 0]
        curr_lbl = [0]*len(classes)
        curr_lbl[idx] = 1

        # za svaki direktorij trenutne klase, ucitaj slike i dodaj oznake
        for dir in cls[dirIdx]:
            # ucitaj slike
            curr_img = load_images(dir)
            # dodaj ucitane slike u izlazni niz
            images += curr_img
            # dodaj oznake za ucitane slike u izlazni niz
            labels += [curr_lbl]*len(curr_img)
        
    return images, labels


# Stvara convolucijsku mrezu
# Ulazni parametri:
# Izlaz:
#   convnet: konvolucijska mreza
def get_convnet():
    convnet = input_data(shape = [None, tf_img_size, tf_img_size, 1], name = 'input')
    convnet = conv_2d(convnet, 32, 5,   activation = tf_conv_2d_act)
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 64, 5,   activation = tf_conv_2d_act)
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 128, 5,  activation = tf_conv_2d_act)
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 256, 5,  activation = tf_conv_2d_act)
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 128, 5,  activation = tf_conv_2d_act)
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 64, 5,   activation = tf_conv_2d_act)
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 32, 5,   activation = tf_conv_2d_act)
    convnet = max_pool_2d(convnet, 2)
    convnet = fully_connected(convnet, 1024, activation = tf_full_act)
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, 2, activation = tf_full_act)

    convnet = regression(convnet,   
                         optimizer = tf_optimizer, 
                         learning_rate = tf_learning_rate, 
                         loss = 'categorical_crossentropy', 
                         name = 'output')

    return convnet


# Treniranje modela
# Ulazni parametri:
#   model: model za treniranje
#   curr_object: ime objekta za koji se trenira
#   curr_classes: popis klasa i pripadajucih slika
# Izlaz:
#   datoteka sa imenom curr_object + '.model.tflearn' sadrzi trenirani model
def train_model(model, curr_object, curr_classes):
    # kreiraj data za treniranje
    (train_img, train_lbl) = create_data(curr_classes)

    # napravi trening i spremi rezultate
    model.fit({'input': train_img}, {'output': train_lbl}, 
                n_epoch         = tf_num_of_epochs,
                validation_set  = tf_validation_set,
                snapshot_step   = tf_snapshot_step, 
                show_metric     = True, 
                run_id          = curr_object + '.model')
    model.save('models/' + curr_object + '/model.tflearn')


# Testiranje modela
# Ulazni parametri:
#   model: model za testiranje
#   curr_object: ime objekta za koji se testira
#   curr_classes: popis klasa i pripadajucih slika
# Izlaz:
#   prikazuje se plot sa rezultatim i ispisuje statistika
def test_model(model, curr_object, curr_classes):
    # kreiraj data za testiranje
    (test_img, test_lbl) = create_data(curr_classes, True)

    # kreiraj plot
    plot_num_total = plot_num_x*plot_num_y
    fig = plt.figure()

    # prebrojat cemo koliko smo promasili
    miss = 0
    all = len(test_img)

    # uzimamo samo onoliko slika za plot koliko ih stane 
    # (tj prema zadanim parametrima velicine plota)
    # Stoga, prvo kreiramo array sa indexima svih slika,
    # zatim ih random preslozimo, i uzimamo samo prvih plot_num_total
    idxs = np.arange(all)
    np.random.shuffle(idxs)

    # testiramo sve slike, na plot stavljamo samo prvih plot_num_total
    for idx in range(0, all): 
        # pripremi sliku i pripadajucu oznaku
        img_idx = idxs[idx]
        img = test_img[img_idx]
        label = test_lbl[img_idx]
        real_class = np.argmax(label)

        # napravi predikciju
        predictions  = model.predict([img])
        predicted_class = np.argmax(predictions[0])

        # povecaj broj netocnih
        if predicted_class != real_class:
            miss += 1

        # nastavi petlju, tj nemoj prikazat na plotu, 
        # ako smo vec prosli prvih plot_num_total
        if idx >= plot_num_total:
            continue

        # prikazi na plotu
        y = fig.add_subplot(plot_num_x, plot_num_y, idx + 1)
        y.imshow(img, cmap = 'gray')
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

        str_label = curr_classes[predicted_class][0]
        title_obj = plt.title(str_label)

        if predicted_class != real_class:
            plt.setp(title_obj, color = 'r')

    # prikazi statistiku i plot    
    print('-------' + curr_object + '-------')
    print('Ukupno:', all)
    print('Promaseno:', miss)
    print('Postotak tocnosti:', (all - miss)/all*100)

    plt.show()


# Stvaranje modela - treniranje/testiranje/ucitavanje, ovisno o parametrima
# Ulazni parametri:
#   convnet: konvolucijska mreza
#   curr_object: ime objekta za koji se stvara model
#   curr_classes: popis klasa i pripadajucih slika
#   train: True ako se model trenira
#   test: True ako se model testira
# Izlaz:
#   model za prepoznavanje
def get_model(convnet, curr_object, curr_classes, train = False, test = False):
    # kreiraj model
    model = tflearn.DNN(convnet)

    # treniraj po potrebi
    if train:
        train_model(model, curr_object, curr_classes)
    else:
        model.load('models/' + curr_object + '/model.tflearn')

    # testiraj po potrebi
    if test:
        test_model(model, curr_object, curr_classes)

    return model
