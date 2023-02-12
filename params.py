import cv2


#################### konstante i parametri ####################
#### parametri modela za ucenje ####
# velicina slike
tf_img_size = 50
# Learning rate, korišten pri učenju mreže. Jako dobar learning rate za ADAM
tf_learning_rate = 3e-4
# broj epoha
tf_num_of_epochs = 10
# postotak za validaciju
tf_validation_set = (0.3)
# snapshot korak
tf_snapshot_step = 500
# optimizer
tf_optimizer = 'adam'
# aktivacijska funkcija za unutarnje slojeve
tf_conv_2d_act = 'relu'
# aktivacijska funkcija za vanjske slojeve
tf_full_act = 'sigmoid'


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
# plot postavke
plot_num_x = 5
plot_num_y = 5


#### ostalo ####
DBG_OUT = 1