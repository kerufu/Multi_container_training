import tensorflow as tf
from tensorflow import TensorSpec
from tensorflow.keras.regularizers import l2
import numpy as np
import dlib
import glob
import random
import ctypes
import os
import cv2

import pickle

import scipy.io


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))

os.system(
    'cc /home/keru/Keru/pico-master/rnt//picornt.c -O3 -fPIC -shared -o picornt.lib.so')
pico = ctypes.cdll.LoadLibrary('./picornt.lib.so')
os.system('rm picornt.lib.so')
bytes = open('/home/keru/Keru/pico-master/rnt/cascades/facefinder', 'rb').read()
#bytes = open('../../cascades/d', 'rb').read()
cascade = np.frombuffer(bytes, dtype=np.uint8)
slot = np.zeros(1, dtype=np.int32)
maxndets = 2048

landmark_model_path = "/home/keru/Keru/expression/shape_predictor_68_face_landmarks.dat"

training_dataset_length = 100
batch_size = 20
model_path = "/home/keru/Keru/expression/model2/mix_model"
# model_path="/mix_model"
epoch = 1

facedb_path = "/home/keru/Keru/expression/facesdb/*/*/*.bmp"

jaffe_path = "/home/keru/Keru/expression/jaffedbase/*.tiff"

officedb_path = "/home/keru/Keru/expression/officedb/*.MP4"

SoF_path = "/home/keru/Keru/expression/SoF/*.jpg"

ised_mat_path = '/home/keru/Keru/expression/ISED_database/ISED_details.mat'
ised_image_path = '/home/keru/Keru/expression/ISED_database/'

RAVDESS_path = "/home/keru/Keru/expression/RAVDESS/*/*/*.mp4"

BAUM1_label_path = "/home/keru/Keru/expression/BAUM1/*.xlsx"
BAUM1_data_path = "/home/keru/Keru/expression/BAUM1/"


std_width = 50
face_threshold = 200

# HOG
winSize = (std_width, std_width)
blockSize = (20, 20)
blockStride = (10, 10)
cellSize = (5, 5)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
# winStride = (8,8)
# padding = (8,8)
# locations = ((10,20),)

homography_transform = 150

bound = 20


landmark_predictor = dlib.shape_predictor(
    "/home/keru/Keru/expression/shape_predictor_68_face_landmarks.dat")
# landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

num_of_experssion_types = 3
enum_emotions = [
    "neutral",
    "happy",
    "angry",
    "sad",
    "other"
]

# class_weight=np.float32([1,1,1,1])
# class_weight=class_weight/np.mean(class_weight)


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def process_frame(gray, angle, scale_factor, stride_factor, minsize, maxsize):
    dets = np.zeros(4*maxndets, dtype=np.float32)
    ndets = pico.find_objects(
        ctypes.c_void_p(dets.ctypes.data), ctypes.c_int(maxndets),
        ctypes.c_void_p(cascade.ctypes.data), ctypes.c_float(angle),  # angle
        ctypes.c_void_p(gray.ctypes.data), ctypes.c_int(
            gray.shape[0]), ctypes.c_int(gray.shape[1]), ctypes.c_int(gray.shape[1]),
        ctypes.c_float(scale_factor), ctypes.c_float(
            stride_factor), ctypes.c_float(minsize), ctypes.c_float(maxsize)
    )
    ndets = pico.cluster_detections(
        ctypes.c_void_p(dets.ctypes.data), ctypes.c_int(ndets)
    )

    return list(dets.reshape(-1, 4))[0:ndets]


def distance_between(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def point_between(p1, p2):
    return np.array([abs(p1[0] - p2[0]) / 2.0 + min(p1[0], p2[0]), abs(p1[1] - p2[1]) / 2.0 + min(p1[1], p2[1])])


def cal_features(landmarks):
    # the shape of landmarks should be (68, 2)
    features = [0] * 89

    lmk = landmarks
    left_eye = point_between(lmk[36], lmk[39])
    right_eye = point_between(lmk[42], lmk[45])
    nose = point_between(lmk[30], lmk[33])
    between_eyes = distance_between(left_eye, right_eye)
    # between_eyes = distance_between(lmk[0], lmk[16])
    i = 0

    for x in range(0, 17):
        features[i] = distance_between(lmk[x], nose) / between_eyes
        i += 1
    for x in range(17, 22):
        features[i] = distance_between(lmk[x], left_eye) / between_eyes
        i += 1
    for x in range(22, 27):
        features[i] = distance_between(lmk[x], right_eye) / between_eyes
        i += 1
    for x in range(31, 36):
        features[i] = distance_between(lmk[x], nose) / between_eyes
        i += 1
    for x in range(36, 42):
        features[i] = distance_between(lmk[x], left_eye) / between_eyes
        i += 1
    for x in range(42, 48):
        features[i] = distance_between(lmk[x], right_eye) / between_eyes
        i += 1
    for x in range(48, 68):
        features[i] = distance_between(lmk[x], nose) / between_eyes
        i += 1
    for x in range(0, 5):
        features[i] = distance_between(lmk[17+x], lmk[26-x]) / between_eyes
        i += 1
    for x in range(17, 22):
        features[i] = distance_between(lmk[x], nose) / between_eyes
        i += 1
    for x in range(22, 27):
        features[i] = distance_between(lmk[x], nose) / between_eyes
        i += 1
    features[i] = distance_between(lmk[52], lmk[56]) / between_eyes
    i += 1
    features[i] = distance_between(lmk[51], lmk[57]) / between_eyes
    i += 1
    features[i] = distance_between(lmk[50], lmk[58]) / between_eyes
    i += 1
    features[i] = distance_between(lmk[48], lmk[54]) / between_eyes
    i += 1
    features[i] = distance_between(lmk[49], lmk[53]) / between_eyes
    i += 1
    features[i] = distance_between(lmk[59], lmk[55]) / between_eyes
    i += 1
    features[i] = distance_between(lmk[60], lmk[64]) / between_eyes
    i += 1
    features[i] = distance_between(lmk[63], lmk[65]) / between_eyes
    i += 1
    features[i] = distance_between(lmk[62], lmk[66]) / between_eyes
    i += 1
    features[i] = distance_between(lmk[61], lmk[67]) / between_eyes

    return features


class CNN_model(tf.keras.layers.Layer):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.model = [
            tf.keras.layers.Reshape((std_width, std_width, 1)),
            tf.keras.layers.Conv2D(
                8, 3, activation='relu', kernel_regularizer=l2(0.00001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(
                16, 3, activation='relu', kernel_regularizer=l2(0.00001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(
                32, 3, activation='relu', kernel_regularizer=l2(0.00001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_regularizer=l2(0.00001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu',
                                  kernel_regularizer=l2(0.00001))
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class HOG_model(tf.keras.layers.Layer):

    def __init__(self):
        super(HOG_model, self).__init__()
        self.model = [
            # tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(
                16, activation='relu', kernel_regularizer=l2(0.00000)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.0),
            tf.keras.layers.Dense(
                4, activation='relu', kernel_regularizer=l2(0.00000))
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class Landmark_model(tf.keras.layers.Layer):

    def __init__(self):
        super(Landmark_model, self).__init__()
        self.model = [
            # tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(
                32, activation='relu', kernel_regularizer=l2(0.00000)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(
                8, activation='relu', kernel_regularizer=l2(0.00000))
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class mymodel(tf.keras.Model):
    def __init__(self):
        super(mymodel, self).__init__()
        # self.CNN_module=CNN_model()
        self.HOG_module = HOG_model()
        self.Landmark_module = Landmark_model()
        self.neutral_output_module = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.0),
            tf.keras.layers.Dense(16, activation='relu',
                                  kernel_regularizer=l2(0.00000)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='sigmoid',
                                  kernel_regularizer=l2(0.00000))
        ]
        self.smile_output_module = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.0),
            tf.keras.layers.Dense(16, activation='relu',
                                  kernel_regularizer=l2(0.00000)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='sigmoid',
                                  kernel_regularizer=l2(0.00000))
        ]

    def call(self, inputs):
        face_image, HOG, landmark = inputs
        # feature_CNN=self.CNN_module(face_image)
        feature_HOG = self.HOG_module(HOG)
        feature_landmark = self.Landmark_module(landmark)
        # neutral_feature=tf.keras.layers.concatenate([feature_CNN,feature_HOG,feature_landmark])
        # smile_feature=tf.keras.layers.concatenate([feature_CNN,feature_HOG,feature_landmark])
        neutral_feature = tf.keras.layers.concatenate(
            [feature_HOG, feature_landmark])
        smile_feature = tf.keras.layers.concatenate(
            [feature_HOG, feature_landmark])

        for layer in self.neutral_output_module:
            neutral_feature = layer(neutral_feature)

        for layer in self.smile_output_module:
            smile_feature = layer(smile_feature)

        return [neutral_feature, smile_feature]


def detect_face(img_path):
    gray = cv2.imread(img_path, 0)
    face = process_frame(gray, 0, 1.01, 0.05, 100, 10000)
    face = sorted(face, key=lambda y: y[3], reverse=True)[0]
    face = np.array([face[1]-face[2]/2, face[0]-face[2]/2, face[2], face[2]])
    face = face.astype(np.int32)
    resized_face = cv2.resize(gray[face[1]:face[1]+face[3], face[0]:face[0]+face[2]],
                              dsize=(std_width, std_width), interpolation=cv2.INTER_CUBIC)
    # resized_face = cv2.equalizeHist(resized_face)
    resized_face = clahe.apply(resized_face)
    cv2.imshow("", resized_face)
    cv2.waitKey(1)
    return resized_face


def read_image(txt, emotion_set, nuetral_set):

    with open(txt, "r") as f:
        for path in f:
            # pos = path.split("emotion")[0].split("Emotion")[1][:-1]
            # emotion_set.append(detect_face("Images" + pos + ".png"))
            # middle=int(pos[-2:])//2
            # end=int(pos[-2:])
            # for i in range(middle, end):
            #     emotion_set.append(detect_face("Images" +pos[:-2]+"{:02d}".format(i)+".png"))
            # neg = path.split("emotion")[0].split("Emotion")[1][:-3]
            # nuetral_set.append(detect_face("Images" + neg + "01.png"))
            # nuetral_set.append(detect_face("Images" + neg + "02.png"))
            # nuetral_set.append(detect_face("Images" + neg + "03.png"))
            # nuetral_set.append(detect_face("Images" + neg + "04.png"))
            folder_path = path.split("emotion")[0].split("Emotion")[1][:-3]
            end = int(path.split("emotion")[0].split("Emotion")[1][-3:-1])
            middle = end//3

            for i in range(1, middle):
                nuetral_set.append(detect_face(
                    "Images" + folder_path+"{:02d}".format(i)+".png"))
            for i in range(middle*2, end):
                emotion_set.append(detect_face(
                    "Images" + folder_path+"{:02d}".format(i)+".png"))

    f.close()


def read_other_neutral_image(nuetral_set):

    txt = "other.txt"
    with open(txt, "r") as f:
        for path in f:
            folder_path = path.split("emotion")[0].split("Emotion")[1][:-3]
            end = int(path.split("emotion")[0].split("Emotion")[1][-3:-1])
            middle = end//3
            for i in range(1, middle):
                nuetral_set.append(detect_face(
                    "Images" + folder_path+"{:02d}".format(i)+".png"))


def read_other_negative_image(negative_set):
    txt = "other.txt"
    with open(txt, "r") as f:
        for path in f:
            folder_path = path.split("emotion")[0].split("Emotion")[1][:-3]
            end = int(path.split("emotion")[0].split("Emotion")[1][-3:-1])
            middle = end//3
            for i in range(middle, end):
                negative_set.append(detect_face(
                    "Images" + folder_path+"{:02d}".format(i)+".png"))


def prepare_facesdb(nuetral_set, positive_set, negative_set):
    file_paths = glob.glob(facedb_path)
    for _, file_path in enumerate(file_paths):
        label = int(file_path.split("_img")[0].split("-")[1])
        if label == 2:
            label = 3
        elif label == 4:
            label = 2
        face = detect_face(file_path)
        # cv2.imshow(" ", face)
        # cv2.waitKey(1)
        if label == 0:
            nuetral_set.append(face)
        elif label == 1:
            positive_set.append(face)
        else:
            negative_set.append(face)


def prepare_jaffe(nuetral_set, positive_set, negative_set):
    file_paths = glob.glob(jaffe_path)
    for _, file_path in enumerate(file_paths):
        emotion = file_path[41:43]
        label = 2
        if emotion == "NE":
            label = 0
        elif emotion == "HA":
            label = 1

        face = detect_face(file_path)
        if label == 0:
            nuetral_set.append(face)
        elif label == 1:
            positive_set.append(face)
        else:
            negative_set.append(face)


def prepare_SoF(nuetral_set, happy_set, sad_set):
    file_paths = glob.glob(SoF_path)
    for _, file_path in enumerate(file_paths):
        emotion = file_path[55:57]
        label = -1
        if emotion == "no":
            label = 0
        elif emotion == "hp":
            label = 1
        elif emotion == "sd":
            label = 3.

        else:
            label = 3
        gray = cv2.imread(file_path, 0)
        face = process_frame(gray, 0, 1.01, 0.05, 100, 10000)
        if len(face) > 0:
            face = sorted(face, key=lambda y: y[3], reverse=True)[0]
            if face[3] > face_threshold:
                face = np.array([face[1]-face[2]/2, face[0] -
                                face[2]/2, face[2], face[2]])
                face = face.astype(np.int32)
                resized_face = cv2.resize(gray[face[1]:face[1]+face[3], face[0]:face[0]+face[2]], dsize=(
                    std_width, std_width), interpolation=cv2.INTER_CUBIC)
                # resized_face = cv2.equalizeHist(resized_face)
                resized_face = clahe.apply(resized_face)
                cv2.imshow("", resized_face)
                cv2.waitKey(1)
                if label == 0:
                    nuetral_set.append(resized_face)
                elif label == 1:
                    happy_set.append(resized_face)
                elif label == 3:
                    sad_set.append(resized_face)


def prepare_ised(positive_set, negative_set):
    mat = scipy.io.loadmat(ised_mat_path)
    mat = mat['ISED_details']

    for m in mat:
        img_path = ised_image_path+m[5][0]
        label = m[6][0][0]
        gray = cv2.imread(img_path, 0)

        w, h = gray.shape
        gray = cv2.resize(gray, (h//2, w//2), interpolation=cv2.INTER_CUBIC)

        face = process_frame(gray, 0, 1.01, 0.05, 100, 10000)
        if len(face) > 0:
            face = sorted(face, key=lambda y: y[3], reverse=True)[0]
            if face[3] > face_threshold:
                face = np.array([face[1]-face[2]/2, face[0] -
                                face[2]/2, face[2], face[2]])
                face = face.astype(np.int32)
                resized_face = cv2.resize(gray[face[1]:face[1]+face[3], face[0]:face[0]+face[2]], dsize=(
                    std_width, std_width), interpolation=cv2.INTER_CUBIC)
                # resized_face = cv2.equalizeHist(resized_face)
                resized_face = clahe.apply(resized_face)
                cv2.imshow("", resized_face)
                cv2.waitKey(1)
                if label == 1:
                    positive_set.append(resized_face)
                else:
                    negative_set.append(resized_face)


def prepare_BAUM1(nuetral_set, positive_set, negative_set):
    xlsx_paths = glob.glob(BAUM1_label_path)
    for x in xlsx_paths:
        xtype = x[-6]
        wb = openpyxl.load_workbook(x)
        sheet = wb.get_sheet_by_name("Sheet1")
        for i in sheet["E"]:
            emotion = i.value
            # if emotion=="Neutral" or emotion=="Happiness" or emotion=="Anger" or emotion=="Sadness":
            try:
                file_name = sheet["D"][i.row-1].value
                vedio_file_path = BAUM1_data_path+xtype + \
                    "/s"+file_name[1:4]+"/"+file_name+".mp4"
                cap = cv2.VideoCapture(vedio_file_path)
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    if random.random() < 0.2:  # around 20000 random frames in total
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        w, h = gray.shape

                        # gray = cv2.resize(gray, (h//2, w//2), interpolation=cv2.INTER_CUBIC)

                        face = process_frame(gray, 0, 1.01, 0.05, 100, 10000)
                        if len(face) > 0:
                            face = sorted(
                                face, key=lambda y: y[3], reverse=True)[0]
                            if face[3] > face_threshold:
                                face = np.array(
                                    [face[1]-face[2]/2, face[0]-face[2]/2, face[2], face[2]])
                                face = face.astype(np.int32)
                                resized_face = cv2.resize(gray[face[1]:face[1]+face[3], face[0]:face[0]+face[2]], dsize=(
                                    std_width, std_width), interpolation=cv2.INTER_CUBIC)
                                # resized_face = cv2.equalizeHist(resized_face)
                                resized_face = clahe.apply(resized_face)
                                cv2.imshow("", resized_face)
                                cv2.waitKey(1)

                                if emotion == "Neutral":
                                    nuetral_set.append(resized_face)
                                elif emotion == "Happiness":
                                    positive_set.append(resized_face)
                                else:
                                    negative_set.append(resized_face)
                cap.release()
            except:
                pass


def prepare_officedb(nuetral_set, positive_set, negative_set):
    file_paths = glob.glob(officedb_path)
    for vedio_file_path in file_paths:
        emotion = vedio_file_path[36:38]

        cap = cv2.VideoCapture(vedio_file_path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            if random.random() < 0.8:  # around 8000 random frames in total
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                w, h = gray.shape

                gray = cv2.resize(gray, (h//2, w//2),
                                  interpolation=cv2.INTER_CUBIC)

                gray = rotate_image(gray, -90)
                # cv2.imshow("  ", gray)
                # cv2.waitKey(1)
                face = process_frame(gray, 0, 1.01, 0.05, 100, 10000)
                if len(face) > 0:
                    face = sorted(face, key=lambda y: y[3], reverse=True)[0]
                    if face[3] > face_threshold:
                        face = np.array(
                            [face[1]-face[2]/2, face[0]-face[2]/2, face[2], face[2]])
                        face = face.astype(np.int32)
                        resized_face = cv2.resize(gray[face[1]:face[1]+face[3], face[0]:face[0]+face[2]], dsize=(
                            std_width, std_width), interpolation=cv2.INTER_CUBIC)
                        # resized_face = cv2.equalizeHist(resized_face)
                        resized_face = clahe.apply(resized_face)
                        cv2.imshow("", resized_face)
                        cv2.waitKey(1)

                        if emotion == "nt":
                            nuetral_set.append(resized_face)
                        elif emotion == "hp":
                            positive_set.append(resized_face)
                        else:
                            negative_set.append(resized_face)
        cap.release()


def get_dataset(lenth):

    nuetral_set = []
    positive_set = []
    negative_set = []

    prepare_RAVDESS(nuetral_set, positive_set, negative_set)

    prepare_BAUM1(nuetral_set, positive_set, negative_set)

    # prepare CK
    read_image("happy.txt", positive_set, nuetral_set)
    read_image("angry.txt", negative_set, nuetral_set)
    read_image("sad.txt", negative_set, nuetral_set)
    read_other_neutral_image(nuetral_set)

    read_other_negative_image(negative_set)  # angry set means negetive set

    prepare_facesdb(nuetral_set, positive_set, negative_set)

    prepare_jaffe(nuetral_set, positive_set, negative_set)

    prepare_officedb(nuetral_set, positive_set, negative_set)

    prepare_ised(positive_set, negative_set)

    # prepare_SoF(nuetral_set, happy_set, sad_set)

    data = nuetral_set+positive_set+negative_set
    nuetral_set_len = len(nuetral_set)
    negative_set_len = len(negative_set)
    positive_set_len = len(positive_set)
    lable = [0]*nuetral_set_len+[1]*positive_set_len+[2]*negative_set_len

    # Augmentation
    while len(lable) < lenth:
        database_selection = random.random()
        # database_selection=0.9
        if database_selection < 0.7:  # CK+
            new_lable = np.argmin(
                [nuetral_set_len, positive_set_len, negative_set_len])

            if new_lable == 0:
                txt_name = enum_emotions[random.randint(1, 4)]+".txt"
                image_path = random.choice(list(open(txt_name)))
                folder_path = image_path.split(
                    "emotion")[0].split("Emotion")[1][:-3]
                end = int(image_path.split("emotion")[
                          0].split("Emotion")[1][-3:-1])
                middle = end//3
                index = random.randint(1, middle)
            elif new_lable == 1:
                txt_name = enum_emotions[1]+".txt"
                image_path = random.choice(list(open(txt_name)))
                folder_path = image_path.split(
                    "emotion")[0].split("Emotion")[1][:-3]
                end = int(image_path.split("emotion")[
                          0].split("Emotion")[1][-3:-1])
                middle = end//3
                index = random.randint(middle*2, end)
            else:
                txt_name = enum_emotions[random.choice([2, 3, 4])]+".txt"
                image_path = random.choice(list(open(txt_name)))
                folder_path = image_path.split(
                    "emotion")[0].split("Emotion")[1][:-3]
                end = int(image_path.split("emotion")[
                          0].split("Emotion")[1][-3:-1])
                middle = end//3
                index = random.randint(middle*2, end)

            image_path = "Images" + folder_path+"{:02d}".format(index)+".png"
            gray = cv2.imread(image_path, 0)

        elif database_selection >= 0.7 and database_selection < 0.8:  # FaceDB
            # Pick random image
            file_paths = glob.glob(facedb_path)
            image_path = random.choice(list(file_paths))
            new_lable = int(image_path.split("_img")[0].split("-")[1])
            if new_lable == 4:
                new_lable = 2
            gray = cv2.imread(image_path, 0)

        elif database_selection >= 0.8 and database_selection < 0.9:  # ised
            mat = scipy.io.loadmat(ised_mat_path)
            mat = mat['ISED_details']
            m = random.choice(mat)
            image_path = ised_image_path+m[5][0]
            if m[6][0][0] == 1:
                new_lable = 1
            else:
                new_lable = 2
            gray = cv2.imread(image_path, 0)
            w, h = gray.shape
            gray = cv2.resize(gray, (h//2, w//2),
                              interpolation=cv2.INTER_CUBIC)

        else:  # JAFFE
            file_paths = glob.glob(jaffe_path)
            image_path = random.choice(list(file_paths))
            if image_path[41:43] == "NE":
                new_lable = 0
            elif image_path[41:43] == "HA":
                new_lable = 1
            else:
                new_lable = 2
            gray = cv2.imread(image_path, 0)
            w, h = gray.shape
            gray = cv2.resize(gray, (int(h*2.5), int(w*2.5)),
                              interpolation=cv2.INTER_CUBIC)

        gray = cv2.copyMakeBorder(gray, bound, bound, bound, bound,
                                  cv2.BORDER_REFLECT_101)  # mirror bound to enable detection at bound
        h, w = gray.shape

        # Randomly augment

        # 1. Filters
        filter_type = random.randint(0, 4)
        kernel_size = random.randint(1, 3)*2+1
        if filter_type == 1:
            gray = cv2.blur(gray, (kernel_size, kernel_size))
        elif filter_type == 2:
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        elif filter_type == 3:
            gray = cv2.medianBlur(gray, kernel_size)
        elif filter_type == 4:
            gray = cv2.bilateralFilter(gray, kernel_size, 75, 75)

        # 2. Mirror
        if random.random() < 0.6:
            gray = cv2.flip(gray, 1)

        # 3. Rotation
        gray = rotate_image(gray, random.randint(-20, 20))

        # 4. Perspective transform
        # 5 modes of perspective transform
        mode = random.randint(1, 5)
        if mode == 1:
            pts1 = np.array([[random.randint(0, homography_transform), random.randint(0, homography_transform)],
                             [random.randint(w-homography_transform, w),
                              random.randint(0, homography_transform)],
                             [random.randint(0, homography_transform),
                              random.randint(h-homography_transform, h)],
                             [random.randint(w-homography_transform, w), random.randint(h-homography_transform, h)]])
        else:
            transform_extent = random.randint(
                homography_transform//3, homography_transform)
            if mode == 2:
                pts1 = np.array([[transform_extent, 0],
                                 [w-transform_extent, 0],
                                 [0, h],
                                 [w, h]])
            elif mode == 3:
                pts1 = np.array([[0, 0],
                                 [w, 0],
                                 [transform_extent, h],
                                 [w-transform_extent, h]])
            elif mode == 4:
                pts1 = np.array([[0, 0],
                                 [w, transform_extent],
                                 [0, h],
                                 [w, h-transform_extent]])

            else:
                pts1 = np.array([[0, transform_extent],
                                 [w, 0],
                                 [0, h-transform_extent],
                                 [w, h]])

        pts1 = pts1.astype(np.float32)
        pts2 = np.array([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = pts2.astype(np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        gray = cv2.warpPerspective(gray, M, (w, h))

        # Get face
        face = process_frame(gray, 0, 1.01, 0.05, 100, 10000)
        try:
            face = sorted(face, key=lambda y: y[3], reverse=True)[0]

            if face[3] > face_threshold:
                face = face.astype(np.int32)
                face = np.array([face[1]-face[2]/2, face[0] -
                                face[2]/2, face[2], face[2]])
                face = face.astype(np.int32)
                resized_face = cv2.resize(gray[face[1]:face[1]+face[3], face[0]:face[0]+face[2]], dsize=(
                    std_width, std_width), interpolation=cv2.INTER_CUBIC)
                # resized_face = cv2.equalizeHist(resized_face)
                resized_face = clahe.apply(resized_face)
                cv2.imshow("", resized_face)
                cv2.waitKey(1)
        # Add to dataset
                data.append(resized_face)
                if new_lable == 0:
                    nuetral_set_len += 1
                elif new_lable == 1:
                    positive_set_len += 1
                elif new_lable == 2:
                    negative_set_len += 1

                lable.append(new_lable)
        except:
            pass

    cv2.destroyAllWindows()
    return data, lable


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def train():

    model.load_weights(model_path)

    # face_image, lable=get_dataset(training_dataset_length)

    # with open("training_data.pkl", "wb") as fp:
    #     pickle.dump(face_image, fp)
    # with open("training_lable.pkl", "wb") as fp:
    #     pickle.dump(lable, fp)

    with open("training_data.pkl", "rb") as fp:
        face_image = pickle.load(fp)
    with open("training_lable.pkl", "rb") as fp:
        lable = pickle.load(fp)

    face_image = np.array(face_image)
    lable = np.array(lable)
    shuffle_index = np.arange(len(lable))
    np.random.shuffle(shuffle_index)
    face_image = face_image[shuffle_index]
    lable = lable[shuffle_index]

    neutral_lable = []
    smile_lable = []
    for l in lable:
        if l == 0:
            neutral_lable.append(1)
            smile_lable.append(0)
        elif l == 1:
            neutral_lable.append(0)
            smile_lable.append(1)
        else:
            neutral_lable.append(0)
            smile_lable.append(0)

    neutral_lable = np.array(neutral_lable)
    smile_lable = np.array(smile_lable)

    get_hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    hog = [get_hog.compute(f).flatten() for f in face_image]
    # landmark=[shape_to_np(landmark_predictor(f, dlib.rectangle(0, 0, std_width, std_width))).flatten() for f in face_image]
    landmark = [cal_features(shape_to_np(landmark_predictor(
        f, dlib.rectangle(0, 0, std_width, std_width)))) for f in face_image]

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.fit(x=[np.array(face_image).astype(np.float32), np.array(hog).astype(np.float32), np.array(landmark).astype(np.float32)], y=[neutral_lable, smile_lable],
              epochs=epoch,
              callbacks=[cp_callback],
              shuffle=True,
              batch_size=batch_size,
              validation_split=0)


def test():
    # model.load_weights(model_path)
    model = tf.keras.models.load_model('saved_model/my_model')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

    while True:

        ret, image = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = process_frame(gray, 0, 1.01, 0.05, 100, 10000)
        if len(face) < 1:
            continue
        face = sorted(face, key=lambda y: y[3], reverse=True)[0]
        face = np.array([face[1]-face[2]/2, face[0] -
                        face[2]/2, face[2], face[2]])
        face = face.astype(np.int32)
        resized_face = cv2.resize(gray[face[1]:face[1]+face[3], face[0]:face[0]+face[2]],
                                  dsize=(std_width, std_width), interpolation=cv2.INTER_CUBIC)
        # resized_face = cv2.equalizeHist(resized_face)
        face_image = [clahe.apply(resized_face)]

        get_hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                    histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        hog = [get_hog.compute(f).flatten() for f in face_image]
        # landmark=[shape_to_np(landmark_predictor(f, dlib.rectangle(0, 0, std_width, std_width))).flatten() for f in face_image]
        landmark = [cal_features(shape_to_np(landmark_predictor(
            f, dlib.rectangle(0, 0, std_width, std_width)))) for f in face_image]

        face_image = tf.convert_to_tensor(face_image, dtype=tf.float32)
        hog = tf.convert_to_tensor(hog, dtype=tf.float32)
        landmark = tf.convert_to_tensor(landmark, dtype=tf.float32)
        a = model.predict([face_image, hog, landmark])

        if a[0] > 0.3 and a[1] < 0.5:
            print("neutral")
        elif a[0] < 0.3 and a[1] > 0.5:
            print("positive")
        else:
            print("negative")

        cv2.imshow("emotion", image)
        cv2.waitKey(100)


def topb():
    model.load_weights(model_path)
    model._set_inputs([TensorSpec(shape=(None, 50, 50), dtype=tf.float32, name='inputs/0'), TensorSpec(shape=(None, 2304),
                      dtype=tf.float32, name='inputs/1'), TensorSpec(shape=(None, 89), dtype=tf.float32, name='inputs/2')])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    # model.save('saved_model/my_model')


if __name__ == '__main__':

    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()

    print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    model = mymodel()
    model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    topb()

