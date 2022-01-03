import numpy as np
import dlib
import cv2

std_width = 50
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
get_hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
landmark_predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")

face_image_load_path = '/data/data_prepare/face_image.txt'
label_load_path = '/data/data_prepare/label.txt'
mutex_load_path = '/data/data_prepare/mutex.txt'

face_image_save_path = '/data/preprocess/face_image.txt'
neutral_label_save_path = '/data/preprocess/neutral_label.txt'
positive_label_save_path = '/data/preprocess/positive_label.txt'
landmark_save_path = '/data/preprocess/landmark.txt'
hog_save_path = '/data/preprocess/hog.txt'
mutex_save_path = '/data/preprocess/mutex.txt'


def point_between(p1, p2):
    return np.array([abs(p1[0] - p2[0]) / 2.0 + min(p1[0], p2[0]), abs(p1[1] - p2[1]) / 2.0 + min(p1[1], p2[1])])


def distance_between(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def cal_features(landmarks):
    # the shape of landmarks should be (68, 2)
    features = [0] * 89

    lmk = landmarks
    left_eye = point_between(lmk[36], lmk[39])
    right_eye = point_between(lmk[42], lmk[45])
    nose = point_between(lmk[30], lmk[33])
    between_eyes = distance_between(left_eye, right_eye)
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


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def preprocess(face_image, label):
    face_image = np.array(face_image)
    label = np.array(label)
    neutral_label = []
    positive_label = []
    for l in label:
        if l == 0:
            neutral_label.append(1)
            positive_label.append(0)
        elif l == 1:
            neutral_label.append(0)
            positive_label.append(1)
        else:
            neutral_label.append(0)
            positive_label.append(0)

    neutral_label = np.array(neutral_label)
    positive_label = np.array(positive_label)

    hog = [get_hog.compute(f).flatten() for f in face_image]
    landmark = [cal_features(shape_to_np(landmark_predictor(
        f, dlib.rectangle(0, 0, std_width, std_width)))) for f in face_image]
    return hog, landmark, neutral_label, positive_label
    

if __name__ == '__main__':
    while True:
        if np.loadtxt(mutex_load_path, dtype=int)[0] == 1 and np.loadtxt(mutex_save_path, dtype=int)[0] == 0:
            hog = []
            landmark = []
            neutral_label = []
            positive_label = []
            face_image = np.loadtxt(face_image_load_path, dtype=int)
            label = np.loadtxt(label_load_path, dtype=int)
            for fi, l in zip(face_image, label):
                h, lm, nl, sl = preprocess(fi, l)
                hog.append(h)
                landmark.append(lm)
                neutral_label.append(nl)
                positive_label.append(sl)
            np.savetxt(face_image_save_path, face_image, fmt='%d')
            np.savetxt(hog_save_path, np.array(hog), fmt='%d')
            np.savetxt(landmark_save_path, np.array(face_image), fmt='%d')
            np.savetxt(neutral_label_save_path, np.array(neutral_label), fmt='%d')
            np.savetxt(positive_label_save_path, np.array(positive_label), fmt='%d')

            np.savetxt(mutex_load_path, np.array([0]), fmt='%d')
            np.savetxt(mutex_save_path, np.array([1]), fmt='%d')

