import numpy as np
import glob
import random
import cv2
import scipy.io
import openpyxl

enum_emotions = [
    "neutral",
    "happy",
    "angry",
    "sad",
    "other"
]

bound = 20
homography_transform = 150

dataset_path = "/data/dataset"
facedb_path = dataset_path+"/facesdb/*/*/*.bmp"
jaffe_path = dataset_path+"/jaffedbase/*.tiff"
officedb_path = dataset_path+"/officedb/*.MP4"
SoF_path = dataset_path+"/SoF/*.jpg"
ised_mat_path = dataset_path+'/ISED_database/ISED_details.mat'
ised_image_path = dataset_path+'/ISED_database/'
RAVDESS_path = dataset_path+"/RAVDESS/*/*/*.mp4"
BAUM1_label_path = dataset_path+"/BAUM1/*.xlsx"
BAUM1_data_path = dataset_path+"/BAUM1/"

face_image_save_path = '/data/data_prepare/face_image.txt'
label_save_path = '/data/data_prepare/label.txt'
mutex_save_path = '/data/data_prepare/mutex.txt'

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
std_width = 50
face_threshold = 200
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))

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


def detect_face(gray):
    face = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(face) > 0:
        face = face[0]
        face = np.array([face[1], face[0], face[2], face[2]])
        if face[3] > face_threshold:
            face = face.astype(np.int32)
            resized_face = cv2.resize(gray[face[1]:face[1]+face[3], face[0]:face[0]+face[2]], dsize=(
                std_width, std_width), interpolation=cv2.INTER_CUBIC)
            resized_face = clahe.apply(resized_face)
            cv2.imshow("", resized_face)
            cv2.waitKey(1)
            return resized_face
    return False

def get_dataset():
    nuetral_set = []
    positive_set = []
    negative_set = []

    # RAVDESS
    file_paths = glob.glob(RAVDESS_path)
    for vedio_file_path in file_paths:
        mod = vedio_file_path.split("/Actor_")[1][3:5]
        if mod == "02":
            emotion = vedio_file_path.split("/Actor_")[1][9:11]
            cap = cv2.VideoCapture(vedio_file_path)
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                if random.random() < 0.05:  # around 20000 random frames in total
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    w, h = gray.shape
                    gray = cv2.resize(gray, (h//2, w//2),
                                      interpolation=cv2.INTER_CUBIC)
                    face = detect_face(gray)
                    if face:
                        if emotion == "01":
                            nuetral_set.append(face)
                        elif emotion == "03":
                            positive_set.append(face)
                        else:
                            negative_set.append(face)
            cap.release()

    # BAUM1
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
                        face = detect_face(gray)
                        if face:
                            if emotion == "Neutral":
                                nuetral_set.append(face)
                            elif emotion == "Happiness":
                                positive_set.append(face)
                            else:
                                negative_set.append(face)
                cap.release()
            except:
                pass

    def read_image(txt, emotion_set, nuetral_set):
        with open(txt, "r") as f:
            for path in f:
                folder_path = path.split("emotion")[0].split("Emotion")[1][:-3]
                end = int(path.split("emotion")[0].split("Emotion")[1][-3:-1])
                middle = end//3
                for i in range(1, middle):
                    nuetral_set.append(detect_face(cv2.imread("Images" + folder_path +
                                                              "{:02d}".format(i)+".png", 0)))
                for i in range(middle*2, end):
                    emotion_set.append(detect_face(cv2.imread("Images" + folder_path +
                                                              "{:02d}".format(i)+".png", 0)))
        f.close()

    # CK
    read_image("happy.txt", positive_set, nuetral_set)
    read_image("angry.txt", negative_set, nuetral_set)
    read_image("sad.txt", negative_set, nuetral_set)
    read_image("other.txt", negative_set, nuetral_set)

    # facesdb
    file_paths = glob.glob(facedb_path)
    for _, file_path in enumerate(file_paths):
        label = int(file_path.split("_img")[0].split("-")[1])
        if label == 2:
            label = 3
        elif label == 4:
            label = 2
        face = detect_face(cv2.imread(file_path))
        if label == 0:
            nuetral_set.append(face)
        elif label == 1:
            positive_set.append(face)
        else:
            negative_set.append(face)

    # jaffe
    file_paths = glob.glob(jaffe_path)
    for _, file_path in enumerate(file_paths):
        emotion = file_path[41:43]
        label = 2
        if emotion == "NE":
            label = 0
        elif emotion == "HA":
            label = 1

        face = detect_face(cv2.imread(file_path))
        if label == 0:
            nuetral_set.append(face)
        elif label == 1:
            positive_set.append(face)
        else:
            negative_set.append(face)

    # SoF
    file_paths = glob.glob(SoF_path)
    for _, file_path in enumerate(file_paths):
        emotion = file_path[55:57]
        label = -1
        if emotion == "no":
            label = 0
        elif emotion == "hp":
            label = 1
        else:
            label = 3
        gray = cv2.imread(file_path, 0)
        face = detect_face(cv2.imread(file_path, 0))
        if face:
            if label == 0:
                nuetral_set.append(face)
            elif label == 1:
                positive_set.append(face)
            elif label == 3:
                negative_set.append(face)

    # officedb
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
                face = detect_face(gray)
                if face:
                    if emotion == "nt":
                        nuetral_set.append(face)
                    elif emotion == "hp":
                        positive_set.append(face)
                    else:
                        negative_set.append(face)
        cap.release()

    # ised
    mat = scipy.io.loadmat(ised_mat_path)
    mat = mat['ISED_details']
    for m in mat:
        img_path = ised_image_path+m[5][0]
        label = m[6][0][0]
        gray = cv2.imread(img_path, 0)

        w, h = gray.shape
        gray = cv2.resize(gray, (h//2, w//2), interpolation=cv2.INTER_CUBIC)

        face = detect_face(gray)
        if face:
            if label == 1:
                positive_set.append(face)
            else:
                negative_set.append(face)

    face_image = nuetral_set+positive_set+negative_set
    nuetral_set_len = len(nuetral_set)
    negative_set_len = len(negative_set)
    positive_set_len = len(positive_set)
    label = [0]*nuetral_set_len+[1]*positive_set_len+[2]*negative_set_len
    return face_image, label


def augmentation(lenth):
    face_image = []
    label = []
    nuetral_set_len = 0
    negative_set_len = 0
    positive_set_len = 0
    while len(label) < lenth:
        database_selection = random.random()
        if database_selection < 0.7:  # CK+
            new_label = np.argmin(
                [nuetral_set_len, positive_set_len, negative_set_len])

            if new_label == 0:
                txt_name = enum_emotions[random.randint(1, 4)]+".txt"
                image_path = random.choice(list(open(txt_name)))
                folder_path = image_path.split(
                    "emotion")[0].split("Emotion")[1][:-3]
                end = int(image_path.split("emotion")[
                    0].split("Emotion")[1][-3:-1])
                middle = end//3
                index = random.randint(1, middle)
            elif new_label == 1:
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
            new_label = int(image_path.split("_img")[0].split("-")[1])
            if new_label == 4:
                new_label = 2
            gray = cv2.imread(image_path, 0)

        elif database_selection >= 0.8 and database_selection < 0.9:  # ised
            mat = scipy.io.loadmat(ised_mat_path)
            mat = mat['ISED_details']
            m = random.choice(mat)
            image_path = ised_image_path+m[5][0]
            if m[6][0][0] == 1:
                new_label = 1
            else:
                new_label = 2
            gray = cv2.imread(image_path, 0)
            w, h = gray.shape
            gray = cv2.resize(gray, (h//2, w//2),
                              interpolation=cv2.INTER_CUBIC)

        else:  # JAFFE
            file_paths = glob.glob(jaffe_path)
            image_path = random.choice(list(file_paths))
            if image_path[41:43] == "NE":
                new_label = 0
            elif image_path[41:43] == "HA":
                new_label = 1
            else:
                new_label = 2
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
        face = detect_face(gray)
        try:
            if face:
                face_image.append(face)
                if new_label == 0:
                    nuetral_set_len += 1
                elif new_label == 1:
                    positive_set_len += 1
                elif new_label == 2:
                    negative_set_len += 1
                label.append(new_label)
        except:
            pass

    cv2.destroyAllWindows()
    return face_image, label


if __name__ == '__main__':
    while True:
        if np.loadtxt(mutex_save_path, dtype=int)[0] == 0:
            face_image, label = augmentation(10000)
            np.savetxt(face_image_save_path, np.array(face_image), fmt='%d')
            np.savetxt(label_save_path, np.array(label), fmt='%d')
            np.savetxt(mutex_save_path, np.array([1]), fmt='%d')
