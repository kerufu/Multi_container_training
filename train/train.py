import tensorflow as tf
from tensorflow.keras.regularizers import l2
import numpy as np

std_width = 50

face_image_load_path = '/data/preprocess/face_image.txt'
neutral_label_load_path = '/data/preprocess/neutral_label.txt'
positive_label_load_path = '/data/preprocess/positive_label.txt'
landmark_load_path = '/data/preprocess/landmark.txt'
hog_load_path = '/data/preprocess/hog.txt'
mutex_load_path = '/data/preprocess/mutex.txt'


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
            tf.keras.layers.Dense(16, activation='relu',
                                  kernel_regularizer=l2(0.00000)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.0),
            tf.keras.layers.Dense(4, activation='relu',
                                  kernel_regularizer=l2(0.00000))
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
            tf.keras.layers.Dense(32, activation='relu',
                                  kernel_regularizer=l2(0.00000)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(8, activation='relu',
                                  kernel_regularizer=l2(0.00000))
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class mymodel(tf.keras.Model):
    def __init__(self):
        super(mymodel, self).__init__()
        self.CNN_module = CNN_model()
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
        feature_CNN = self.CNN_module(face_image)
        feature_HOG = self.HOG_module(HOG)
        feature_landmark = self.Landmark_module(landmark)
        neutral_feature = tf.keras.layers.concatenate(
            [feature_CNN, feature_HOG, feature_landmark])
        smile_feature = tf.keras.layers.concatenate(
            [feature_CNN, feature_HOG, feature_landmark])

        for layer in self.neutral_output_module:
            neutral_feature = layer(neutral_feature)

        for layer in self.smile_output_module:
            smile_feature = layer(smile_feature)

        return [neutral_feature, smile_feature]


model_path = "./saved_model"
epoch = 10000
model = mymodel()
model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 save_weights_only=True,
                                                 verbose=1)
batch_size = 50


def train(face_image, hog, landmark, neutral_lable, smile_lable):
    model.load_weights(model_path)
    model.fit(x=[np.array(face_image).astype(np.float32), np.array(hog).astype(np.float32), np.array(landmark).astype(np.float32)], y=[neutral_lable, smile_lable],
              epochs=epoch,
              callbacks=[cp_callback],
              shuffle=True,
              batch_size=batch_size,
              validation_split=0)


def convert():
    model.load_weights(model_path)
    model._set_inputs([tf.TensorSpec(shape=(None, 50, 50), dtype=tf.float32, name='inputs/0'), tf.TensorSpec(shape=(None, 2304),
                      dtype=tf.float32, name='inputs/1'), tf.TensorSpec(shape=(None, 89), dtype=tf.float32, name='inputs/2')])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

if __name__ == '__main__':
    while True:
        if np.loadtxt(mutex_load_path, dtype=int)[0] == 1:
            face_image = np.loadtxt(face_image_load_path, dtype=int)
            hog = np.loadtxt(hog_load_path, dtype=int)
            landmark = np.loadtxt(landmark_load_path, dtype=int)
            neutral_label = np.loadtxt(neutral_label_load_path, dtype=int)
            positive_label = np.loadtxt(positive_label_load_path, dtype=int)
            train(face_image, hog, landmark, neutral_label, positive_label)

            np.savetxt(mutex_load_path, np.array([0]), fmt='%d')

