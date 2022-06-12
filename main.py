import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
import cv2
import numpy as np

def load_model(path):
    inputs = layers.Input(shape=(224, 224, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(units = 1, activation="sigmoid", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy",keras.metrics.Precision(),keras.metrics.Recall()]
    )
    model.load_weights(path)
    return model

def faceDetection(faceCasc,frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCasc.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    faces_list=[]
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h,x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = np.array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        faces_list.append(face_frame)
    return faces_list,faces







