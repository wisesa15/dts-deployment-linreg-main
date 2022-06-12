from flask import Flask, request, render_template, Response, redirect,url_for
from werkzeug.utils import secure_filename
import os
import main
import numpy as np
from PIL import Image
import cv2

UPLOAD_FOLDER = os.path.join('static', 'uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = main.load_model('Model/model_1.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
width = camera. get(3)
width = int(width)
height = camera. get(4)
height = int(height)
global faceDetection
faceDetection = 1

def faceMaskWebcam():
    while True:
        success, frame = camera.read() 
        if success:
            if(faceDetection):
                listFaces,coorFaces = main.faceDetection(face_cascade,frame)
                frame = cv2.flip(frame,1)
                if len(listFaces)>0:
                    labels=[]
                    colors=[]
                    for face in listFaces:
                        pred = model.predict(face)
                        label = 'Masked' if pred[0][0] > 0.5 else 'Not Masked'
                        color = (0, 255, 0) if pred[0][0] > 0.5 else (0, 0, 255)
                        labels.append(label)
                        colors.append(color)
                    nFace=0
                    for (x, y, w, h) in coorFaces:
                        cv2.putText(frame, labels[nFace], (width - x - w, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[nFace], 2)
                        cv2.rectangle(frame, (width-x, y), (width-x - w, y + h),colors[nFace], 2)
                        nFace+=1
            else:
                image = cv2.resize(frame, (224, 224))
                image = np.array(image)
                image = np.expand_dims(image, axis=0)
                pred = model.predict(image)
                label = 'Masked' if pred[0][0] > 0.5 else 'Not Masked'
                color = (0, 255, 0) if pred[0][0] > 0.5 else (0, 0, 255)
                frame = cv2.flip(frame,1)
                cv2.putText(frame, label, (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            ret, jpeg = cv2.imencode('.jpg',frame )
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            pass


@app.route('/')
def mask():
    return render_template('index.html', isSubmitted=False, isWebCam=False, faceDetection = faceDetection)

@app.route('/', methods=['POST'])
def maskdetect():
    # acces image file that has been uploaded
    file = request.files['file']
    filename = secure_filename(file.filename)
    filePath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
    file.save(filePath)
    
    # opening image and resize for detection without facial recognition
    img = Image.open(filePath)
    img2 = img.resize((224, 224))
    img2 = np.array(img2)
    img2 = img2.reshape((1, img2.shape[0], img2.shape[1], img2.shape[2]))
    img = np.array(img)


    if(faceDetection):
        # detecting face and its coordinate 
        listFaces,coorFaces = main.faceDetection(face_cascade,img)

        # detecting face mask 
        if len(listFaces)>0:
            colors = []
            labels = []
            
            for face in listFaces:
                pred = model.predict(face)
                label = 'Masked' if pred[0][0] > 0.5 else 'Not Masked'
                color = (0, 255, 0) if pred[0][0] > 0.5 else (255, 0, 0)
                labels.append(label)
                colors.append(color)
            nFace=0
            for (x, y, w, h) in coorFaces:
                cv2.putText(img, labels[nFace], (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[nFace], 2)
                cv2.rectangle(img, (x, y), (x + w, y + h),colors[nFace], 2)
                nFace+=1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(filePath,img)
            return render_template('index.html', isSubmitted=True,isWebCam=False ,image=filePath, isMulti=True,faceDetection = faceDetection )
        else:
            return render_template('index.html', isSubmitted=True,isWebCam=False ,image=filePath, result = -1,isMulti=False,faceDetection = faceDetection)
    
    else:
        preds = model.predict(img2)
        labels = 1*(preds > 0.5)

    return render_template('index.html', isSubmitted=True,isWebCam=False ,image=filePath, result = labels[0][0],isMulti=False,faceDetection = faceDetection)

@app.route('/webcam')
def webcam():
    return render_template('index.html', isSubmitted=False,isWebCam=True,faceDetection = faceDetection)


@app.route('/video_feed')
def video_feed():
    return Response(faceMaskWebcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch')
def switch():
    global faceDetection
    faceDetection = 0 if faceDetection == 1 else 1
    return redirect(url_for('mask'))

if __name__ == '__main__':
    app.run(debug=True)
