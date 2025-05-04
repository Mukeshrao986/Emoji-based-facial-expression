from tkinter import *
from tkinter import messagebox, filedialog, simpledialog
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

# Initialize main window
main = Tk()
main.title("Displaying Emoji Based Facial Expressions")
main.geometry("1200x1200")
main.config(bg='brown')

# Global variables
filename = None
faces = None
frame = None
detection_model_path = 'models/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'

# Load models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprise", "neutral"]

# Upload image
def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="images")
    pathlabel.config(text=filename)

# Preprocess image and detect faces
def preprocess():
    global filename, frame, faces
    text.delete('1.0', END)
    orig_frame = cv2.imread(filename)
    orig_frame = cv2.resize(orig_frame, (48, 48))
    frame = cv2.imread(filename, 0)
    faces = face_detection.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    text.insert(END, "Total number of faces detected : " + str(len(faces)))

# Detect expression in uploaded image
def detectExpression():
    global faces
    if len(faces) > 0:
        (fX, fY, fW, fH) = sorted(faces, reverse=True,
                                  key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]
        img = cv2.imread('Emoji/' + label + ".png")
        img = cv2.resize(img, (600, 400))
        cv2.putText(img, "Facial Expression Detected As : " + label,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Facial Expression Detected As : " + label, img)
        cv2.waitKey(0)
    else:
        messagebox.showinfo("Facial Expression Prediction Screen", "No face detected in uploaded image")

# Detect from a single video frame
def detectfromvideo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))

    if len(faces) > 0:
        (fX, fY, fW, fH) = sorted(faces, reverse=True,
                                  key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        roi = image[fY:fY + fH, fX:fX + fW]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        return EMOTIONS[preds.argmax()]
    return 'none'

# Detect expression from webcam
def detectWebcamExpression():
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        height, width, _ = img.shape
        result = detectfromvideo(img)
        if result != 'none':
            print(result)
            emoji = cv2.imread('Emoji/' + result + ".png")
            emoji = cv2.resize(emoji, (width, height))
            cv2.putText(emoji, "Facial Expression Detected As : " + result,
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Emoji Output", emoji)

        cv2.putText(img, "Facial Expression Detected As : " + result,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Facial Expression Output", img)

        if cv2.waitKey(650) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI components
font_title = ('times', 20, 'bold')
title = Label(main, text='Displaying Emoji Based Facial Expressions', bg='brown', fg='white', font=font_title)
title.config(height=3, width=80)
title.place(x=5, y=5)

font_btn = ('times', 14, 'bold')

upload_btn = Button(main, text="Upload Image With Face", command=upload, font=font_btn)
upload_btn.place(x=50, y=100)

pathlabel = Label(main, bg='brown', fg='white', font=font_btn)
pathlabel.place(x=300, y=100)

preprocess_btn = Button(main, text="Preprocess & Detect Face in Image", command=preprocess, font=font_btn)
preprocess_btn.place(x=50, y=150)

detect_img_btn = Button(main, text="Detect Facial Expression", command=detectExpression, font=font_btn)
detect_img_btn.place(x=50, y=200)

detect_webcam_btn = Button(main, text="Detect Facial Expression from WebCam", command=detectWebcamExpression, font=font_btn)
detect_webcam_btn.place(x=50, y=250)

font_text = ('times', 12, 'bold')
text = Text(main, height=10, width=150, font=font_text)
text.place(x=10, y=300)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)

main.mainloop()