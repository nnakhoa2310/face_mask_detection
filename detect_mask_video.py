# import the necessary packages
import winsound

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

from PIL import Image, ImageTk

import face_mask_support
import os.path
import glob

from keras.utils import np_utils
import tensorflow as tf

import wget
from tkinter import filedialog
from tkinter import messagebox
import smtplib
import threading
from time import sleep
from PIL import Image, ImageTk
from tkinter import *
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import tkinter.filedialog as tkFileDialog
from pygame import mixer

from email.message import EmailMessage

import sys

import threading
import multiprocessing as mp
try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True


def vp_start_gui():

    global val, w, root
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    root = tk.Tk()
    root.iconbitmap('favicon_vku.ico')
    top = Toplevel1(root)
    face_mask_support.init(root, top)

    root.mainloop()


w = None


def create_Toplevel1(rt, *args, **kwargs):
    global w, w_win, root
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    # rt = root
    root = rt
    w = tk.Toplevel(root)
    top = Toplevel1(w)
    face_mask_support.init(w, top, *args, **kwargs)
    return (w, top)


def destroy_Toplevel1():
    global w
    w.destroy()
    w = None
# def stream():
#     global vs
#     vs = VideoStream(src=0).start()

alert_status = False
def save_frame(f):
    cv2.imwrite("./Output/detected.jpg", f)
    print("[INFO] saving image...")

def play_alarm_sound_function():
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    sound.play()
def send_mail_function():
    try:
         msg = EmailMessage()
         msg['Subject'] = 'VKU WARNING!! Someone was not wearing facemask in school'
         msg['From'] = 'nnakhoa2310@gmail.com'
         msg['To'] = 'nnakhoa2310@gmail.com'
         msg.set_content(
             'A person has been detected without a face mask. Below is the attached image of that person.Please Alert the Authorities.\n'
         )

         with open("Output/detected.jpg", "rb") as f:
             fdata = f.read()
             fname = f.name
             msg.add_attachment(fdata, maintype='Image',
                                subtype="jpg", filename=fname)

         with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
             smtp.login('nnakhoa2310@gmail.com', '')
             smtp.send_message(msg)
    except Exception as e:
         print(e)
    print('[INFO] alert mail Sent to authorities')


class Toplevel1:
    def detect_mask(self):
        def detect_and_predict_mask(frame, faceNet, maskNet):
            # grab the dimensions of the frame and then construct a blob
            # from it
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                         (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the face detections
            faceNet.setInput(blob)
            detections = faceNet.forward()
            print(detections.shape)

            # initialize our list of faces, their corresponding locations,
            # and the list of predictions from our face mask network
            faces = []
            locs = []
            preds = []

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the detection
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # ensure the bounding boxes fall within the dimensions of
                    # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # extract the face ROI, convert it from BGR to RGB channel
                    # ordering, resize it to 224x224, and preprocess it
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)

                    # add the face and bounding boxes to their respective
                    # lists
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

            # only make a predictions if at least one face was detected
            if len(faces) > 0:
                # for faster inference we'll make batch predictions on *all*
                # faces at the same time rather than one-by-one predictions
                # in the above `for` loop
                faces = np.array(faces, dtype="float32")
                preds = MaskNet.predict(faces, batch_size=32)

            # return a 2-tuple of the face locations and their corresponding
            # locations
            return (locs, preds)

        # load our serialized face detector model from disk
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the face mask detector model from disk
        MaskNet = load_model("mask_detector.model")

        # initialize the video stream
        print("[INFO] starting video stream...")

        vs = VideoStream(src=0).start()
        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            start_point = (15, 15)
            end_point = (370, 80)
            thickness = -1
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = detect_and_predict_mask(frame, faceNet, MaskNet)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutmask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text

                label = "Mask" if mask > withoutmask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)
                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                label = "Mask" if mask > withoutmask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                if (label == 'No Mask'):
                    threading.Thread(target=save_frame(f=frame)).start()
                    threading.Thread(target=send_mail_function).start()
                    threading.Thread(target=play_alarm_sound_function).start()

                elif (label == 'Mask'):
                    image = cv2.rectangle(frame, start_point,
                                          end_point, (0, 255, 0), thickness)
                    cv2.putText(image, label, (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
                    pass
                    break
                else:
                    print("Invalid")

            # show the output frame
            cv2.imshow("VKU FaceMask", frame)

            # if the `q` key was pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()
    def __init__(self, top=None):

        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9'  # X11 color: 'gray85'
        _ana1color = '#d9d9d9'  # X11 color: 'gray85'
        _ana2color = '#ececec'  # Closest X11 color: 'gray92'
        font9 = "-family {Castellar} -size 23 -weight bold -underline " \
                "1"

        top.geometry("600x577+650+150")
        top.minsize(148, 1)
        top.maxsize(1924, 1055)
        top.resizable(1, 1)
        top.title("VKU Check Mask!! Corona Virus")
        top.configure(background="#000000")

        self.Label1 = tk.Label(top)
        self.Label1.place(relx=-0.017, rely=-0.017, height=468, width=616)
        self.Label1.configure(background="#d9d9d9")
        self.Label1.configure(cursor="fleur")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(foreground="#000000")
        photo_location = os.path.join(prog_location, "vku_mask.jpg")
        global _img0
        _img0 = ImageTk.PhotoImage(file=photo_location)
        self.Label1.configure(image=_img0)

        self.Button1 = tk.Button(top)
        self.Button1.place(relx=0.0, rely=0.78, height=133, width=606)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#00ff40")
        self.Button1.configure(borderwidth="15")
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(font=font9)
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''VKU Check for Mask!''')
        self.Button1.configure(command=self.detect_mask)


if __name__ == '__main__':

    vp_start_gui()
