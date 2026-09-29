import cv2
import sys
import numpy as np
import face_recognition
import os
from PIL import Image, ImageTk
from tkinter import *

def onClossing():
    cap.release()
    root.destroy()
def captureFace():
    pass

procesar_frame = True
loc_rostros = []
nombre_rostros = []
ultimo_frame_leido = True
def callback():
    global procesar_frame, loc_rostros, nombre_rostros, ultimo_frame_leido

    ret, frame = cap.read()
    if ret:
        ultimo_frame_leido = frame.copy()
        frame_pequeno = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rgb_frame_pequeno = cv2.cvtColor(frame_pequeno,cv2.COLOR_BGR2RGB)
        if procesar_frame:
            for encoding_rostro in encoding_rostros:
                nombre_rostro = 'Rostro Desconocido'
                if (rostros_conocidos_encoding) > 0:
                    coincidencias = face_recognition.compare_faces(rostros_conocidos_encoding,encoding_rostro,tolerance=0.6)
                    distancia = face_recognition.face_distances(rostros_conocidos_encoding)
                nombre_rostro.append(nombre)
    procesar_frame = not procesar_frame
    pass
url = 'http://ip:81/stream'
cap = cv2.VideoCapture(url)
root = Tk()
root.protocol("WM_DELETE_WINDOW",onClossing)
root.title("ESPCAMP Facial Recognition")
label = Label(root)
label.grid(row=0,column=0,columnspan=2)

frame_controls = Frame(root)
frame_controls.grid(row=1,column=1,columnspan=2, pady=12)

lbl1 = Label(frame_controls, text="Ingresar nombre del usuario: ")
lbl1.pack(side=LEFT,padx=7)

tbx1 = Entry(frame_controls,width=20)
tbx1.pack(side=LEFT,padx=7)

btn1 = Button(frame_controls,text="Captura Rosto",command=captureFace)
btn1.pack(side=LEFT,padx=7)

root.after(1,callback)
root.mainloop()