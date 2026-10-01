import cv2
import sys
import numpy as np
import face_recognition
import os
from tkinter import *
from PIL import Image, ImageTk

CARPETA_ROSTROS = 'rostros_conocidos'

def cargar_rostros_conocidos():
    nombres = []
    encodings = []
    os.makedirs(CARPETA_ROSTROS, exist_ok=True)
    for archivo in os.listdir(CARPETA_ROSTROS):
        ruta = os.path.join(CARPETA_ROSTROS, archivo)
        imagen = face_recognition.load_image_file(ruta)
        encoding_rostro = face_recognition.face_encodings(imagen)
        if encoding_rostro:
            encodings.append(encoding_rostro[0])
            nombres.append(os.path.splitext(archivo)[0])
    return nombres, encodings

def onClossing():
    cap.release()
    root.destroy()

def captureFace():
    global rostros_conocidos_nombres, rostros_conocidos_encoding

    nombre = tbx1.get().strip()
    if not nombre:
        lbl_estado.configure(text="Ingresa un nombre antes de capturar", fg="red")
        return
    if ultimo_frame_leido is None:
        lbl_estado.configure(text="Todavia no hay imagen de la camara", fg="red")
        return

    rgb_frame = cv2.cvtColor(ultimo_frame_leido, cv2.COLOR_BGR2RGB)
    ubicaciones = face_recognition.face_locations(rgb_frame)
    if not ubicaciones:
        lbl_estado.configure(text="No se detecto ningun rostro, acercate o mejora la luz", fg="red")
        return
    if len(ubicaciones) > 1:
        lbl_estado.configure(text="Se detecto mas de un rostro, debe haber solo uno", fg="red")
        return

    encoding_rostro = face_recognition.face_encodings(rgb_frame, ubicaciones)[0]

    ruta = os.path.join(CARPETA_ROSTROS, f'{nombre}.jpg')
    cv2.imwrite(ruta, ultimo_frame_leido)

    rostros_conocidos_nombres.append(nombre)
    rostros_conocidos_encoding.append(encoding_rostro)
    tbx1.delete(0, END)
    lbl_estado.configure(text=f"Rostro de '{nombre}' registrado correctamente", fg="green")

procesar_frame = True
loc_rostros = []
nombre_rostros = []
ultimo_frame_leido = None
rostros_conocidos_nombres, rostros_conocidos_encoding = cargar_rostros_conocidos()

def callback():
    global procesar_frame, loc_rostros, nombre_rostros, ultimo_frame_leido

    if not cap.isOpened():
        cap.open(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, frame = cap.read()
    if ret:
        ultimo_frame_leido = frame.copy()
        frame_pequeno = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame_pequeno = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)

        if procesar_frame:
            loc_rostros = face_recognition.face_locations(rgb_frame_pequeno)
            encoding_rostros = face_recognition.face_encodings(rgb_frame_pequeno, loc_rostros)

            nombre_rostros = []
            for encoding_rostro in encoding_rostros:
                nombre_rostro = 'Rostro Desconocido'
                if len(rostros_conocidos_encoding) > 0:
                    coincidencias = face_recognition.compare_faces(rostros_conocidos_encoding, encoding_rostro, tolerance=0.6)
                    distancia = face_recognition.face_distance(rostros_conocidos_encoding, encoding_rostro)
                    mejor_coincidencia = np.argmin(distancia)
                    if coincidencias[mejor_coincidencia]:
                        nombre_rostro = rostros_conocidos_nombres[mejor_coincidencia]
                nombre_rostros.append(nombre_rostro)

        procesar_frame = not procesar_frame

        escala = frame.shape[0] / 480
        grosor_linea = max(1, round(2 * escala))
        tamano_fuente = 0.6 * escala
        grosor_fuente = max(1, round(1 * escala))
        alto_etiqueta = round(25 * escala)

        for (top, right, bottom, left), nombre_rostro in zip(loc_rostros, nombre_rostros):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), grosor_linea)
            cv2.rectangle(frame, (left, bottom - alto_etiqueta), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, nombre_rostro, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, tamano_fuente, (255, 255, 255), grosor_fuente)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((640, 480))

        tkimage = ImageTk.PhotoImage(img)
        label.configure(image=tkimage)
        label.image = tkimage

        root.after(1, callback)
    else:
        cap.release()
        root.after(1, callback)

url = 'http://10.62.82.181/320x240.mjpeg'
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

lbl_estado = Label(root, text="")
lbl_estado.grid(row=2,column=0,columnspan=2)

root.after(1,callback)
root.mainloop()
