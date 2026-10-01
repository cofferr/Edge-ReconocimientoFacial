import cv2
import sys
import numpy as np
import face_recognition 
import os  
from datetime import datetime 
from tkinter import *
from PIL import Image, ImageTk

def onClossing():
    cap.release()
    print('Camara desconectada!!!!')
    root.destroy()


CARPA_ROSTROS = "rostros"  
if not os.path.exists(CARPA_ROSTROS):  
    os.makedirs(CARPA_ROSTROS)

rostros_conocidos_encodings = []  
rostros_conocidos_nombres = []  

def cargar_rostros_locales():  
    global rostros_conocidos_encodings, rostros_conocidos_nombres
    rostros_conocidos_encodings.clear() 
    rostros_conocidos_nombres.clear()
    
    for archivo in os.listdir(CARPA_ROSTROS):  
        if archivo.endswith(".jpg") or archivo.endswith(".png"):  
            ruta_completa = os.path.join(CARPA_ROSTROS, archivo)  
            imagen = face_recognition.load_image_file(ruta_completa)  
            encodings = face_recognition.face_encodings(imagen) 
            
            if len(encodings) > 0:  
                rostros_conocidos_encodings.append(encodings[0])  
                nombre_sin_extension = os.path.splitext(archivo)[0]  
                rostros_conocidos_nombres.append(nombre_sin_extension)  
    print(f"Base de datos actualizada. Rostros cargados: {rostros_conocidos_nombres}")

cargar_rostros_locales()  

rostros_detectados_recientemente = {}  

def registrar_acceso(nombre):  
    ahora = datetime.now()  
    fecha_hoy = ahora.strftime("%Y-%m-%d")  
    hora_actual = ahora.strftime("%H:%M:%S")  
    
    
    if nombre in rostros_detectados_recientemente:
        if (ahora - rostros_detectados_recientemente[nombre]).total_seconds() < 10:
            return
            
    rostros_detectados_recientemente[nombre] = ahora  
    
    with open("registro_accesos.txt", "a") as archivo_txt:  
        archivo_txt.write(f"[{fecha_hoy} {hora_actual}] - {nombre} detectado en camara.\n")  
    print(f"[HISTORIAL] {nombre} registrado a las {hora_actual}")

def registrar_nuevo_usuario():  
    nombre_nuevo = entrada_nombre.get().strip()  
    if not nombre_nuevo:  
        print("ERROR: Por favor ingresa un nombre antes de tomar la foto.")
        return
        
    ret, frame_registro = cap.read()  
    if ret:  
        ruta_guardar = os.path.join(CARPA_ROSTROS, f"{nombre_nuevo}.jpg")  
        cv2.imwrite(ruta_guardar, frame_registro)  
        print(f"FOTO GUARDADA: {ruta_guardar}")
        entrada_nombre.delete(0, END)  
        cargar_rostros_locales()  

procesar_este_frame = True
loc_rostros = []
nombres_rostros = []
ultimo_frame_leido = None  

def callback():
    global procesar_este_frame, loc_rostros, nombres_rostros, ultimo_frame_leido
    
    ret, frame = cap.read()

    if ret:
        ultimo_frame_leido = frame.copy()  
        frame_pequeno = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame_pequeno = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)

        if procesar_este_frame:
            loc_rostros = face_recognition.face_locations(rgb_frame_pequeno)
            encodings_rostros = face_recognition.face_encodings(rgb_frame_pequeno, loc_rostros)

            nombres_rostros = []
            for encoding_rostro in encodings_rostros:
                nombre = "Desconocido"
                
                if len(rostros_conocidos_encodings) > 0:
                    coincidencias = face_recognition.compare_faces(rostros_conocidos_encodings, encoding_rostro, tolerance=0.55)
                    distancias = face_recognition.face_distance(rostros_conocidos_encodings, encoding_rostro)
                    
                    if len(distancias) > 0:
                        mejor_coincidencia = np.argmin(distancias)
                        if coincidencias[mejor_coincidencia]:
                            nombre = rostros_conocidos_nombres[mejor_coincidencia]

                nombres_rostros.append(nombre)
                registrar_acceso(nombre)

        procesar_este_frame = not procesar_este_frame

        for (top, right, bottom, left), nombre in zip(loc_rostros, nombres_rostros):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, nombre, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((400, 400))
        tkimage = ImageTk.PhotoImage(img)
        label.configure(image = tkimage)
        label.image = tkimage
        root.after(1, callback)
    else:
        onClossing()

url = url ='http://IP:81/stream'
cap = cv2.VideoCapture(url)

if cap.isOpened():
    print('Camara Inicializada Correctamente!!!')
else:
    sys.exit('Revise la conexión a la camara.....')

root = Tk()
root.protocol('WM_DELETE_WINDOW', onClossing)
root.title('Ejemplo Manejo Camara ESP-CAM con Tkinter')

label = Label(root)
label.grid(row=0, column=0, columnspan=2)  


frame_controles = Frame(root)  
frame_controles.grid(row=1, column=0, columnspan=2, pady=10)  

lbl_instruccion = Label(frame_controles, text="Nombre del nuevo usuario:")  
lbl_instruccion.pack(side=LEFT, padx=5)  

entrada_nombre = Entry(frame_controles, width=20)  
entrada_nombre.pack(side=LEFT, padx=5)  

btn_guardar = Button(frame_controles, text="Tomar Foto y Registrar", command=registrar_nuevo_usuario, bg="#4CAF50", fg="white")  # NUEVO: Botón verde
btn_guardar.pack(side=LEFT, padx=5)  


root.after(1, callback)
root.mainloop()
