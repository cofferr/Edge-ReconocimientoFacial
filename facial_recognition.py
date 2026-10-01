import os
from datetime import datetime

import cv2
import numpy as np
import face_recognition
from tkinter import *
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

URL_CAMARA = 'http://10.62.82.181/320x240.mjpeg'
CARPETA_ROSTROS = 'rostros_conocidos'
EXTENSIONES_VALIDAS = ('.jpg', '.png')
ARCHIVO_REGISTRO_ACCESOS = 'registro_accesos.txt'
SEGUNDOS_ENTRE_REGISTROS = 10    # evita re-registrar al mismo rostro en cada frame

FACTOR_REDUCCION = 0.25          # cuanto se reduce el frame antes de analizarlo
TOLERANCIA_COINCIDENCIA = 0.6    # distancia maxima para considerar un rostro "conocido"
TAMANO_MINIATURA = (640, 480)    # tamano maximo mostrado en la ventana

COLOR_CONOCIDO = (0, 255, 0)
COLOR_DESCONOCIDO = (0, 0, 255)
NOMBRE_DESCONOCIDO = 'Rostro Desconocido'


# ---------------------------------------------------------------------------
# Carga de rostros conocidos
# ---------------------------------------------------------------------------

def cargar_rostros_conocidos(carpeta):
    nombres = []
    encodings = []

    os.makedirs(carpeta, exist_ok=True)
    for archivo in os.listdir(carpeta):
        if not archivo.lower().endswith(EXTENSIONES_VALIDAS):
            continue

        ruta = os.path.join(carpeta, archivo)
        imagen = face_recognition.load_image_file(ruta)
        encoding_rostro = face_recognition.face_encodings(imagen)
        if encoding_rostro:
            encodings.append(encoding_rostro[0])
            nombres.append(os.path.splitext(archivo)[0])

    return nombres, encodings


# ---------------------------------------------------------------------------
# Reconocimiento facial
# ---------------------------------------------------------------------------

def detectar_rostros(frame_pequeno):
    """Devuelve (ubicaciones, encodings) de los rostros en un frame reducido."""
    rgb_frame = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)
    ubicaciones = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, ubicaciones)
    return ubicaciones, encodings


def identificar_encoding(encoding_rostro, nombres_conocidos, encodings_conocidos):
    """Devuelve el nombre asociado a un encoding, o NOMBRE_DESCONOCIDO."""
    if not encodings_conocidos:
        return NOMBRE_DESCONOCIDO

    coincidencias = face_recognition.compare_faces(
        encodings_conocidos, encoding_rostro, tolerance=TOLERANCIA_COINCIDENCIA
    )
    distancias = face_recognition.face_distance(encodings_conocidos, encoding_rostro)
    mejor_indice = np.argmin(distancias)

    if coincidencias[mejor_indice]:
        return nombres_conocidos[mejor_indice]
    return NOMBRE_DESCONOCIDO


def identificar_rostros(encodings_rostros, nombres_conocidos, encodings_conocidos):
    return [
        identificar_encoding(encoding, nombres_conocidos, encodings_conocidos)
        for encoding in encodings_rostros
    ]


# ---------------------------------------------------------------------------
# Registro de accesos (bitacora en archivo)
# ---------------------------------------------------------------------------

class RegistroAccesos:
    """Escribe en disco cada vez que se reconoce un rostro, evitando duplicados
    del mismo nombre dentro de un intervalo corto (SEGUNDOS_ENTRE_REGISTROS)."""

    def __init__(self, ruta_archivo, segundos_entre_registros):
        self.ruta_archivo = ruta_archivo
        self.segundos_entre_registros = segundos_entre_registros
        self.ultimo_registro_por_nombre = {}

    def registrar(self, nombre):
        if nombre == NOMBRE_DESCONOCIDO:
            return

        ahora = datetime.now()
        ultimo_registro = self.ultimo_registro_por_nombre.get(nombre)
        if ultimo_registro and (ahora - ultimo_registro).total_seconds() < self.segundos_entre_registros:
            return

        self.ultimo_registro_por_nombre[nombre] = ahora
        with open(self.ruta_archivo, 'a') as archivo:
            archivo.write(f"[{ahora.strftime('%Y-%m-%d %H:%M:%S')}] - {nombre} detectado en camara.\n")


# ---------------------------------------------------------------------------
# Dibujo sobre el frame
# ---------------------------------------------------------------------------

def escalar_ubicacion(ubicacion, factor):
    top, right, bottom, left = ubicacion
    return top * factor, right * factor, bottom * factor, left * factor


def dibujar_rostro(frame, ubicacion, nombre, escala):
    grosor_linea = max(1, round(2 * escala))
    grosor_fuente = max(1, round(1 * escala))
    tamano_fuente = 0.6 * escala
    alto_etiqueta = round(25 * escala)
    color = COLOR_CONOCIDO if nombre != NOMBRE_DESCONOCIDO else COLOR_DESCONOCIDO

    top, right, bottom, left = escalar_ubicacion(ubicacion, round(1 / FACTOR_REDUCCION))

    cv2.rectangle(frame, (left, top), (right, bottom), color, grosor_linea)
    cv2.rectangle(frame, (left, bottom - alto_etiqueta), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        frame, nombre, (left + 6, bottom - 6),
        cv2.FONT_HERSHEY_DUPLEX, tamano_fuente, (255, 255, 255), grosor_fuente,
    )


def dibujar_rostros(frame, ubicaciones, nombres):
    escala = frame.shape[0] / TAMANO_MINIATURA[1]
    for ubicacion, nombre in zip(ubicaciones, nombres):
        dibujar_rostro(frame, ubicacion, nombre, escala)


def frame_a_imagen_tk(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img.thumbnail(TAMANO_MINIATURA)
    return ImageTk.PhotoImage(img)


# ---------------------------------------------------------------------------
# Estado de la aplicacion
# ---------------------------------------------------------------------------

class EstadoApp:
    def __init__(self, carpeta_rostros, registro_accesos):
        self.nombres_conocidos, self.encodings_conocidos = cargar_rostros_conocidos(carpeta_rostros)
        self.registro_accesos = registro_accesos
        self.procesar_frame = True
        self.ubicaciones_rostros = []
        self.nombres_rostros = []
        self.ultimo_frame_leido = None

    def registrar_rostro(self, nombre, encoding):
        self.nombres_conocidos.append(nombre)
        self.encodings_conocidos.append(encoding)

    def actualizar_deteccion(self, ubicaciones, nombres):
        self.ubicaciones_rostros = ubicaciones
        self.nombres_rostros = nombres
        for nombre in nombres:
            self.registro_accesos.registrar(nombre)

    def alternar_procesamiento(self):
        self.procesar_frame = not self.procesar_frame


# ---------------------------------------------------------------------------
# Logica de camara / video
# ---------------------------------------------------------------------------

def abrir_camara(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def asegurar_camara_abierta(cap, url):
    if not cap.isOpened():
        cap.open(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def procesar_frame_video(frame, estado):
    """Actualiza el estado de deteccion (solo en frames alternos) y devuelve el frame dibujado."""
    estado.ultimo_frame_leido = frame.copy()

    if estado.procesar_frame:
        frame_pequeno = cv2.resize(frame, (0, 0), fx=FACTOR_REDUCCION, fy=FACTOR_REDUCCION)
        ubicaciones, encodings = detectar_rostros(frame_pequeno)
        nombres = identificar_rostros(encodings, estado.nombres_conocidos, estado.encodings_conocidos)
        estado.actualizar_deteccion(ubicaciones, nombres)

    estado.alternar_procesamiento()

    dibujar_rostros(frame, estado.ubicaciones_rostros, estado.nombres_rostros)
    return frame


# ---------------------------------------------------------------------------
# Registro de un nuevo rostro (boton "Captura Rostro")
# ---------------------------------------------------------------------------

def capturar_rostro(nombre, frame, carpeta_rostros):
    """
    Intenta registrar un rostro a partir de un frame.
    Devuelve (encoding, mensaje_error). Si encoding no es None, el registro fue exitoso
    y el frame ya quedo guardado en disco.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ubicaciones = face_recognition.face_locations(rgb_frame)

    if not ubicaciones:
        return None, "No se detecto ningun rostro, acercate o mejora la luz"
    if len(ubicaciones) > 1:
        return None, "Se detecto mas de un rostro, debe haber solo uno"

    encoding = face_recognition.face_encodings(rgb_frame, ubicaciones)[0]

    ruta = os.path.join(carpeta_rostros, f'{nombre}.jpg')
    cv2.imwrite(ruta, frame)

    return encoding, None


# ---------------------------------------------------------------------------
# Interfaz Tkinter
# ---------------------------------------------------------------------------

class VentanaPrincipal:
    def __init__(self, root, cap, estado):
        self.root = root
        self.cap = cap
        self.estado = estado

        root.protocol("WM_DELETE_WINDOW", self.cerrar)
        root.title("ESPCAMP Facial Recognition")

        self.label_video = Label(root)
        self.label_video.grid(row=0, column=0, columnspan=2)

        frame_controls = Frame(root)
        frame_controls.grid(row=1, column=1, columnspan=2, pady=12)

        Label(frame_controls, text="Ingresar nombre del usuario: ").pack(side=LEFT, padx=7)

        self.entry_nombre = Entry(frame_controls, width=20)
        self.entry_nombre.pack(side=LEFT, padx=7)

        Button(frame_controls, text="Captura Rosto", command=self.on_capturar_rostro).pack(side=LEFT, padx=7)

        self.label_estado = Label(root, text="")
        self.label_estado.grid(row=2, column=0, columnspan=2)

    def mostrar_estado(self, texto, color):
        self.label_estado.configure(text=texto, fg=color)

    def on_capturar_rostro(self):
        nombre = self.entry_nombre.get().strip()
        if not nombre:
            self.mostrar_estado("Ingresa un nombre antes de capturar", "red")
            return
        if self.estado.ultimo_frame_leido is None:
            self.mostrar_estado("Todavia no hay imagen de la camara", "red")
            return

        encoding, error = capturar_rostro(nombre, self.estado.ultimo_frame_leido, CARPETA_ROSTROS)
        if error:
            self.mostrar_estado(error, "red")
            return

        self.estado.registrar_rostro(nombre, encoding)
        self.entry_nombre.delete(0, END)
        self.mostrar_estado(f"Rostro de '{nombre}' registrado correctamente", "green")

    def actualizar_video(self):
        asegurar_camara_abierta(self.cap, URL_CAMARA)

        ret, frame = self.cap.read()
        if ret:
            frame = procesar_frame_video(frame, self.estado)
            tkimage = frame_a_imagen_tk(frame)
            self.label_video.configure(image=tkimage)
            self.label_video.image = tkimage
        else:
            self.cap.release()

        self.root.after(1, self.actualizar_video)

    def cerrar(self):
        self.cap.release()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    registro_accesos = RegistroAccesos(ARCHIVO_REGISTRO_ACCESOS, SEGUNDOS_ENTRE_REGISTROS)
    estado = EstadoApp(CARPETA_ROSTROS, registro_accesos)
    cap = abrir_camara(URL_CAMARA)

    root = Tk()
    ventana = VentanaPrincipal(root, cap, estado)

    root.after(1, ventana.actualizar_video)
    root.mainloop()


if __name__ == '__main__':
    main()
