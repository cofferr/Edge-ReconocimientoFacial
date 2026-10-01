import os
import threading
from datetime import datetime

import cv2
import numpy as np
import face_recognition
from tkinter import *
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

DISPOSITIVOS_CAMARA = {
    'Camara Local': 0,
    'Camara ESP32': 'http://10.62.82.181/320x240.mjpeg',
    'Camara Celular (DroidCam)': 'http://10.160.128.141:4747/video',
}
DISPOSITIVO_INICIAL = 'Camara Local'

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

def detectar_rostros(rgb_frame_pequeno):
    """Devuelve (ubicaciones, encodings) de los rostros en un frame RGB reducido."""
    ubicaciones = face_recognition.face_locations(rgb_frame_pequeno)
    encodings = face_recognition.face_encodings(rgb_frame_pequeno, ubicaciones)
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
    """
    Guarda el estado compartido entre el hilo principal (dibuja el video y
    atiende la UI) y el hilo de reconocimiento (calcula ubicaciones/nombres
    en segundo plano para no congelar Tkinter).
    """

    def __init__(self, carpeta_rostros, registro_accesos):
        self.nombres_conocidos, self.encodings_conocidos = cargar_rostros_conocidos(carpeta_rostros)
        self.registro_accesos = registro_accesos
        self.ubicaciones_rostros = []
        self.nombres_rostros = []
        self.ultimo_frame_leido = None

        self._lock = threading.Lock()
        self._reconociendo = False

    def registrar_rostro(self, nombre, encoding):
        with self._lock:
            self.nombres_conocidos.append(nombre)
            self.encodings_conocidos.append(encoding)

    def obtener_deteccion_actual(self):
        with self._lock:
            return list(self.ubicaciones_rostros), list(self.nombres_rostros)

    def _actualizar_deteccion(self, ubicaciones, nombres):
        with self._lock:
            self.ubicaciones_rostros = ubicaciones
            self.nombres_rostros = nombres
        for nombre in nombres:
            self.registro_accesos.registrar(nombre)

    def lanzar_reconocimiento_si_libre(self, frame_pequeno):
        """Lanza el calculo de reconocimiento en un hilo aparte, solo si no hay
        uno corriendo ya (evita acumular hilos si el calculo es mas lento que
        la tasa de frames)."""
        with self._lock:
            if self._reconociendo:
                return
            self._reconociendo = True
            nombres_conocidos = list(self.nombres_conocidos)
            encodings_conocidos = list(self.encodings_conocidos)

        hilo = threading.Thread(
            target=self._tarea_reconocimiento,
            args=(frame_pequeno, nombres_conocidos, encodings_conocidos),
            daemon=True,
        )
        hilo.start()

    def _tarea_reconocimiento(self, frame_pequeno, nombres_conocidos, encodings_conocidos):
        ubicaciones, encodings = detectar_rostros(frame_pequeno)
        nombres = identificar_rostros(encodings, nombres_conocidos, encodings_conocidos)
        self._actualizar_deteccion(ubicaciones, nombres)
        with self._lock:
            self._reconociendo = False


# ---------------------------------------------------------------------------
# Logica de camara / video
# ---------------------------------------------------------------------------

def es_fuente_local(fuente):
    """Las camaras locales se identifican por indice (int); las de red, por URL (str)."""
    return isinstance(fuente, int)


def abrir_camara(fuente):
    if es_fuente_local(fuente):
        cap = cv2.VideoCapture(fuente)
    else:
        cap = cv2.VideoCapture(fuente, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def asegurar_camara_abierta(cap, fuente):
    if cap.isOpened():
        return
    if es_fuente_local(fuente):
        cap.open(fuente)
    else:
        cap.open(fuente, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def procesar_frame_video(frame, estado):
    """
    Lanza el reconocimiento facial en segundo plano (si no hay uno corriendo) y
    dibuja sobre el frame la ultima deteccion disponible. Nunca bloquea el
    hilo que la llama: el calculo pesado ocurre en un hilo aparte.
    """
    estado.ultimo_frame_leido = frame.copy()

    frame_pequeno = cv2.resize(frame, (0, 0), fx=FACTOR_REDUCCION, fy=FACTOR_REDUCCION)
    rgb_frame_pequeno = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)
    estado.lanzar_reconocimiento_si_libre(rgb_frame_pequeno)

    ubicaciones, nombres = estado.obtener_deteccion_actual()
    dibujar_rostros(frame, ubicaciones, nombres)
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
    def __init__(self, root, estado, dispositivo_inicial):
        self.root = root
        self.estado = estado
        self.dispositivo_actual = StringVar(value=dispositivo_inicial)
        self.cap = abrir_camara(DISPOSITIVOS_CAMARA[dispositivo_inicial])

        root.protocol("WM_DELETE_WINDOW", self.cerrar)
        root.title("ESPCAMP Facial Recognition")

        self.label_video = Label(root)
        self.label_video.grid(row=0, column=0, columnspan=2)

        frame_dispositivo = Frame(root)
        frame_dispositivo.grid(row=1, column=0, columnspan=2, pady=(12, 0))

        Label(frame_dispositivo, text="Dispositivo de camara: ").pack(side=LEFT, padx=7)

        self.botones_dispositivo = {}
        for nombre_dispositivo in DISPOSITIVOS_CAMARA:
            boton = Button(
                frame_dispositivo, text=nombre_dispositivo,
                command=lambda n=nombre_dispositivo: self.on_cambiar_dispositivo(n),
            )
            boton.pack(side=LEFT, padx=4)
            self.botones_dispositivo[nombre_dispositivo] = boton

        self._resaltar_boton_dispositivo(dispositivo_inicial)

        frame_controls = Frame(root)
        frame_controls.grid(row=2, column=0, columnspan=2, pady=12)

        Label(frame_controls, text="Ingresar nombre del usuario: ").pack(side=LEFT, padx=7)

        self.entry_nombre = Entry(frame_controls, width=20)
        self.entry_nombre.pack(side=LEFT, padx=7)

        self.btn_capturar = Button(frame_controls, text="Captura Rosto", command=self.on_capturar_rostro)
        self.btn_capturar.pack(side=LEFT, padx=7)

        self.label_estado = Label(root, text="")
        self.label_estado.grid(row=3, column=0, columnspan=2)

    def mostrar_estado(self, texto, color):
        self.label_estado.configure(text=texto, fg=color)

    def _resaltar_boton_dispositivo(self, nombre_dispositivo):
        for nombre, boton in self.botones_dispositivo.items():
            if nombre == nombre_dispositivo:
                boton.configure(relief=SUNKEN, bg="#cfe8ff")
            else:
                boton.configure(relief=RAISED, bg="SystemButtonFace")

    def on_cambiar_dispositivo(self, nombre_dispositivo):
        self.dispositivo_actual.set(nombre_dispositivo)
        self._resaltar_boton_dispositivo(nombre_dispositivo)

        self.cap.release()
        self.cap = abrir_camara(DISPOSITIVOS_CAMARA[nombre_dispositivo])
        self.estado.ultimo_frame_leido = None
        self.mostrar_estado(f"Conectando a '{nombre_dispositivo}'...", "black")

    def on_capturar_rostro(self):
        nombre = self.entry_nombre.get().strip()
        if not nombre:
            self.mostrar_estado("Ingresa un nombre antes de capturar", "red")
            return
        if self.estado.ultimo_frame_leido is None:
            self.mostrar_estado("Todavia no hay imagen de la camara", "red")
            return

        frame_a_procesar = self.estado.ultimo_frame_leido.copy()
        self.btn_capturar.configure(state=DISABLED)
        self.mostrar_estado("Procesando...", "black")

        threading.Thread(
            target=self._tarea_capturar_rostro,
            args=(nombre, frame_a_procesar),
            daemon=True,
        ).start()

    def _tarea_capturar_rostro(self, nombre, frame):
        encoding, error = capturar_rostro(nombre, frame, CARPETA_ROSTROS)
        self.root.after(0, self._finalizar_captura_rostro, nombre, encoding, error)

    def _finalizar_captura_rostro(self, nombre, encoding, error):
        self.btn_capturar.configure(state=NORMAL)

        if error:
            self.mostrar_estado(error, "red")
            return

        self.estado.registrar_rostro(nombre, encoding)
        self.entry_nombre.delete(0, END)
        self.mostrar_estado(f"Rostro de '{nombre}' registrado correctamente", "green")

    def actualizar_video(self):
        fuente_actual = DISPOSITIVOS_CAMARA[self.dispositivo_actual.get()]
        asegurar_camara_abierta(self.cap, fuente_actual)

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

    root = Tk()
    ventana = VentanaPrincipal(root, estado, DISPOSITIVO_INICIAL)

    root.after(1, ventana.actualizar_video)
    root.mainloop()


if __name__ == '__main__':
    main()
