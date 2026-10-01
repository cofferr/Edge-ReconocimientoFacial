# Congelamiento de la UI al reconocer/registrar rostros

## Diagnostico

`facial_recognition.py` corre todo en un solo hilo: el mismo hilo que maneja
la ventana de Tkinter (dibujar, mover, procesar clics) es el que llama a
`face_recognition.face_locations()` y `face_recognition.face_encodings()`.

Estas dos funciones son costosas en CPU (el detector HOG de dlib recorre la
imagen completa buscando rostros, y el encoding corre una red neuronal
pequena por cada rostro encontrado). Mientras estan calculando, Tkinter no
puede:
- Redibujar la ventana (por eso se ve "congelada").
- Procesar el clic en el boton "Captura Rostro" (el evento queda en cola y
  se ejecuta tarde, a veces capturando un frame ya desactualizado).

Esto ocurre en dos lugares del codigo:
- `callback()` (video en vivo): cada vez que `procesar_frame` es `True`,
  llama a `face_locations` + `face_encodings` sobre el frame reducido.
- `captureFace()` (boton de registro): llama a `face_locations` +
  `face_encodings` sobre el frame a resolucion completa (mas lento aun que
  en el video, porque no se reduce la imagen).

No es una limitacion irreversible del hardware — es que todo el trabajo
pesado compite por el mismo hilo que necesita la interfaz para responder.
El hardware (CPU sin GPU para dlib) determina *cuanto* tarda cada calculo,
pero la arquitectura de un solo hilo es la que convierte esa demora en un
freeze visible.

## Solucion propuesta: mover el calculo a un hilo en segundo plano

Idea general: usar el modulo `threading` para correr
`face_locations`/`face_encodings` en un hilo aparte, y que el hilo principal
de Tkinter solo lea los resultados ya calculados (sin esperar a que
terminen). Se necesita un `Lock` (o variables simples, dado que el GIL de
Python ya evita condiciones de carrera en asignaciones simples) para pasar
los resultados del hilo de trabajo al hilo principal sin corromper datos a
medio escribir.

### Para el video en vivo (`callback`)

En vez de calcular el reconocimiento dentro del propio `callback()`
(bloqueando el loop de `root.after`), se lanza un hilo que:
1. Toma una copia del frame reducido.
2. Calcula `face_locations` + `face_encodings` + comparacion con rostros
   conocidos.
3. Al terminar, guarda los resultados en variables compartidas
   (`loc_rostros`, `nombre_rostros`) protegidas por un `Lock`.

El `callback()` sigue corriendo cada pocos milisegundos (dibujando el ultimo
frame y los ultimos resultados disponibles), sin esperar a que el hilo de
reconocimiento termine. Si el hilo de reconocimiento tarda mas que un
frame de video, simplemente se siguen mostrando las ultimas coordenadas
conocidas hasta que el hilo entregue las nuevas (igual que ya hace
`procesar_frame` al alternar frames, pero sin bloquear).

Esqueleto:

```python
import threading

lock_resultados = threading.Lock()
hilo_reconociendo = False

def tarea_reconocimiento(rgb_frame_pequeno):
    global loc_rostros, nombre_rostros, hilo_reconociendo

    nuevas_loc = face_recognition.face_locations(rgb_frame_pequeno)
    nuevos_encodings = face_recognition.face_encodings(rgb_frame_pequeno, nuevas_loc)

    nuevos_nombres = []
    for encoding_rostro in nuevos_encodings:
        nombre_rostro = 'Rostro Desconocido'
        if len(rostros_conocidos_encoding) > 0:
            coincidencias = face_recognition.compare_faces(rostros_conocidos_encoding, encoding_rostro, tolerance=0.6)
            distancia = face_recognition.face_distance(rostros_conocidos_encoding, encoding_rostro)
            mejor_coincidencia = np.argmin(distancia)
            if coincidencias[mejor_coincidencia]:
                nombre_rostro = rostros_conocidos_nombres[mejor_coincidencia]
        nuevos_nombres.append(nombre_rostro)

    with lock_resultados:
        loc_rostros = nuevas_loc
        nombre_rostros = nuevos_nombres
        hilo_reconociendo = False

def callback():
    global hilo_reconociendo
    ...
    ret, frame = cap.read()
    if ret:
        ...
        if not hilo_reconociendo:
            hilo_reconociendo = True
            threading.Thread(
                target=tarea_reconocimiento,
                args=(rgb_frame_pequeno,),
                daemon=True,
            ).start()

        with lock_resultados:
            loc_rostros_actual = list(loc_rostros)
            nombre_rostros_actual = list(nombre_rostros)

        # dibujar con loc_rostros_actual / nombre_rostros_actual
        ...
        root.after(1, callback)
```

Puntos clave:
- `daemon=True` para que el hilo no impida cerrar la app.
- La bandera `hilo_reconociendo` evita lanzar un hilo nuevo mientras el
  anterior sigue corriendo (si no, se acumularian hilos si el reconocimiento
  es mas lento que la tasa de frames).
- Se copian `loc_rostros`/`nombre_rostros` dentro del `lock` antes de usarlos
  para dibujar, evitando leer una lista a medio modificar.

### Para el registro de rostro (`captureFace`)

Mismo patron: en vez de bloquear el callback del boton, lanzar un hilo que
haga la deteccion + guardado, y que al terminar actualice la UI (el
`lbl_estado` y las listas de conocidos) usando `root.after(0, ...)` para
volver a tocar los widgets de Tkinter desde el hilo principal (Tkinter no es
thread-safe: los widgets solo se deben tocar desde el hilo principal).

```python
def captureFace():
    nombre = tbx1.get().strip()
    if not nombre:
        lbl_estado.configure(text="Ingresa un nombre antes de capturar", fg="red")
        return
    if ultimo_frame_leido is None:
        lbl_estado.configure(text="Todavia no hay imagen de la camara", fg="red")
        return

    btn1.configure(state=DISABLED)
    lbl_estado.configure(text="Procesando...", fg="black")

    frame_a_procesar = ultimo_frame_leido.copy()
    threading.Thread(
        target=tarea_captura,
        args=(nombre, frame_a_procesar),
        daemon=True,
    ).start()

def tarea_captura(nombre, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ubicaciones = face_recognition.face_locations(rgb_frame)

    if not ubicaciones:
        root.after(0, lambda: finalizar_captura_error("No se detecto ningun rostro, acercate o mejora la luz"))
        return
    if len(ubicaciones) > 1:
        root.after(0, lambda: finalizar_captura_error("Se detecto mas de un rostro, debe haber solo uno"))
        return

    encoding_rostro = face_recognition.face_encodings(rgb_frame, ubicaciones)[0]
    ruta = os.path.join(CARPETA_ROSTROS, f'{nombre}.jpg')
    cv2.imwrite(ruta, frame)

    root.after(0, lambda: finalizar_captura_exito(nombre, encoding_rostro))

def finalizar_captura_error(mensaje):
    lbl_estado.configure(text=mensaje, fg="red")
    btn1.configure(state=NORMAL)

def finalizar_captura_exito(nombre, encoding_rostro):
    rostros_conocidos_nombres.append(nombre)
    rostros_conocidos_encoding.append(encoding_rostro)
    tbx1.delete(0, END)
    lbl_estado.configure(text=f"Rostro de '{nombre}' registrado correctamente", fg="green")
    btn1.configure(state=NORMAL)
```

Puntos clave:
- Se deshabilita el boton (`btn1.configure(state=DISABLED)`) mientras se
  procesa, para que el usuario vea que algo esta pasando y no haga doble
  clic.
- Todo lo que toca widgets de Tkinter (`lbl_estado`, `tbx1`, `btn1`) se
  ejecuta siempre a traves de `root.after(0, ...)`, nunca directamente desde
  el hilo de `tarea_captura` — esa es la regla de oro de Tkinter con hilos.
- `rostros_conocidos_nombres`/`rostros_conocidos_encoding` solo se modifican
  desde el hilo principal (dentro de `finalizar_captura_exito`), evitando
  tener que proteger esas listas con un lock aparte.

## Alternativa mas simple (si no se quiere usar threading)

Si se prefiere no anadir concurrencia, se puede reducir directamente el
costo computacional:
- Reducir aun mas el factor de `cv2.resize` (por ejemplo `fx=0.15` en vez de
  `0.25`) para que el frame analizado sea mas pequeno y rapido de procesar.
- Procesar el reconocimiento cada 3-4 frames en vez de cada 2
  (`procesar_frame` ya hace esto cada 2; se puede usar un contador en vez de
  un booleano para espaciarlo mas).
- Usar `face_recognition.face_locations(..., model="hog")` explicitamente
  (ya es el default, pero evitando `model="cnn"` que es mucho mas lento sin
  GPU).

Estas opciones bajan la duracion del freeze pero no lo eliminan: mientras el
calculo siga en el hilo principal, cualquier reduccion solo lo hace mas
corto, no desaparece. El threading es la unica forma de que la UI deje de
congelarse por completo durante el calculo.
