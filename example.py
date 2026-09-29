import cv2
import sys
from tkinter import *
from PIL import Image, ImageTk

url = 'http://10.140.228.181/640x480.jpg'
cap = cv2.VideoCapture(url)

def on_closing():
    print("Camara desconectada")
    root.destroy()

def callback(url):
    cap.open(url)
    ret, frame = cap.read()

    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((400, 400))

        tkimage = ImageTk.PhotoImage(img)
        label.configure(image=tkimage)
        label.image = tkimage

        root.after(1, lambda: callback(url))
    else:
        on_closing()

root = Tk()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.title("Probando ESP32 CAM")

label = Label(root)
label.grid(row=0)

if not cap.isOpened():
    sys.exit("Error de conexion")

callback(url)
root.mainloop()
