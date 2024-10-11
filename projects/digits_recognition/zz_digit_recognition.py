import tkinter as tk
from tkinter import messagebox
from PIL import Image
import os
import io
import joblib
import sklearn

def get_image_vect(img):
    imagen_redimensionada = img.resize((30, 30))
    imagen_escala_de_grises = imagen_redimensionada.convert("L")
    datos_pixeles = list(imagen_escala_de_grises.getdata())
    datos_pixeles_normalizados = [int(pixel / 255 * 15) for pixel in datos_pixeles]
    return datos_pixeles_normalizados

def comprobar_digito(img):
    loaded_model = joblib.load('digit_recognition_model.pkl')
    image_vect = get_image_vect(img)
    predicted_val = loaded_model.predict([image_vect])[0]
    return predicted_val

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Comprobacion numero")

        self.canvas = tk.Canvas(self.master, width=300, height=300, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.button_send = tk.Button(self.master, text="Comprobar", command=self.comprobar)
        self.button_send.pack(side=tk.LEFT)

        self.line_width = 20
        self.line_join_style = tk.ROUND

    def draw(self, event):
        x, y = (event.x), (event.y)
        self.canvas.create_oval(x, y, x + self.line_width, y + self.line_width, fill="black", outline="black")

    def comprobar(self):
        ps = self.canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        predicted_val = comprobar_digito(img)
        self.canvas.delete("all")
        respuesta = messagebox.askquestion("Confirmación",f"¿El numero escrito es un: {predicted_val}?")

def main():
    print(os.getcwd())
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()