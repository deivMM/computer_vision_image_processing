import tkinter as tk
from tkinter import messagebox
from PIL import Image
import os

def obtener_siguiente_valor(valor, lista):
    elementos_filtrados = [x for x in lista if x.startswith(str(valor) + '_')]
    if not elementos_filtrados:
        return str(valor) + '_1'
    ultimo_numero = max([int(x.split('_')[1]) for x in elementos_filtrados])
    siguiente_valor = str(valor) + '_' + str(ultimo_numero + 1)
    return siguiente_valor

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Drawing App")

        self.canvas = tk.Canvas(self.master, width=300, height=300, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.selected_number = tk.StringVar(master)
        self.selected_number.set("0")  # Establecer el valor inicial en 0
        numbers = [str(i) for i in range(10)]  # Crear lista de números del 0 al 9
        self.number_dropdown = tk.OptionMenu(self.master, self.selected_number, *numbers)
        self.number_dropdown.pack()

        self.button_send = tk.Button(self.master, text="Guardar", command=self.send)
        self.button_send.pack(side=tk.LEFT)

        self.button_clear = tk.Button(self.master, text="Borrar", command=self.clear)
        self.button_clear.pack(side=tk.RIGHT)

        self.line_width = 20
        self.line_join_style = tk.ROUND

    def draw(self, event):
        x, y = (event.x), (event.y)
        self.canvas.create_oval(x, y, x + self.line_width, y + self.line_width, fill="black", outline="black")

    def send(self):
        numero = self.selected_number.get()
        pngs = [f.split('.')[0] for f in os.listdir('data/digits') if f.endswith('.png')]
        siguiente_valor = obtener_siguiente_valor(numero, pngs)
        filename = f"data/digits/{siguiente_valor}.png"
        self.canvas.postscript(file=filename, colormode='color')
        img = Image.open(filename)
        img.save(filename)
        self.canvas.delete("all")
        print(f'Imagen guargada: {siguiente_valor}')

    def clear(self):
        self.canvas.delete("all")
        print('Imagen borrada')

def main():
    print(os.getcwd())
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
