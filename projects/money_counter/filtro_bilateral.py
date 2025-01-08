import cv2
import numpy as np
import os
from glob import glob

# def update_blur(val):
#     if val % 2 == 0:
#         print('El kernel debe ser impar')
#         val += 1
#     blurred = cv2.GaussianBlur(image, (val, val), 0)
#     cv2.imshow('Gaussian Blur', blurred)

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_files = glob('training/*jpg')
except NameError:
    script_dir = os.getcwd()
    img_files = glob('training/*jpg')

# Carga la imagen
image_path_number= 10  # Cambia esta ruta a la imagen que quieres usar
image = cv2.imread(img_files[image_path_number])

def empty(x):
    pass

# Crear una ventana
cv2.namedWindow('Filtro Bilateral')

# Crear trackbars para los parámetros del filtro
cv2.createTrackbar('d', 'Filtro Bilateral', 19, 20, empty)  # Diámetro del área
cv2.createTrackbar('SigmaColor', 'Filtro Bilateral', 44, 200, empty)  # Desviación en color
cv2.createTrackbar('SigmaSpace', 'Filtro Bilateral', 193, 200, empty)  # Desviación en coordenadas espaciales

while True:
    # Leer los valores de los trackbars
    d = cv2.getTrackbarPos('d', 'Filtro Bilateral')
    sigma_color = cv2.getTrackbarPos('SigmaColor', 'Filtro Bilateral')
    sigma_space = cv2.getTrackbarPos('SigmaSpace', 'Filtro Bilateral')
    
    # Asegurarse de que d sea al menos 1 (evitar errores)
    d = max(1, d)

    # Aplicar el filtro bilateral
    # image = cv2.GaussianBlur(image, (3, 3), 1)
    image = cv2.medianBlur(image, 3)
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    # Mostrar la imagen filtrada
    cv2.imshow('Filtro Bilateral', filtered_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar todas las ventanas
cv2.destroyAllWindows()




