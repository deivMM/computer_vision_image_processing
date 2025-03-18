import os
import re
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy import stats

# Funciones
####
# hue_promedio_anillo
# generate_design
# medir_tiempo
# imagen_con_texto
# visualizar_color
# get_color_from_circle
# sort_list_of_fs_by_ascending_number
# get_images
# get_video
####

def hue_promedio_anillo(img, x, y, radio_ext, grosor, visualizar=False):
    # Convertir la imagen a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Crear una máscara para el círculo exterior
    mask_ext = np.zeros_like(hsv[:, :, 0], dtype=np.uint8)
    cv2.circle(mask_ext, (x, y), radio_ext, 255, thickness=-1)
    
    # Crear una máscara para el círculo interior
    mask_int = np.zeros_like(hsv[:, :, 0], dtype=np.uint8)
    cv2.circle(mask_int, (x, y), radio_ext - grosor, 255, thickness=-1)
    
    # Obtener la máscara del anillo restando la interior a la exterior
    mask_ring = cv2.bitwise_xor(mask_ext, mask_int)

    # Extraer los valores de hue de los píxeles dentro del anillo
    hues = hsv[:, :, 0][mask_ring == 255]

    # Calcular el hue promedio
    # hue_promedio = np.mean(hues) if hues.size > 0 else None
    hue_promedio = stats.mode(hues, keepdims=True) if hues.size > 0 else None
    
    # Visualizar la imagen con el anillo resaltado
    if visualizar:
        img_viz = img.copy()
        img_viz[mask_ring == 255] = [0, 0, 255]  # Pintar el anillo en rojo
        f, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.show()
    
    return hue_promedio


def generate_design(params, rename_index=False):
    """
    Genera un diseño de experimentos con todas las combinaciones posibles.
    ejemplo: generate_design({'dp':[.5, 1.5, 11, 1], 'param2':[5, 25, 6,0]})

    Args:
        params (dict): Diccionario donde las claves son los nombres de los parámetros y
                      los valores son listas de la forma [inicio, fin, num_puntos, sensibilidad].
        rename_index (bool): Si es True, redefine los índices como Model_1, Model_2, ...

    Returns:
        pd.DataFrame: DataFrame con todas las combinaciones posibles de valores.
    """
    param_values = {
        key: np.round(np.linspace(value[0], value[1], value[2]), value[3])
        for key, value in params.items()
    }

    design = pd.DataFrame(
        [row for row in itertools.product(*param_values.values())],
        columns=param_values.keys()
    )
    
    if rename_index:
        design.index = [f"Model_{i+1}" for i in range(len(design))]
    else:
        design.index = range(1, len(design)+1)

    return design

def medir_tiempo(func, print_info = True):
    def wrapper(*args, **kwargs):
        inicio = time.perf_counter()
        resultado = func(*args, **kwargs)
        fin = time.perf_counter()
        duracion = fin - inicio
        if print_info:
            if duracion < 60:
                print(f"Tiempo de ejecución de {func.__name__}: {duracion:.6f} segundos")
            elif duracion < 3600:
                minutos = int(duracion // 60)
                segundos = int(duracion % 60)
                print(f"Tiempo de ejecución de {func.__name__}: {minutos} minutos y {segundos} segundos")
            else:
                horas = int(duracion // 3600)
                minutos = int((duracion % 3600) // 60)
                print(f"Tiempo de ejecución de {func.__name__}: {horas} horas y {minutos} minutos")
        return resultado, duracion
    return wrapper


def imagen_con_texto(img, text, posicion= [[0, 0], [.3, .1]]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    img_h, img_w = img.shape[0], img.shape[1]
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    rect_x1, rect_y1 = int(img_w * posicion[0][0]), int(img_h * posicion[0][1])
    rect_x2, rect_y2 = int(img_w * posicion[1][0]), int(img_h * posicion[1][1])
    
    text_x_pos = ((rect_x2-rect_x1) - text_width) // 2 + int(rect_x1)
    text_y_pos = ((rect_y2-rect_y1) + text_height) // 2 + int(rect_y1)
    
    overlay = img.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (100, 100, 100), -1) 

    alpha = 0.5
    img_blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    cv2.putText(img_blended, text, (text_x_pos, text_y_pos), font, font_scale , (0, 0, 0), thickness)

    return img_blended

def visualizar_color(color, formato='RGB'):
    """
    Visualiza el color dado en el formato especificado.
    
    Args:
        color (tuple): Tupla con los valores de color.
        formato (str): Formato de color ('RGB' o 'BGR'). Por defecto es 'RGB'.
    """
    color = tuple(int(x) for x in color)

    if formato == 'BGR':
        # Convertir BGR a RGB
        color_01 = (color[2] / 255, color[1] / 255, color[0] / 255)
    elif formato == 'RGB':
        color_01 = (color[0] / 255, color[1] / 255, color[2] / 255)
    else:
        raise ValueError("Formato no reconocido. Usa 'RGB' o 'BGR'.")

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_facecolor(color_01)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Color: {color} | {formato}', fontsize=10)
    plt.show()

def get_color_from_circle(img, center, radius, vis= False):
    """
    Computes the average color within a circular region of an image and optionally visualizes it.

    Parameters:
        img (numpy.ndarray): The input image in BGR format.
        center (tuple): The (x, y) coordinates of the circle's center.
        radius (int): The radius of the circle in pixels.
        vis (bool): If True, visualizes the circle and the average color on the image (default is False).
    Returns:
        tuple: A 3-tuple representing the average color within the circle in (B, G, R) format.
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    mean_color = cv2.mean(masked_image, mask=mask)
    mean_color = tuple(int(x) for x in mean_color)[:3]
    if vis:
        f, ax = plt.subplots(figsize=(6, 6))
        cv2.circle(img, center, radius, (0, 255, 0), 2)  # Color verde para el círculo, grosor 2
        cv2.circle(img, center, 20, (0, 0, 0), 10)
        cv2.circle(img, center, 20, (mean_color[0], mean_color[1], mean_color[2]), -1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.show()
    return mean_color

def sort_list_of_fs_by_ascending_number(list_of_fs, r_pattern = ''):
    '''
    This function sorts a list of files/words by ascending order 
    E.g.(1): sort_list_of_fs_by_ascending_number(['Model_10.inp','Model_1.inp'])
    E.g.(2): sort_list_of_fs_by_ascending_number(['t_1 Model_10.inp','t_2 Model_1.inp'], 'Model_')
    ----------
    list_of_fs: [list of files/words]
    r_pattern: [str/regex] | regex pattern | Def. arg.: ''
    ----------
    the function modifies the list
    '''
    list_of_fs.sort(key=lambda el:int(re.search(f'{r_pattern}(\d+)',el).group(1)))


def get_images(folder_name, VidCap= 0, prefix_name = None):
    script_dir = os.getcwd()

    images_path = os.path.join(script_dir, folder_name)
    image_names = [f for f in os.listdir(images_path) if f.endswith('jpg') and (f.startswith(prefix_name) if prefix_name else f.startswith('image_'))]

    if image_names:
        sort_list_of_fs_by_ascending_number(image_names)
        im_nun = int(image_names[-1].split("_")[-1].split(".")[0])
        i = im_nun+1
    else:
        i = 1
    
    cap = cv2.VideoCapture(VidCap)
    time.sleep(2)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        exit()
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: No se pudo leer el frame.")
            break
        
        cv2.imshow('Presiona espacio para capturar la foto', frame)

        key = cv2.waitKey(2)
        if key == 32:  # Tecla espacio
            if prefix_name is not None:
                image_n = f'{prefix_name}_image_{i}'
            else:
                image_n = f'image_{i}'
            frame = cv2.resize(frame, None, fx= .5, fy= .5)
            cv2.imwrite(f'{images_path}/{image_n}.jpg', frame)
            print(f"Foto guardada como '{image_n}.jpg'")
            i += 1

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
def get_video(folder_name, VidCap = 0):
    script_dir = os.getcwd()
    video_path = os.path.join(script_dir, folder_name)

    videos_names = [f for f in os.listdir(video_path) if f.endswith('avi')]

    if videos_names:
        im_n = [int(vd_nanme.split("_")[1].split(".")[0]) for vd_nanme in videos_names]
        i = max(im_n)+1
    else:
        i = 1

    cap = cv2.VideoCapture(VidCap)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        exit()

    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec
    fps = 20.0  # Frames per second
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(f'{video_path}/video_{i}.avi', fourcc, fps, frame_size)

    print(f"Grabando video a resolución: {frame_size}, FPS: {fps}")
    print("Presiona 'q' para salir y guardar el video.")

    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Error al capturar el frame. Finalizando...")
            break

        # Write the frame into the file
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('Grabando', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    