import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import re

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

def get_video(VidCap= 0):
    
    cap = cv2.VideoCapture(VidCap)

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
    