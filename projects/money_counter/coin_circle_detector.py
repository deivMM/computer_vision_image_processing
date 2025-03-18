import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def coins_visualization(img, circulos, text=None):
    '''
    circulos = np.array([[202, 128,  35], [221,  43,  34], [157,  73,  35], [89, 105,  36]])
    '''
    
    img_ = img.copy()
    for i in circulos:
        cv2.circle(img_, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img_, (i[0], i[1]), 1, (0, 0, 255), 3)
        
    f, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))

    ax.axis('off')
    if text: plt.title(text)
    plt.show()

def image_prepros(img):
    ####
    # imgPre = cv2.GaussianBlur(img, (3,3), 7)
    # imgPre = cv2.bilateralFilter(img, 9, 75, 75)
    # imgPre = cv2.medianBlur(imgPre, 3)
    ####
    
    imgPre = cv2.resize(img, None, fx = .5, fy=.5)    
    
    imgPre = cv2.bilateralFilter(imgPre, 9, 75, 75)
    imgPre = cv2.GaussianBlur(imgPre, (5,5), 5)
    
    return imgPre

def coins_detec_alg(image_path, vis_result=True, vis_proc = False, params=None):
    default_params = {
        "dp": 1.1, # Inverso de la resolución acumuladora
        "minDist": 40, # Distancia mínima entre centros de círculos || 80
        "param1": 100, # Umbral del detector de bordes Canny
        "param2": 12, # Umbral del acumulador (más bajo = más círculos detectados)
        "minRadius": 20, # Radio mínimo de los círculos
        "maxRadius": 60 # Radio máximo de los círculos
    }
    
    if params is not None:
        default_params.update(params)  # Sobreescribe los valores por los proporcionados    
    
    img = cv2.imread(image_path)
    if vis_proc:
        f, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.title('original')
        plt.show()
        
    imgPre = image_prepros(img)
    img = cv2.resize(img, None, fx = .5, fy=.5)
    
    if vis_proc:
        f, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(cv2.cvtColor(imgPre, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.title('filtro gaussiano y filtro_mediana')
        plt.show()

    hsv_image = cv2.cvtColor(imgPre, cv2.COLOR_BGR2HSV)
        
    lower_green = np.array([45, 58, 0])  # Límite inferior (H, S, V)
    upper_green = np.array([85, 255, 255])  # Límite superior (H, S, V)

    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    inverse_mask = cv2.bitwise_not(mask)

    kernel = np.ones((3,3),np.uint8)
    inverse_mask = cv2.morphologyEx(inverse_mask,cv2.MORPH_OPEN,kernel, iterations = 4)

    result = cv2.bitwise_and(img, img, mask=inverse_mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    if vis_proc:
        f, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.title('mascara con imagen')
        plt.show()
    if vis_proc:
        f, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(cv2.cvtColor(inverse_mask, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.title('mascara')
        plt.show()

    circulos = cv2.HoughCircles(
    inverse_mask,
    cv2.HOUGH_GRADIENT,
    **default_params
    )
    
    if circulos is not None:
        circulos = np.round(circulos[0]).astype(int)
        r_sum, n_circulos = circulos[:,2].sum(), circulos.shape[0]
        if vis_result:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            coins_visualization(img, circulos, f'{image_name} | R_sum: {r_sum} | Nº coins: {n_circulos}')
        return r_sum, n_circulos, circulos
    else: return None, None, None
