import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

def find_darkest_region(image, a, b):
    min_intensity = float('inf')
    darkest_rect = None
    
    height, width = image.shape[:2]
    
    for y in range(height):
        for x in range(width):
            x1 = max(0, x - a // 2)
            y1 = max(0, y - b // 2)
            x2 = min(width, x + a // 2)
            y2 = min(height, y + b // 2)
            
            region = image[y1:y2, x1:x2]
            average_intensity = np.mean(region)
            
            if average_intensity < min_intensity:
                min_intensity = average_intensity
                darkest_rect = (x1, y1)
                
    return darkest_rect

def dibujar_rectangulo(image, coordenada, dimensiones_a, dimensiones_b):
    cv2.rectangle(image, coordenada, (coordenada[0]+dimensiones_a, coordenada[1]+dimensiones_b), (0, 0, 255), 2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')  # Turn off axis
    plt.show()

image = cv2.imread('pop.png')
darkest_rect = find_darkest_region(image, 100, 50)
dibujar_rectangulo(image, darkest_rect, 100, 50)


def generate_data(x_mean, y_mean, x_std, y_std):
    data_1 = np.column_stack((np.random.normal(loc=x_mean, scale=x_std, size=1000)
                        ,np.random.normal(loc=y_mean, scale=y_std, size=1000)))
    data_2 = np.random.uniform(low=10.0, high=0, size=(500, 2))
    return np.concatenate([data_1, data_2])

stds = np.random.uniform(0.7, 1.1, size=(20, 2))

for n, std in enumerate(stds):
    data = generate_data(7.5, 7.5, std[0], std[1])
    f, ax = plt.subplots(figsize=(6,6))
    ax.scatter(data[:,0], data[:,1], s =1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(),f'projects\darker_region_in_a_image\data\image_{n+1}.png'))



