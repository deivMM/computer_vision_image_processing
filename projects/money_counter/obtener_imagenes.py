import os
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(script_dir, '00_data')

# [os.remove(f'{images_path}/{f}') for f in os.listdir(images_path)]

image_names = os.listdir(images_path)

if image_names:
    im_n = [int(im_nanme.split("_")[1].split(".")[0]) for im_nanme in image_names if im_nanme.endswith('jpg')]
    i = max(im_n)
else:
    i = 1

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: No se pudo leer el frame.")
        break
    
    cv2.imshow('Presiona espacio para capturar la foto', frame)

    key = cv2.waitKey(1)
    if key == 32:  # Tecla espacio
        image_n = f'image_{i}'
        frame = cv2.resize(frame, None, fx= .5, fy= .5)
        cv2.imwrite(f'{images_path}/{image_n}.jpg', frame)
        print(f"Foto guardada como '{image_n}.jpg'")
        i += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

