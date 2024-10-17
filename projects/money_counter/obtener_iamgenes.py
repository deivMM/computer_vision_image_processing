import os
os.chdir('C:/Users/dmarquez/OneDrive - Lortek S.Coop/Escritorio/David/digital/00_main/IA/projects/money_counter')
import cv2
[os.remove(f'imagenes/{f}') for f in os.listdir('imagenes')]

i = 1
cap = cv2.VideoCapture(0)

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
        cv2.imwrite(f'imagenes/{image_n}.jpg', frame)
        print(f"Foto guardada como '{image_n}.jpg'")
        i += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


