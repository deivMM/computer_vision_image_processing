import os
import cv2

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

video_path = os.path.join(script_dir, '00_data')

print("Script Directory:", script_dir)


videos_names = [f for f in os.listdir(video_path) if f.endswith('avi')]

if videos_names:
    im_n = [int(vd_nanme.split("_")[1].split(".")[0]) for vd_nanme in videos_names]
    i = max(im_n)+1
else:
    i = 1

cap = cv2.VideoCapture(1)

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

