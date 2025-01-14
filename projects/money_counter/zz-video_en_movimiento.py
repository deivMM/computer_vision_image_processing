
import cv2
import numpy as np

# Path to the video file (replace 'video.mp4' with your video file path)
video_path = '0_data/video_3.avi'

# Open the video
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
msecs = int(1000/fps)
# msecs = 1000
print(f"Frames por segundo (FPS): {fps}")

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

prev_frame = None
threshold = .01

ret, frame = cap.read()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
t_parado = 0
t_entra_script = 1
while True:
    # Read a frame
    ret, frame = cap.read()
    if frame is not None: working_f = frame.copy()

    # If the frame was not successfully read, break the loop
    if not ret:
        break
        
    working_f = cv2.resize(working_f, None, fx = .5, fy=.5)
    working_f = cv2.GaussianBlur(working_f,(11,11), 5)

    working_f = cv2.cvtColor(working_f, cv2.COLOR_BGR2GRAY)
    
    frame = working_f
    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame, working_f)
        mean_diff = np.mean(diff) / 255
        
        if mean_diff > threshold:
            cv2.putText(frame, 'En movimiento o algo', (100, 50), 0, 1, (255, 0, 0), 3) # (image, text, bottom_right, font, font_scale, color, thickness)
            t_parado = 0
        else:
            t_parado += msecs/1000
            if t_parado > t_entra_script:
                cv2.putText(frame, 'aqui entra', (100, 50), 0, 1, (255, 0, 0), 3)
            else:
                cv2.putText(frame, 'En movimiento o algo', (100, 50), 0, 1, (255, 0, 0), 3) # (image, text, bottom_right, font, font_scale, color, thickness)


    cv2.imshow('Video modificado', frame)
    
    prev_frame = working_f
    
    # Press 'q' to exit the video early
    if cv2.waitKey(msecs) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
