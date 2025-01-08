import cv2
import numpy as np

def nothing(x):
    pass

def get_frame(img_name=None):
    """Return the frame either from an image or from the camera."""
    if img_name:
        return cv2.imread(img_name)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    return frame

def apply_filter(img):
    img = cv2.resize(img, None, fx = .7, fy=.7)
    imgPre = cv2.GaussianBlur(img, (7,7), 7)
    imgPre = cv2.medianBlur(imgPre, 7)
    return imgPre

H_MIN_init, S_MIN_init, V_MIN_init = 41, 58, 0
H_MAX_init, S_MAX_init, V_MAX_init = 85, 255, 255

# Create a window
cv2.namedWindow("Trackbars")
cv2.moveWindow("Trackbars", 50, 0)

# Create Trackbars for adjusting HSV ranges
cv2.createTrackbar("H Min", "Trackbars", H_MIN_init, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", S_MIN_init, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", V_MIN_init, 255, nothing)
cv2.createTrackbar("H Max", "Trackbars", H_MAX_init, 179, nothing)
cv2.createTrackbar("S Max", "Trackbars", S_MAX_init, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", V_MAX_init, 255, nothing)

# Initialize the test image
img_name = 'imagen_test.jpg'  # None for live capture
frame = get_frame(img_name)

# Create a window for displaying the combined result
cv2.namedWindow("Combined View")
cv2.moveWindow("Combined View", 50, 450)

while True:
    frame = get_frame(img_name)  # Get the current frame
    frame = apply_filter(frame)

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of Trackbars
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")

    # Define HSV range
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    # Create images for the trackbar ranges (display colors corresponding to the HSV range)
    img_lower_bgr = cv2.cvtColor(np.full((100, 225, 3), lower_hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
    img_upper_bgr = cv2.cvtColor(np.full((100, 225, 3), upper_hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)

    # Combine images for trackbars and HSV view
    hsv_image = cv2.imread('hsv_color_space.jpeg')
    combined_colors = np.hstack((img_lower_bgr, img_upper_bgr))
    combined_colors_resized = cv2.resize(combined_colors, (hsv_image.shape[1], hsv_image.shape[0]))
    final_combined = np.hstack((combined_colors_resized, hsv_image))

    # Display the combined window with the trackbars
    cv2.imshow("Trackbars", final_combined)

    # Create a mask for the selected HSV range
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Combine the original image, mask, and result
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert mask to 3 channels
    combined = np.hstack((frame, mask_colored, result))  # Stack images horizontally

    # Resize the combined image for display
    scale = 0.6  # Adjust scale factor as desired
    height, width = combined.shape[:2]
    combined_resized = cv2.resize(combined, (int(width * scale), int(height * scale)))

    # Display the combined result
    cv2.imshow("Combined View", combined_resized)

    # Exit when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()