import cv2
import numpy as np

def nothing(x):
    pass

# Create a window
cv2.namedWindow("Trackbars")

# Create Trackbars for adjusting HSV ranges
cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)  # Hue Min
cv2.createTrackbar("S Min", "Trackbars", 255, 255, nothing)  # Saturation Min
cv2.createTrackbar("V Min", "Trackbars", 255, 255, nothing)  # Value Min
cv2.createTrackbar("H Max", "Trackbars", 100, 179, nothing)  # Hue Max
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)  # Saturation Max
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)  # Value Max

lower_hsv = np.zeros((100, 225, 3), np.uint8)
upper_hsv = np.zeros((100, 225, 3), np.uint8)

hsv_image = cv2.imread('hsv_color_space.jpeg')

while True:
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")

    lower_hsv[:] = (h_min, s_min, v_min)
    img_lower_bgr = cv2.cvtColor(lower_hsv, cv2.COLOR_HSV2BGR)

    upper_hsv[:] = (h_max, s_max, v_max)
    img_upper_bgr = cv2.cvtColor(upper_hsv, cv2.COLOR_HSV2BGR)
    
    combined_colors = np.hstack((img_lower_bgr, img_upper_bgr))
    
    combined_colors_resized = cv2.resize(combined_colors, (hsv_image.shape[1], hsv_image.shape[0]))
    final_combined = np.hstack((combined_colors_resized, hsv_image))
    
    cv2.imshow("Trackbars", final_combined)


    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
