import cv2
import numpy as np

# Define HSV color range for GREEN
lower_green = np.array([40, 50, 50])    # lower bound for green
upper_green = np.array([85, 255, 255])  # upper bound for green

# Create kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply color thresholding for green
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean the mask (Opening → remove noise, Closing → fill holes)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # Find contours and trigger an action if green object detected
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)

        # Consider only large enough green objects
        if area > 1500:
            green_detected = True

            # Draw bounding box around detected green object
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Trigger action → display message
            cv2.putText(frame, "Green Object Detected!", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            break

    if not green_detected:
        cv2.putText(frame, "No green object detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display windows
    cv2.imshow("Webcam Frame", frame)
    cv2.imshow("Green Mask (Clean)", mask_clean)

    # Exit with ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()