import cv2
import numpy as np

# Function: Apply average blur with selectable kernel size
def apply_average_blur(frame, kernel_size):
    return cv2.blur(frame, (kernel_size, kernel_size))

# Function: Apply Gaussian blur using custom kernel
def apply_gaussian_blur_custom(frame, ksize=5, sigma=1):
    # Generate 1D Gaussian kernel
    g_kernel_1d = cv2.getGaussianKernel(ksize, sigma)

    # Create 2D Gaussian kernel by outer product
    g_kernel_2d = g_kernel_1d @ g_kernel_1d.T

    # Apply convolution using filter2D
    blurred = cv2.filter2D(frame, -1, g_kernel_2d)
    return blurred

# Function: Apply sharpening with given kernel
def apply_sharpen(frame):
    kernel_sharpen = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(frame, -1, kernel_sharpen)
    return sharpened

# Main Program
cap = cv2.VideoCapture(0)

mode = 0  # 0 = normal, 1 = average blur, 2 = gaussian blur, 3 = sharpen
avg_kernel = 5  # default kernel for average blur

print("Press keys:")
print("1 = Average Blur (5x5)")
print("2 = Average Blur (9x9)")
print("3 = Gaussian Blur")
print("4 = Sharpen")
print("0 = Normal")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame based on selected mode
    if mode == 1:
        output = apply_average_blur(frame, 5)
        text = "Average Blur (5x5)"
    elif mode == 2:
        output = apply_average_blur(frame, 9)
        text = "Average Blur (9x9)"
    elif mode == 3:
        output = apply_gaussian_blur_custom(frame, ksize=9, sigma=1.5)
        text = "Gaussian Blur (Custom Kernel)"
    elif mode == 4:
        output = apply_sharpen(frame)
        text = "Sharpen Filter"
    else:
        output = frame
        text = "Normal Mode"

    # Display mode text on screen
    cv2.putText(output, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show output window
    cv2.imshow("Webcam Filter", output)

    # Keyboard control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        mode = 1  # average blur 5x5
    elif key == ord('2'):
        mode = 2  # average blur 9x9
    elif key == ord('3'):
        mode = 3  # gaussian blur custom
    elif key == ord('4'):
        mode = 4  # sharpen
    elif key == ord('0'):
        mode = 0  # normal
    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()