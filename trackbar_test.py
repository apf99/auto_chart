import cv2
import numpy as np

def nothing(x):
    pass


# Create a window
cv2.namedWindow('Trackbar Test')

# Create a trackbar
cv2.createTrackbar('Test', 'Trackbar Test', 0, 255, nothing)

while True:
    # Get current position of the trackbar
    value = cv2.getTrackbarPos('Test', 'Trackbar Test')
    print(f'Trackbar Value: {value}')

    # Display a black image as a placeholder
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.imshow('Trackbar Test', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
