import cv2
import numpy as np
import matplotlib.pyplot as plt

# List to store points
points = []

# Mouse callback function to capture points
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('image', image)

# Load the original image
image_path = 'Flooring.jpg'
image = cv2.imread(image_path)
original_image = image.copy()

# Display the image to capture the polygon points
cv2.imshow('image', image)
cv2.setMouseCallback('image', mouse_callback)

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('d'):
        if points:
            points.pop()
            image = original_image.copy()
            for point in points:
                cv2.circle(image, point, 5, (0, 0, 255), -1)
            cv2.imshow('image', image)
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

# Ensure the polygon is closed
if len(points) > 2 and points[0] != points[-1]:
    points.append(points[0])

# Create a mask from the polygon points
polygon_mask = np.zeros_like(original_image[:, :, 0])
points_array = None
if len(points) > 0:
    points_array = np.array([points], dtype=np.int32)
    cv2.fillPoly(polygon_mask, points_array, 255)

# Invert the polygon mask to keep only the area outside the polygon
polygon_mask_inv = cv2.bitwise_not(polygon_mask)

# Mask the original image to keep only the area outside the polygon
masked_image = cv2.bitwise_and(original_image, original_image, mask=polygon_mask_inv)

# Convert the masked image to grayscale
gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to find edges
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Define kernels to detect diagonal lines (e.g., 45 degrees and 135 degrees)
kernel_45 = np.array([[2, 1, 0],
                      [1, 0, -1],
                      [0, -1, -2]], dtype=np.float32)

kernel_135 = np.array([[0, 1, 2],
                       [-1, 0, 1],
                       [-2, -1, 0]], dtype=np.float32)

# Apply the kernels to detect diagonal lines
diagonal_45 = cv2.filter2D(edges, -1, kernel_45)
diagonal_135 = cv2.filter2D(edges, -1, kernel_135)

# Combine the results to highlight regions with diagonal shading
combined_diagonals = cv2.bitwise_or(diagonal_45, diagonal_135)

# Use morphological operations to highlight the detected patterns
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(combined_diagonals, kernel, iterations=1)

# Apply the inverted polygon mask to the dilated image to keep only edges outside the polygon
dilated_masked = cv2.bitwise_and(dilated, polygon_mask_inv)

# Create a mask from the detected regions
mask = np.zeros_like(original_image)
mask[dilated_masked > 0] = [0, 0, 255]  # Red color for detected regions

# Combine the mask with the original image
highlighted_image = cv2.addWeighted(original_image, 0.7, mask, 0.3, 0)

# Draw the polygon in red on the final image with a thicker line
if points_array is not None:
    cv2.polylines(highlighted_image, [points_array], isClosed=True, color=(0, 0, 255), thickness=4)

# Display the final result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
plt.title('Final Image with Traced Edges Outside Polygon')
plt.show()

# Save the output image
output_image_path = 'detected_diagonal_shading_polygon_outside.png'
cv2.imwrite(output_image_path, highlighted_image)
