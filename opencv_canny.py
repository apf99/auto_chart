import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
image_path = 'Flooring.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

# Create a mask from the detected regions
mask = np.zeros_like(image)
mask[dilated > 0] = [0, 0, 255]  # Red color for detected regions

# Combine the mask with the original image
highlighted_image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
plt.title('Detected Diagonal Shading Regions')
plt.show()

# Save the output image
output_image_path = 'detected_diagonal_shading.png'
cv2.imwrite(output_image_path, highlighted_image)
