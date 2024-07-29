import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_diagonal_shading(image_path, output_image_path):
    # Load the original image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection with specified thresholds
    edges = cv2.Canny(gray, 5, 10, apertureSize=3)

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
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(combined_diagonals, kernel, iterations=1)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to exclude regions with multiple parallel diagonal lines
    mask = np.zeros_like(gray)

    # Iterate through contours and filter out single or double lines
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        # Calculate the number of points in the contour
        num_points = len(contour)
        # Filter out small areas and contours with few points (likely single or double lines)
        if area > 100 and num_points > 69:  # Adjust these thresholds based on your image
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Invert the mask to highlight regions with multiple parallel lines
    mask_inv = cv2.bitwise_not(mask)

    # Create a blue mask image
    blue_mask = np.zeros_like(image)
    blue_mask[mask_inv == 0] = [255, 0, 0]  # Blue color for detected regions

    # Apply the blue mask to the original image
    masked_image = cv2.addWeighted(image, 0.7, blue_mask, 0.3, 0)

    # Display the masked image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.title('Masked Image Excluding Shaded Regions (in Blue)')
    plt.show()

    # Save the output image
    cv2.imwrite(output_image_path, masked_image)

# Specify the input and output image paths
image_path = 'Flooring.jpg'
output_image_path = 'masked_image_excluding_shaded_regions_blue.png'

# Call the function to process the image
detect_diagonal_shading(image_path, output_image_path)
