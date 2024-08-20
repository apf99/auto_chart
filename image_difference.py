import cv2
import numpy as np

def find_difference(image1, image2):
    # Ensure both images are the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must be the same size for comparison.")

    # Convert both images to grayscale
    gray_image1 = cv2.cvtColor(cropped_with_border_image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(cropped_with_border_image2, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the images
    difference = cv2.absdiff(gray_image1, gray_image2)

    # Calculate the mean difference
    mean_diff = np.mean(difference)

    # Count the number of non-zero pixels in the difference image
    non_zero_count = np.count_nonzero(difference)

    # Visualize the difference
    cv2.imshow('Difference Image', difference)
    cv2.imwrite('difference_image.jpg', difference)

    # Print the results
    print(f"Mean Brightness Difference: {mean_diff}")
    print(f"Number of Different Pixels: {non_zero_count}")

    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the cropped and bordered images
cropped_with_border_image1 = cv2.imread('cropped_with_border_image1.jpg')
cropped_with_border_image2 = cv2.imread('cropped_with_border_image2.jpg')

# Find the difference between the images
find_difference(cropped_with_border_image1, cropped_with_border_image2)
