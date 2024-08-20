import cv2
import numpy as np

def crop_to_content(image, padding=5):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the coordinates of the bounding box of non-white pixels
    # Assuming white pixels have a value of 255 in grayscale
    coords = cv2.findNonZero(255 - gray)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the image to the bounding box
    cropped_image = image[y:y+h, x:x+w]

    # Add a white border of 5 pixels around the cropped image
    cropped_with_border = cv2.copyMakeBorder(
        cropped_image,
        top=padding, bottom=padding, left=padding, right=padding,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # White color border
    )

    return cropped_with_border

# Load the images
image1 = cv2.imread('original_image.jpg')
image2 = cv2.imread('original_image2.jpg')

# Crop to content and add white border
cropped_image1 = crop_to_content(image1, padding=5)
cropped_image2 = crop_to_content(image2, padding=5)

print(cropped_image1.shape)
print(cropped_image2.shape)

# Save or display the results
cv2.imwrite('cropped_with_border_image1.jpg', cropped_image1)
cv2.imwrite('cropped_with_border_image2.jpg', cropped_image2)

cv2.imshow('Cropped Image 1', cropped_image1)
cv2.imshow('Cropped Image 2', cropped_image2)

cv2.waitKey(0)
cv2.destroyAllWindows()
