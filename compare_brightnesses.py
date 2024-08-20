import cv2

def compare_darkest_pixels(image1, image2):
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(cropped_image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2GRAY)

    # Find the minimum pixel value (darkest pixel) in each image
    darkest_pixel_image1 = gray_image1.min()
    darkest_pixel_image2 = gray_image2.min()

    # Print the darkest pixel values
    print(f"Darkest Pixel in Image 1: {darkest_pixel_image1}")
    print(f"Darkest Pixel in Image 2: {darkest_pixel_image2}")

    # Compare the two darkest pixel values
    if darkest_pixel_image1 < darkest_pixel_image2:
        print("Image 1 has darker pixels.")
    elif darkest_pixel_image1 > darkest_pixel_image2:
        print("Image 2 has darker pixels.")
    else:
        print("Both images have equally dark pixels.")

# Load the cropped and bordered images
cropped_image1 = cv2.imread('cropped_with_border_image1.jpg')
cropped_image2 = cv2.imread('cropped_with_border_image2.jpg')

# Compare the darkness of the darkest pixels
compare_darkest_pixels(cropped_image1, cropped_image2)
