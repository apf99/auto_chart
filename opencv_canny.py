import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# List to store points
points = []

# Zoom window size and scale
zoom_size = 800  # Increased size for picture-in-picture
zoom_scale = 4   # Increased scale to maintain the same zoom level with larger window

def create_zoom_window(img, x, y, size, scale):
    h, w = img.shape[:2]
    
    # Calculate zoom area
    x1, y1 = max(x - size // (2 * scale), 0), max(y - size // (2 * scale), 0)
    x2, y2 = min(x1 + size // scale, w), min(y1 + size // scale, h)
    
    # Create zoomed image
    zoomed = img[y1:y2, x1:x2]
    zoomed = cv2.resize(zoomed, (size, size), interpolation=cv2.INTER_LINEAR)
    
    # Add crosshair
    cv2.line(zoomed, (size // 2, 0), (size // 2, size), (0, 255, 0), 1)
    cv2.line(zoomed, (0, size // 2), (size, size // 2), (0, 255, 0), 1)
    
    # Add black frame
    framed = cv2.copyMakeBorder(zoomed, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return framed

def overlay_zoom_window(main_image, zoom_window):
    # Create a copy of the main image
    result = main_image.copy()
    
    # Get dimensions
    h, w = main_image.shape[:2]
    zh, zw = zoom_window.shape[:2]
    
    # Calculate position (top-right corner)
    x = w - zw - 10
    y = 10
    
    # Create a subregion in the main image
    roi = result[y:y+zh, x:x+zw]
    
    # Create a mask of the zoom window and its inverse mask
    zoom_gray = cv2.cvtColor(zoom_window, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(zoom_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # Black-out the area of zoom window in ROI
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    
    # Take only region of zoom window from zoom window image
    zoom_fg = cv2.bitwise_and(zoom_window, zoom_window, mask=mask)
    
    # Put zoom window in ROI and modify the main image
    dst = cv2.add(img_bg, zoom_fg)
    result[y:y+zh, x:x+zw] = dst
    
    return result

def save_polygon():
    if len(points) > 2:
        with open('polygon_coordinates.json', 'w') as f:
            json.dump(points, f)
        print("Polygon saved successfully.")
    else:
        print("Not enough points to save a polygon.")

def load_polygon():
    file_path = 'polygon_coordinates.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def downscale_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def upscale_point(point, original_shape, current_shape):
    x_scale = original_shape[1] / current_shape[1]
    y_scale = original_shape[0] / current_shape[0]
    return (int(point[0] * x_scale), int(point[1] * y_scale))

def mouse_callback(event, x, y, flags, param):
    global points, display_image, original_image
    
    # Create zoom window
    zoom_window = create_zoom_window(original_image, 
                                     int(x * original_image.shape[1] / display_image.shape[1]), 
                                     int(y * original_image.shape[0] / display_image.shape[0]), 
                                     zoom_size, zoom_scale)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        upscaled_point = upscale_point((x, y), original_image.shape, display_image.shape)
        points.append(upscaled_point)
        cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
        
        # Draw point on zoom window
        zoom_point = (zoom_size // 2, zoom_size // 2)
        cv2.circle(zoom_window, zoom_point, 5, (0, 0, 255), -1)
    
    # Draw existing points on zoom window
    for point in points:
        downscaled_point = (int(point[0] * display_image.shape[1] / original_image.shape[1]),
                            int(point[1] * display_image.shape[0] / original_image.shape[0]))
        if abs(downscaled_point[0] - x) < zoom_size // (2 * zoom_scale) and abs(downscaled_point[1] - y) < zoom_size // (2 * zoom_scale):
            zoom_x = int((downscaled_point[0] - x) * zoom_scale + zoom_size // 2)
            zoom_y = int((downscaled_point[1] - y) * zoom_scale + zoom_size // 2)
            if 0 <= zoom_x < zoom_size and 0 <= zoom_y < zoom_size:
                cv2.circle(zoom_window, (zoom_x, zoom_y), 5, (0, 0, 255), -1)
    
    # Overlay zoom window on display image
    display_with_zoom = overlay_zoom_window(display_image, zoom_window)
    cv2.imshow('image', display_with_zoom)

def main():
    global image, points, original_image, display_image

    # Load the original image
    image_path = 'Flooring.jpg'
    original_image = cv2.imread(image_path)
    
    # Ensure original_image is in BGR format
    if len(original_image.shape) == 2:  # If grayscale
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    # Downscale the image for display and interaction
    display_image = downscale_image(original_image, scale_percent=50)  # Adjust scale_percent as needed

    # Create a window for display
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', display_image.shape[1], display_image.shape[0])

    # Initialize points
    points = []

    cv2.imshow('image', display_image)
    cv2.setMouseCallback('image', mouse_callback)

    # Try to load saved polygon
    saved_points = load_polygon()

    if saved_points:
        print("Previous polygon coordinates loaded. Press 'q' to use them, 'c' to clear, or start clicking to define a new polygon.")
    else:
        print("No saved polygon found. Start clicking to define a new polygon.")
        print("Press 'f' to finish and save the polygon, or 'q' to quit without saving.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if not points and saved_points:
                points = saved_points
                # Draw saved points on the image
                display_image = downscale_image(original_image)
                for point in points:
                    downscaled_point = (int(point[0] * display_image.shape[1] / original_image.shape[1]),
                                        int(point[1] * display_image.shape[0] / original_image.shape[0]))
                    cv2.circle(display_image, downscaled_point, 5, (0, 0, 255), -1)
                combined_image = overlay_zoom_window(display_image, create_zoom_window(original_image, 0, 0, zoom_size, zoom_scale))
                cv2.imshow('image', combined_image)
                print("Using saved polygon.")
            elif not points and not saved_points:
                print("No polygon defined or saved. Exiting without processing.")
            print("Exiting...")
            break
        elif key == ord('c'):
            points = []
            display_image = downscale_image(original_image)
            combined_image = overlay_zoom_window(display_image, create_zoom_window(original_image, 0, 0, zoom_size, zoom_scale))
            cv2.imshow('image', combined_image)
            print("Coordinates cleared. Please define a new polygon.")
        elif key == ord('d'):
            if points:
                points.pop()
                display_image = downscale_image(original_image)
                for point in points:
                    downscaled_point = (int(point[0] * display_image.shape[1] / original_image.shape[1]),
                                        int(point[1] * display_image.shape[0] / original_image.shape[0]))
                    cv2.circle(display_image, downscaled_point, 5, (0, 0, 255), -1)
                combined_image = overlay_zoom_window(display_image, create_zoom_window(original_image, 0, 0, zoom_size, zoom_scale))
                cv2.imshow('image', combined_image)
        elif key == ord('f'):  # Finish and save polygon
            if len(points) > 2:
                save_polygon()
                print("Polygon saved. Press 'q' to quit and use this polygon, or continue defining a new one.")
            else:
                print("Not enough points to form a polygon. Please add more points.")

    cv2.destroyAllWindows()

    # Only process if we have points
    if points:
        # Ensure the polygon is closed
        if len(points) > 2 and points[0] != points[-1]:
            points.append(points[0])

        # Create a mask from the polygon points
        polygon_mask = np.zeros_like(original_image[:, :, 0])
        points_array = None
        if len(points) > 0:
            points_array = np.array([points], dtype=np.int32)
            cv2.fillPoly(polygon_mask, points_array, 255)

        # Mask the original image to keep only the area inside the polygon
        masked_image = cv2.bitwise_and(original_image, original_image, mask=polygon_mask)

        # Convert the masked image to grayscale
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection to find edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Set the minimum line length
        min_line_length = 100  # Adjust this value as needed

        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=min_line_length, maxLineGap=10)

        # Create a blank mask to draw the lines
        line_mask = np.zeros_like(gray)

        # Draw the detected lines on the mask
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

        # Use morphological operations to highlight the detected patterns
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(line_mask, kernel, iterations=1)

        # Find contours in the dilated image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask to exclude small circles with writing
        exclude_mask = np.zeros_like(polygon_mask)

        for contour in contours:
            # Calculate the area and the perimeter of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Calculate the circularity
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            # Filter out small circular contours
            if area < 1000 and 0.7 < circularity < 1.3:
                cv2.drawContours(exclude_mask, [contour], -1, 255, -1)

        # Apply the exclusion mask to the dilated image
        dilated_masked = cv2.bitwise_and(dilated, dilated, mask=cv2.bitwise_not(exclude_mask))
        dilated_masked = cv2.bitwise_and(dilated_masked, polygon_mask)

        # Create a mask from the detected regions
        mask = np.zeros_like(original_image)
        mask[dilated_masked > 0] = [0, 0, 255]  # Red color for detected regions

        # Combine the mask with the original image
        highlighted_image = original_image.copy()
        highlighted_image[mask > 0] = mask[mask > 0]

        # Draw the polygon in red on the final image with a thicker line
        if points_array is not None:
            cv2.polylines(highlighted_image, [points_array], isClosed=True, color=(0, 0, 255), thickness=4)

        # Display the final result
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
        plt.title('Final Image with Traced Edges Inside Polygon')
        plt.show()

        # Save the output image
        output_image_path = 'detected_diagonal_shading_polygon_inside.png'
        cv2.imwrite(output_image_path, highlighted_image)
    else:
        print("No polygon defined. No processing performed.")

if __name__ == "__main__":
    main()