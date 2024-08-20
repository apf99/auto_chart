import cv2
import numpy as np
import os
import sys

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the image
image_path = os.path.join(script_dir, 'Chart_Image_1.jpg')

# Read the image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Unable to load image from {image_path}")
    exit(1)

# Get and print image dimensions
height, width = img.shape[:2]
print(f"Image dimensions: {width} x {height}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blur_size = 3
blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

# Apply thresholding
thresh_value = 40
_, binary = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)

# Set the specified parameters
min_length = 282
min_area = 570
line_thickness = 7

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on length, area, and proximity to edges
filtered_contours = []
edge_margin = 10  # Margin from the edge to consider as border

for cnt in contours:
    length = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    
    if (length > min_length and
        area > min_area and
        x > edge_margin and y > edge_margin and
        x + w < width - edge_margin and
        y + h < height - edge_margin):
        filtered_contours.append(cnt)

# Create a white background image
contour_img = np.ones_like(img) * 255

# Draw filtered contours with the specified thickness
cv2.drawContours(contour_img, filtered_contours, -1, (0, 0, 255), line_thickness)

# Store a copy of the contour image before adding circles
contour_img_copy = contour_img.copy()

# After creating contour_img_copy but before any further processing
contour_copy_path = os.path.join(script_dir, 'initial_contour_image_copy.jpg')
cv2.imwrite(contour_copy_path, contour_img_copy)
print(f"Initial contour image copy saved to: {contour_copy_path}")

def is_red_pixel(b, g, r):
    return r > 200 and g < 50 and b < 50

# Find starting points (from left to right)
start_points = []
scan_width = int(width * 0.2)  # Scan the first 20% of the image width

for x in range(scan_width):
    y = 0
    while y < height:
        if is_red_pixel(*map(int, contour_img[y, x])):
            top = bottom = y
            while bottom < height - 1 and is_red_pixel(*map(int, contour_img[bottom + 1, x])):
                bottom += 1
            
            connected = False
            if x > 0:
                for check_y in range(max(0, top - 2), min(height, bottom + 3)):
                    if is_red_pixel(*map(int, contour_img[check_y, x - 1])):
                        connected = True
                        break
                
                if not connected:
                    middle_y = (top + bottom) // 2
                    if is_red_pixel(*map(int, contour_img[middle_y, x - 1])):
                        connected = True
            
            if not connected:
                center_y = (top + bottom) // 2
                start_points.append((x, center_y))
            
            y = bottom + 1
        else:
            y += 1

# Find ending points (from right to left)
end_points = []
scan_width = int(width * 0.1)  # Scan the last 10% of the image width

for x in range(width - 1, width - scan_width - 1, -1):
    y = 0
    while y < height:
        if is_red_pixel(*map(int, contour_img[y, x])):
            top = bottom = y
            while bottom < height - 1 and is_red_pixel(*map(int, contour_img[bottom + 1, x])):
                bottom += 1
            
            connected = False
            if x < width - 1:
                for check_y in range(max(0, top - 2), min(height, bottom + 3)):
                    if is_red_pixel(*map(int, contour_img[check_y, x + 1])):
                        connected = True
                        break
                
                if not connected:
                    middle_y = (top + bottom) // 2
                    if is_red_pixel(*map(int, contour_img[middle_y, x + 1])):
                        connected = True
            
            if not connected:
                center_y = (top + bottom) // 2
                end_points.append((x, center_y))
            
            y = bottom + 1
        else:
            y += 1

# Draw circles at the starting points (orange)
for point in start_points:
    cv2.circle(contour_img, point, 10, (0, 165, 255), -1)

# Draw circles at the ending points (cyan)
for point in end_points:
    cv2.circle(contour_img, point, 10, (255, 255, 0), -1)
# Convert the contour image copy to a binary bitmap
_, bitmap = cv2.threshold(cv2.cvtColor(contour_img_copy, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

def save_debug_image(bitmap, filename):
    # Convert boolean or 0-1 float to 0-255 uint8
    if bitmap.dtype == bool or (bitmap.dtype in [np.float32, np.float64] and bitmap.max() <= 1):
        image_to_save = (1 - bitmap.astype(np.uint8)) * 255
    else:
        image_to_save = bitmap.astype(np.uint8)
    
    cv2.imwrite(filename, image_to_save)

# Save the initial bitmap for debugging
save_debug_image(bitmap, 'debug_initial_bitmap.jpg')

def apply_secondary_filter(group, prev_groups):
    start_y, end_y, _ = group
    current_group_pixels = set(range(min(start_y, end_y), max(start_y, end_y) + 1))
    
    prev_groups_pixels = set()
    for prev_start_y, prev_end_y, _ in prev_groups:
        prev_groups_pixels.update(range(min(prev_start_y, prev_end_y), max(prev_start_y, prev_end_y) + 1))
    
    return bool(current_group_pixels.intersection(prev_groups_pixels))

# Debug test for apply_secondary_filter
test_group = (100, 80, 21)  # start_y, end_y, height
test_prev_groups = [(95, 75, 21), (150, 130, 21)]
print(f"Test secondary filter: {apply_secondary_filter(test_group, test_prev_groups)}")

def identify_pixel_groups(bitmap, column, min_height, start_points, end_points, prev_groups):
    height = bitmap.shape[0]
    groups = []
    final_proposed_groups = []
    in_group = False
    start_y = 0
    
    start_ys = set(y for x, y in start_points if x == column)
    end_ys = set(y for x, y in end_points if x == column)
    
    for y in range(height - 1, -1, -1):
        if bitmap[y, column] == 0 and not in_group:
            in_group = True
            start_y = y
        elif (bitmap[y, column] == 255 or y == 0) and in_group:
            end_y = y if bitmap[y, column] == 255 else y + 1
            group_height = start_y - end_y + 1
            
            contains_start_end = any(py in start_ys or py in end_ys for py in range(end_y, start_y + 1))
            
            if group_height >= min_height or contains_start_end:
                group = (start_y, end_y, group_height)
                groups.append(group)
                
                if contains_start_end:
                    final_proposed_groups.append(group)
                elif prev_groups:
                    would_keep = apply_secondary_filter(group, prev_groups)
                    if would_keep:
                        final_proposed_groups.append(group)
                    else:
                        # Log when a group would be eliminated by the secondary filter
                        print(f"\nColumn {column}: Secondary filter would eliminate a group")
                        print(f"prev_groups: {prev_groups}")
                        print(f"current_groups: {groups}")
                        print(f"final_proposed_groups: {final_proposed_groups}")
                else:
                    final_proposed_groups.append(group)
            else:
                for py in range(end_y, start_y + 1):
                    bitmap[py, column] = 255
            in_group = False
    
    return len(groups), groups

def initialize_curve_thicknesses(bitmap, start_points, end_points):
    right_start = max(point[0] for point in start_points)
    left_end = min(point[0] for point in end_points)
    
    sample_range = left_end - right_start
    num_samples = 10
    sample_step = max(1, sample_range // num_samples)
    
    expected_groups = len(start_points)
    valid_thicknesses = []
    
    for x in range(right_start, left_end, sample_step):
        group_count, groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])  # Use empty list for prev_groups
        
        if group_count == expected_groups:
            avg_thickness = sum(group[2] for group in groups) / group_count
            valid_thicknesses.append(avg_thickness)
    
    if valid_thicknesses:
        return sum(valid_thicknesses) / len(valid_thicknesses)
    else:
        return None  # Or some default value

# Initialize curve thicknesses
initial_thickness = initialize_curve_thicknesses(bitmap, start_points, end_points)
if initial_thickness is None:
    print("Failed to initialize curve thicknesses")
    exit(1)

print(f"Initial average curve thickness: {initial_thickness:.2f}")

# Initialize variables for the main analysis
left_x = min(point[0] for point in start_points)
right_x = max(point[0] for point in end_points)
prev_group_count = len(start_points)
avg_heights = [initial_thickness] * len(start_points)
intersection_starts = []
intersection_ends = []
in_intersection = False
prev_groups = []

# Analyze columns from left to right
for x in range(left_x, right_x + 1):
    # Find the minimum of the two moving averages
    min_avg = min(avg_heights)
    
    # Set min_height to 60% of the minimum average
    min_height = max(1, int(min_avg * 0.6))

    group_count, groups = identify_pixel_groups(bitmap, x, min_height, start_points, end_points, prev_groups)
    
    # Check for excess groups and stop processing if found
    if group_count > len(start_points):
        print(f"Column {x}: More groups than starting points")
        print(f"Number of groups: {group_count}")
        print("Group details:")
        for i, (start_y, end_y, height) in enumerate(groups):
            print(f"  Group {i + 1}: Start Y: {start_y}, End Y: {end_y}, Height: {height}")
        print("--------------------")
        
        # Draw a blue line on the final image at this column
        cv2.line(contour_img, (x, 0), (x, contour_img.shape[0]), (255, 0, 0), 1)
        
        # Break the loop to stop further processing
        break

    # Update running averages of heights
    if group_count == len(start_points):
        for i, group in enumerate(groups):
            avg_heights[i] = (avg_heights[i] * 0.9 + group[2] * 0.1)  # Weighted average
    
    # Check for intersection start
    if not in_intersection and group_count < prev_group_count and prev_group_count > 0:
        if groups:
            max_height = max(group[2] for group in groups)
            avg_height = sum(avg_heights) / len(avg_heights) if avg_heights else initial_thickness
            
            if max_height > 1.8 * avg_height:
                intersection_starts.append(x)
                in_intersection = True
                print(f"Intersection start detected at column {x}")
                save_debug_image(bitmap, f'debug_artifact_detected_column_{x}.jpg')
    
    # Check for intersection end
    elif in_intersection and group_count > 1:
        max_height = max(group[2] for group in groups)
        if max_height < 1.5 * initial_thickness:  # Adjust this threshold as needed
            intersection_ends.append(x)
            in_intersection = False
            print(f"Intersection end detected at column {x}")
            save_debug_image(bitmap, f'debug_artifact_cleaned_column_{x}.jpg')
    
    prev_group_count = group_count
    prev_groups = groups  # Store current groups for next iteration

# Draw intersection lines on the final image
image_height = contour_img.shape[0]  # Get the full height of the image
for start, end in zip(intersection_starts, intersection_ends):
    cv2.line(contour_img, (start, 0), (start, image_height), (0, 0, 0), 2)  # Black line for start
    cv2.line(contour_img, (end, 0), (end, image_height), (255, 0, 255), 2)  # Magenta line for end

# Save the final bitmap for debugging
save_debug_image(bitmap, 'debug_final_bitmap.jpg')

# Save and display the final image
final_output_path = os.path.join(script_dir, 'final_result_with_intersections.jpg')
cv2.imwrite(final_output_path, contour_img)
cv2.imshow('Final Result with Intersections', contour_img)

print(f"Final image saved to: {final_output_path}")
print(f"Intersection starts: {intersection_starts}")
print(f"Intersection ends: {intersection_ends}")

cv2.waitKey(0)
cv2.destroyAllWindows()