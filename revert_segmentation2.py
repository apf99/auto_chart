import cv2
import numpy as np
import os
import sys
import time

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
                    # else:
                    #     # Log when a group would be eliminated by the secondary filter
                    #     print(f"\nColumn {column}: Secondary filter eliminated a group")
                    #     print(f"prev_groups: {prev_groups}")
                    #     print(f"current_groups: {groups}")
                    #     print(f"final_proposed_groups: {final_proposed_groups}")
                else:
                    final_proposed_groups.append(group)
            else:
                for py in range(end_y, start_y + 1):
                    bitmap[py, column] = 255  # Set to white (eliminate)
            in_group = False
    
    # Update bitmap to remove eliminated groups
    eliminated_groups = [group for group in groups if group not in final_proposed_groups]
    for group in eliminated_groups:
        start_y, end_y, _ = group
        for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
            bitmap[y, column] = 255  # Set to white (eliminate)

    # Then return as before
    return len(final_proposed_groups), final_proposed_groups

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
    
    # Update running averages of heights
    if group_count == len(start_points):
        for i, group in enumerate(groups):
            avg_heights[i] = (avg_heights[i] * 0.1 + group[2] * 0.9)  # Weighted average
    
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

# After the forward pass, initialize variables for the reverse pass
right_x = max(point[0] for point in end_points)
left_x = min(point[0] for point in start_points)
avg_heights_backward = avg_heights.copy()  # Use the final averages from the forward pass
prev_groups = []

print("\n\nEnding points:")
for x, y in end_points:
    print(f"  ({x}, {y})")

print(f"\n*** Starting reverse pass from x={right_x} to x={left_x} ***")
print(f"Initial avg_heights_backward: {avg_heights_backward}")

# Create a set of ending y-coordinates for quick lookup
end_y_coords = set(y for _, y in end_points)

# Reverse pass: Analyze first 1000 columns from right to left
for i, x in enumerate(range(right_x, max(left_x, right_x - 1000), -1)):
    group_count, groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, prev_groups)
    
    # Check for end point connectivity only if we're in the same column as an ending point
    end_connected_groups = []
    other_groups = []
    for group in groups:
        if x in [point[0] for point in end_points]:
            start_y, end_y, _ = group
            group_y_set = set(range(min(start_y, end_y), max(start_y, end_y) + 1))
            if group_y_set.intersection(end_y_coords):
                end_connected_groups.append(group)
                continue
        other_groups.append(group)
    
    # Apply secondary filter to all groups not connected to ending points
    filtered_groups = end_connected_groups.copy()
    for group in other_groups:
        if apply_secondary_filter(group, prev_groups):
            filtered_groups.append(group)
        else:
            # Remove the group from the bitmap and log
            start_y, end_y, _ = group
            for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
                bitmap[y, x] = 255  # Set to white (eliminate)
            print(f"\nColumn {x}: Secondary filter removed a group {group}")
    
    # Update bitmap to remove eliminated groups
    eliminated_groups = [group for group in groups if group not in filtered_groups]
    for group in eliminated_groups:
        start_y, end_y, _ = group
        for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
            bitmap[y, x] = 255  # Set to white (eliminate)
    
    # Check if we're at an intersection boundary
    in_intersection = any(right <= x <= left for left, right in zip(intersection_starts, intersection_ends))
    
    # Update running averages of heights if not in an intersection
    if len(filtered_groups) == len(end_points) and not in_intersection:
        for j, group in enumerate(filtered_groups):
            avg_heights_backward[j] = (avg_heights_backward[j] * 0.1 + group[2] * 0.9)  # Weighted average
            
        # Check bitmap status for each group
        # for group in groups:
        #     start_y, end_y, _ = group
        #     group_pixels = [bitmap[y, x] for y in range(min(start_y, end_y), max(start_y, end_y) + 1)]
        #     is_eliminated = all(pixel == 255 for pixel in group_pixels)
        #     if is_eliminated:
        #         print(f"  Group {group} eliminated from bitmap: {is_eliminated}")
    
    # Store the filtered groups for the next iteration
    prev_groups = filtered_groups

    if len(filtered_groups) > len(end_points):
        print(f'\n*** Too Many Groups in Column {x}')
        print(f'Groups: {filtered_groups}')
        break

print("\nReverse pass completed")
print(f"Final avg_heights_backward: {avg_heights_backward}")

# After the reverse pass, update the contour image
contour_img_updated = np.ones_like(img) * 255
cv2.drawContours(contour_img_updated, filtered_contours, -1, (0, 0, 255), line_thickness)

# Redraw the circles for start and end points
for point in start_points:
    cv2.circle(contour_img_updated, point, 10, (0, 165, 255), -1)
for point in end_points:
    cv2.circle(contour_img_updated, point, 10, (255, 255, 0), -1)

# Get image height for drawing intersection lines
image_height = contour_img_updated.shape[0]

# Redraw intersection lines
for left, right in zip(intersection_starts, intersection_ends):
    cv2.line(contour_img_updated, (left, 0), (left, image_height), (0, 0, 0), 2)  # Black line for left side
    cv2.line(contour_img_updated, (right, 0), (right, image_height), (255, 0, 255), 2)  # Magenta line for right side

# Save the final updated bitmap for debugging
save_debug_image(bitmap, 'debug_final_bitmap_after_reverse_pass.jpg')

# After the reverse pass

# Initialize anomaly detection
is_anomaly = False
anomaly_start_columns = []
anomaly_end_columns = []
anomaly_start_groups = []
anomaly_end_groups = []

# Use the final avg_heights_backward from the reverse pass
running_avg_heights = avg_heights_backward.copy()

# Start from the leftmost starting point
left_x = min(point[0] for point in start_points)
right_x = max(point[0] for point in end_points)

print("\n*** Starting anomaly detection pass ***")
print(f"Initial running_avg_heights: {running_avg_heights}")
print(f"Start points: {start_points}")

# Create a set of starting point coordinates for quick lookup
start_point_coords = set(start_points)

prev_groups = []

# Anomaly detection pass
for x in range(left_x, right_x + 1):
    
    if not any(start <= x <= end for start, end in zip(intersection_starts, intersection_ends)):
        group_count, groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])
        
        if group_count == len(start_points):
            all_within_range = True
            for i, group in enumerate(groups):
                start_y, end_y, height = group
                is_start_point = (x, start_y) in start_point_coords or (x, end_y) in start_point_coords
                
                if not is_anomaly and height > running_avg_heights[i] * 1.8 and not is_start_point:  # 80% increase
                    is_anomaly = True
                    anomaly_start_columns.append(x)
                    anomaly_start_groups.append((prev_groups, groups))
                    print(f"  Anomaly start detected! Group height: {height}, Running avg: {running_avg_heights[i]}")
                
                # Check if group is within 20% of running average
                if abs(height - running_avg_heights[i]) > 0.2 * running_avg_heights[i]:
                    all_within_range = False
            
            if is_anomaly and all_within_range:
                anomaly_end_columns.append(x)
                anomaly_end_groups.append((prev_groups, groups))
                print(f"  Anomaly end detected at column {x}")
                is_anomaly = False
        
        if not is_anomaly:
            # Update running averages if not in anomaly and not a starting point
            for i, group in enumerate(groups):
                start_y, end_y, height = group
                is_start_point = (x, start_y) in start_point_coords or (x, end_y) in start_point_coords
                if not is_start_point:
                    old_avg = running_avg_heights[i]
                    running_avg_heights[i] = running_avg_heights[i] * 0.1 + height * 0.9
        prev_groups = groups

print("\n*** Anomaly detection pass completed ***")
if anomaly_start_columns:
    for i, (start, end) in enumerate(zip(anomaly_start_columns, anomaly_end_columns)):
        print(f"\nAnomaly {i+1}:")
        print(f"Start column: {start}")
        print(f"Previous groups at start: {anomaly_start_groups[i][0]}")
        print(f"Current groups at start: {anomaly_start_groups[i][1]}")
        print(f"End column: {end}")
        print(f"Previous groups at end: {anomaly_end_groups[i][0]}")
        print(f"Current groups at end: {anomaly_end_groups[i][1]}")
else:
    print("No anomalies detected")

# Draw blue lines for anomaly starts and green lines for anomaly ends
for start in anomaly_start_columns:
    cv2.line(contour_img_updated, (start, 0), (start, image_height), (255, 0, 0), 2)
for end in anomaly_end_columns:
    cv2.line(contour_img_updated, (end, 0), (end, image_height), (0, 255, 0), 2)

# After the anomaly detection pass and before drawing the lines

print("\n*** Starting smoothing process ***")

if anomaly_start_columns:
    for region_index, (start_column, end_column) in enumerate(zip(anomaly_start_columns, anomaly_end_columns)):
        # Define the columns for gradient calculation
        col_start = start_column - 21
        col_end = start_column - 1  # Column B

        # Get groups for both columns
        _, groups_start = identify_pixel_groups(bitmap, col_start, 1, start_points, end_points, [])
        _, groups_end = identify_pixel_groups(bitmap, col_end, 1, start_points, end_points, [])

        # Extract the relevant groups using anomalous_group_index
        anomalous_group_index = next(i for i, (start, end) in enumerate(zip(anomaly_start_groups[region_index][0], anomaly_start_groups[region_index][1]))
                                 if end[2] > start[2] * 1.8)
        group_start = groups_start[anomalous_group_index]
        group_end = groups_end[anomalous_group_index]

        # Calculate average gradient
        gradient_top = (group_end[0] - group_start[0]) / 20
        gradient_bottom = (group_end[1] - group_start[1]) / 20

        # Get the starting point from column B (just before the anomaly starts)
        start_top_y, start_bottom_y, _ = group_end

        print(f"\nSmoothing Anomalous Region {region_index + 1}: columns {start_column} to {end_column}")
        print(f"Anomalous group index: {anomalous_group_index}")
        print(f"Start y-range: {start_top_y} to {start_bottom_y}")
        print(f"Gradient (top, bottom): {gradient_top}, {gradient_bottom}")

        print("\nProcessing columns:")

        # Iterate through the columns of the anomalous region, excluding the end column
        for col in range(start_column, end_column):
            # Calculate steps from column B
            steps = col - col_end

            # Apply linear interpolation
            predicted_top_y = round(start_top_y + gradient_top * steps)
            predicted_bottom_y = round(start_bottom_y + gradient_bottom * steps)
            predicted_height = abs(predicted_bottom_y - predicted_top_y) + 1

            # Get current groups for this column
            _, current_groups = identify_pixel_groups(bitmap, col, 1, start_points, end_points, [])

            # Extract the anomalous group using the anomalous_group_index
            anomalous_group = current_groups[anomalous_group_index]
            actual_top_y, actual_bottom_y, actual_height = anomalous_group

            # Ensure correct order (smaller y value first)
            predicted_min_y, predicted_max_y = min(predicted_top_y, predicted_bottom_y), max(predicted_top_y, predicted_bottom_y)
            actual_min_y, actual_max_y = min(actual_top_y, actual_bottom_y), max(actual_top_y, actual_bottom_y)

            # Initialize pixels_changed
            pixels_changed = 0

            # Modify pixels and count changes
            for y in range(actual_min_y, actual_max_y + 1):
                if y < predicted_min_y or y > predicted_max_y:  # OUTSIDE the predicted range
                    if bitmap[y, col] == 0:  # Only change if the pixel is currently black
                        bitmap[y, col] = 255  # Set to white
                        pixels_changed += 1

            print(f"\nColumn {col}:")
            print(f"  Predicted group: ({predicted_min_y}, {predicted_max_y}, {predicted_height})")
            print(f"  Anomalous group: ({actual_min_y}, {actual_max_y}, {actual_height})")
            print(f"  Pixels changed: {pixels_changed}")

        print(f"\nSmoothing process completed for Region {region_index + 1}.")

    print("\nAll anomalous regions processed.")

    # Save the final bitmap image for debugging
    final_bitmap_path = os.path.join(script_dir, 'final_smoothed_bitmap.jpg')
    cv2.imwrite(final_bitmap_path, bitmap)
    print(f"Final smoothed bitmap saved to: {final_bitmap_path}")

else:
    print("No anomalies detected, no smoothing necessary.")

# Now save the final image with all anomaly markings
final_output_path = os.path.join(script_dir, 'final_result_with_anomalies.jpg')
cv2.imwrite(final_output_path, contour_img_updated)
print(f"Final image with anomaly detections saved to: {final_output_path}")

# After all the previous processing, add this code

print("\n*** Starting curve tracing process ***")

def trace_curve(start_point, bitmap, intersection_starts, intersection_ends):
    curve = [start_point]
    x, y = start_point

    # Get the initial group
    _, initial_groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])
    current_group = next((group for group in initial_groups if y in range(min(group[0], group[1]), max(group[0], group[1]) + 1)), None)

    if current_group is None:
        print(f"Warning: No valid starting group found for point {start_point}")
        return curve

    while x < width - 1:
        x += 1
        # Check if we're entering an intersection
        intersection_index = next((i for i, start in enumerate(intersection_starts) if start == x), None)
        
        if intersection_index is not None:
            # We're entering an intersection, use prediction
            intersection_start = intersection_starts[intersection_index]
            intersection_end = intersection_ends[intersection_index]
            
            # Calculate gradient
            col_start = max(0, intersection_start - 11)
            col_end = intersection_start - 1
            _, groups_start = identify_pixel_groups(bitmap, col_start, 1, start_points, end_points, [])
            _, groups_end = identify_pixel_groups(bitmap, col_end, 1, start_points, end_points, [])
            
            curve_index = start_points.index(start_point)
            group_start = groups_start[curve_index % len(groups_start)]
            group_end = groups_end[curve_index % len(groups_end)]
            
            gradient_top = (group_end[0] - group_start[0]) / (col_end - col_start)
            gradient_bottom = (group_end[1] - group_start[1]) / (col_end - col_start)
            
            print(f"Curve starting at {start_point}:")
            print(f"  Intersection at x={intersection_start}")
            print(f"  Gradient calculation:")
            print(f"    Start column: {col_start}, End column: {col_end}")
            print(f"    Start group: {group_start}, End group: {group_end}")
            print(f"    Calculated gradients - Top: {gradient_top:.4f}, Bottom: {gradient_bottom:.4f}")
            
            # Predict through intersection and two columns beyond
            start_top_y, start_bottom_y = group_end[0], group_end[1]
            for col in range(intersection_start, intersection_end + 1):  
                steps = col - col_end
                predicted_top_y = round(start_top_y + gradient_top * steps)
                predicted_bottom_y = round(start_bottom_y + gradient_bottom * steps)
                middle_y = (predicted_top_y + predicted_bottom_y) // 2
                curve.append((col, middle_y))
            
            # Get the last predicted point
            last_predicted_x, last_predicted_y = curve[-1]
            
            # Find the group containing the last predicted point
            _, groups = identify_pixel_groups(bitmap, last_predicted_x, 1, start_points, end_points, [])
            current_group = next((group for group in groups if last_predicted_y in range(min(group[0], group[1]), max(group[0], group[1]) + 1)), None)
            
            if current_group is None:
                print(f"Warning: No valid group found for last predicted point ({last_predicted_x}, {last_predicted_y})")
                return curve
            
            # Move to the next column
            x = last_predicted_x + 1
            
            # Find connected group in the next column
            _, next_groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])
            prev_y_set = set(range(min(current_group[0], current_group[1]), max(current_group[0], current_group[1]) + 1))
            connected_group = next((group for group in next_groups if prev_y_set.intersection(set(range(min(group[0], group[1]), max(group[0], group[1]) + 1)))), None)
            
            if connected_group is None:
                print(f"Warning: No connected group found after intersection at x={x}")
                return curve
            
            # Use the middle pixel of the connected group as the next trace point
            middle_y = (connected_group[0] + connected_group[1]) // 2
            curve.append((x, middle_y))
            current_group = connected_group
            
            continue  # Skip to the next iteration to start normal tracing from the new x

        # Normal tracing
        prev_y_set = set(range(min(current_group[0], current_group[1]), max(current_group[0], current_group[1]) + 1))
        _, current_groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])
        
        connected_group = next((group for group in current_groups if prev_y_set.intersection(set(range(min(group[0], group[1]), max(group[0], group[1]) + 1)))), None)
        
        if connected_group is None:
            break
        
        middle_y = (connected_group[0] + connected_group[1]) // 2
        curve.append((x, middle_y))
        current_group = connected_group
        
        # Check if we've reached an end point
        if (x, middle_y) in end_points:
            break

    return curve

# Use the trace_curve function
# start_point = start_points[0]
additional_colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Blue, Green, Cyan, Magenta, Yellow
for j, start_point in enumerate(start_points):
    traced_curve = trace_curve(start_point, bitmap, intersection_starts, intersection_ends)

    # Load the final image
    final_image = cv2.imread(final_output_path)

    if final_image is not None:
        # Draw only this single traced curve
        for i in range(1, len(traced_curve)):
            cv2.line(final_image, traced_curve[i-1], traced_curve[i], additional_colors[j % len(additional_colors)], 3)
        cv2.imwrite(final_output_path, final_image)


    else:
        print(f"Error: Unable to load the final image from {final_output_path}")
        break

if final_image is not None:
        # Display the final image with traced curves
    cv2.imshow('Final Result', final_image)
    print("\nPress 'q' to save the image and end the program.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Save the image when 'q' is pressed
            final_result_path = os.path.join(script_dir, 'final_result.jpg')
            cv2.imwrite(final_result_path, final_image)
            print(f"Final result saved to: {final_result_path}")
            break
    cv2.destroyAllWindows()

print("Program ended.\n")