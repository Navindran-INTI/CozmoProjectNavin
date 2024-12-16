import cozmo
import cv2
import numpy as np
import math
from cozmo.util import degrees, distance_mm, speed_mmps

# Global variable to store detected cliffs as avoidance zones
detected_cliffs = []

def process_camera_image(image):
    """Process the camera image to detect dark areas (potential cliffs) and visually mark them."""
    global detected_cliffs

    # Convert Cozmo's camera image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image.raw_image), cv2.COLOR_RGB2BGR)

    # Focus on the lower portion of the image (ground ahead)
    height, width, _ = opencv_image.shape
    lower_half = opencv_image[int(height * 0.5):, :]

    # Convert to grayscale for simpler processing
    gray_image = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)

    # Adjust the threshold to detect slightly lighter dark areas
    _, binary_image = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to combine nearby regions
    kernel = np.ones((5, 5), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Find contours of dark regions (cliffs)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store detected cliffs and mark them visually
    safe_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            print("Detected potential cliff in camera feed.")
            x, y, w, h = cv2.boundingRect(contour)
            detected_cliffs.append((x, y, w, h))  # Store cliff bounding box
            cv2.drawContours(lower_half, [contour], -1, (0, 0, 255), 2)  # Mark cliffs in red

    # Highlight remaining safe areas
    safe_mask = cv2.bitwise_not(binary_image)  # Invert binary image to find non-cliff areas
    contours_safe, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_safe:
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            safe_areas.append((x + w // 2, y + h // 2))  # Safe region center

    # Debugging: Display safe areas visually
    cv2.imshow("Cliff Detection - Visual Representation", opencv_image)
    cv2.waitKey(1)

    return safe_areas  # Return list of safe positions


def reposition_to_safe_area(robot: cozmo.robot.Robot, safe_areas):
    """Reposition Cozmo to a safe area detected in the camera feed."""
    if not safe_areas:
        print("No safe areas detected. Stopping for safety.")
        return

    # Select the closest safe area
    closest_safe_area = safe_areas[0]
    print(f"Repositioning to safe area at {closest_safe_area}...")

    # Move Cozmo slightly to the safe position
    robot.drive_straight(distance_mm(-50), speed_mmps(50)).wait_for_completed()
    robot.turn_in_place(degrees(45)).wait_for_completed()
    robot.drive_straight(distance_mm(100), speed_mmps(50)).wait_for_completed()
    print("Repositioning complete.")


def compare_safe_areas(left_safe, right_safe):
    """Compare the safe areas from left and right images and determine the best direction."""
    left_size = len(left_safe)
    right_size = len(right_safe)

    print(f"Safe zones - Left: {left_size}, Right: {right_size}")

    if left_size > right_size:
        print("Left direction has the largest safe zone.")
        return "left"
    else:
        print("Right direction has the largest safe zone.")
        return "right"

def handle_cliff(robot: cozmo.robot.Robot):
    """Handle cliff detection by analyzing the ground in left and right directions and moving to a safe area."""
    if robot.is_cliff_detected:
        print("Cliff detected! Reversing and analyzing the ground...")
        robot.drive_straight(distance_mm(-100), speed_mmps(50)).wait_for_completed()

        # Adjust head for analysis
        robot.set_lift_height(1.0).wait_for_completed()
        robot.set_head_angle(degrees(-5)).wait_for_completed()

        # Turn left and capture the left image
        robot.turn_in_place(degrees(20)).wait_for_completed()
        latest_image_left = robot.world.latest_image
        left_safe = process_camera_image(latest_image_left) if latest_image_left else []
        print("Processed left image for safe zones.")

        # Turn right (40 degrees total from left) and capture the right image
        robot.turn_in_place(degrees(-40)).wait_for_completed()
        latest_image_right = robot.world.latest_image
        right_safe = process_camera_image(latest_image_right) if latest_image_right else []
        print("Processed right image for safe zones.")

        # Return to center
        robot.turn_in_place(degrees(20)).wait_for_completed()

        # Compare safe zones and decide direction
        best_direction = compare_safe_areas(left_safe, right_safe)

        if best_direction == "left":
            print("Turning left and moving to the safe zone...")
            robot.turn_in_place(degrees(45)).wait_for_completed()
            robot.drive_straight(distance_mm(200), speed_mmps(50)).wait_for_completed()
        else:
            print("Turning right and moving to the safe zone...")
            robot.turn_in_place(degrees(-45)).wait_for_completed()
            robot.drive_straight(distance_mm(200), speed_mmps(50)).wait_for_completed()

        # Return to indicate cliff was handled
        robot.set_lift_height(0.0).wait_for_completed()
        return True
    return False


def move_to_coordinates_with_cliff_handling(robot: cozmo.robot.Robot, x, y):
    """Move to specific coordinates while avoiding cliffs."""
    while True:
        robot_x = robot.pose.position.x
        robot_y = robot.pose.position.y
        dx = x - robot_x
        dy = y - robot_y
        distance = math.sqrt(dx**2 + dy**2)

        if distance <= 80:
            print(f"Reached target at x={x:.2f}, y={y:.2f}")
            break

        if handle_cliff(robot):
            print("Restarting movement after avoiding cliff.")
            continue

        angle = math.degrees(math.atan2(dy, dx))
        turn_angle = angle - robot.pose.rotation.angle_z.degrees
        turn_angle = (turn_angle + 180) % 360 - 180

        if abs(turn_angle) > 5:
            robot.turn_in_place(degrees(turn_angle)).wait_for_completed()

        move_distance = min(30, distance)
        print(f"Driving forward by {move_distance:.2f} mm.")
        robot.drive_straight(distance_mm(move_distance), speed_mmps(30)).wait_for_completed()


def cozmo_program(robot: cozmo.robot.Robot):
    """Cozmo program to test cliff detection, repositioning, and target movement."""
    try:
        robot.set_head_angle(degrees(5)).wait_for_completed()
        print("Starting Cozmo's movement with cliff avoidance...")
        move_to_coordinates_with_cliff_handling(robot, 600, 0)
        print("Successfully reached the target!")
    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        print("Shutting down...")
        print("Detected cliffs:", detected_cliffs)
        cv2.destroyAllWindows()


cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)

