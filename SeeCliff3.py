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
            x, y, w, h = cv2.boundingRect(contour)
            detected_cliffs.append((x, y, w, h))  # Store cliff bounding box
            cv2.drawContours(lower_half, [contour], -1, (0, 0, 255), 2)  # Mark cliffs in red

    # Highlight remaining safe areas
    safe_mask = cv2.bitwise_not(binary_image)
    contours_safe, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_safe:
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            safe_areas.append((x + w // 2, y + h // 2))  # Safe region center

    # Display processed image
    cv2.imshow("Cliff Detection - Visual Representation", opencv_image)
    cv2.waitKey(1)

    return safe_areas


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
    """Immediately stop motion when a cliff is detected and process the safe areas."""
    print("Cliff detected! Stopping movement immediately...")
    robot.stop_all_motors()

    # Reverse for safety
    robot.drive_straight(distance_mm(-100), speed_mmps(50)).wait_for_completed()

    # Analyze ground at left and right
    robot.set_head_angle(degrees(-5)).wait_for_completed()

    # Turn left and capture the left image
    robot.turn_in_place(degrees(5)).wait_for_completed()
    latest_image_left = robot.world.latest_image
    left_safe = process_camera_image(latest_image_left) if latest_image_left else []

    # Turn right (40 degrees total from left) and capture the right image
    robot.turn_in_place(degrees(-10)).wait_for_completed()
    latest_image_right = robot.world.latest_image
    right_safe = process_camera_image(latest_image_right) if latest_image_right else []

    # Return to center
    robot.turn_in_place(degrees(5)).wait_for_completed()

    # Compare and move to the safest direction
    best_direction = compare_safe_areas(left_safe, right_safe)

    if best_direction == "left":
        print("Turning left and moving to the safe zone...")
        robot.turn_in_place(degrees(45)).wait_for_completed()
    else:
        print("Turning right and moving to the safe zone...")
        robot.turn_in_place(degrees(-45)).wait_for_completed()

    robot.drive_straight(distance_mm(200), speed_mmps(50)).wait_for_completed()

    return True


def move_to_coordinates_with_cliff_handling(robot: cozmo.robot.Robot, x, y):
    """Move to specific coordinates while checking for cliffs continuously."""
    print("Moving to target coordinates...")
    move_speed = speed_mmps(30)  # Set constant move speed

    while True:
        robot.drive_wheels(move_speed.speed_mmps, move_speed.speed_mmps)

        # Continuously check for cliff
        if robot.is_cliff_detected:
            handle_cliff(robot)
            continue

        # Stop at target
        robot_x = robot.pose.position.x
        robot_y = robot.pose.position.y
        distance = math.sqrt((x - robot_x)**2 + (y - robot_y)**2)
        if distance <= 80:
            robot.stop_all_motors()
            print(f"Reached target at x={x:.2f}, y={y:.2f}")
            break


def cozmo_program(robot: cozmo.robot.Robot):
    """Main program to move Cozmo and avoid cliffs."""
    try:
        robot.set_head_angle(degrees(5)).wait_for_completed()
        print("Starting movement with immediate cliff handling...")
        move_to_coordinates_with_cliff_handling(robot, 600, 0)
        print("Successfully reached the target!")
    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        print("Shutting down...")
        cv2.destroyAllWindows()


cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)

