import cozmo
import cv2
import numpy as np
import math
from cozmo.util import degrees, distance_mm, speed_mmps
import time
import os  # For creating directories and saving images

# Global variable to store detected cliffs as avoidance zones
detected_cliffs = []

def process_camera_image(image, capture_delay=3, save_path=None, filename="image"):
    """Process the camera image to detect dark areas (potential cliffs), visually mark them, and optionally save them."""
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
    cliffs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            cliffs.append((x, y, w, h))  # Store cliff bounding box
            cv2.drawContours(lower_half, [contour], -1, (0, 0, 255), 2)  # Mark cliffs in red

    # Highlight remaining safe areas
    safe_mask = cv2.bitwise_not(binary_image)
    contours_safe, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    safe_areas = []
    for contour in contours_safe:
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            safe_areas.append((x, y, w, h))  # Safe zone bounding box

    # Display processed image
    cv2.imshow("Cliff Detection - Visual Representation", opencv_image)
    print(f"Displaying image for {capture_delay} seconds.")
    time.sleep(capture_delay)  # Pause to view the image
    cv2.waitKey(1)

    # Save the image if a save path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        image_path = os.path.join(save_path, f"{filename}.png")
        cv2.imwrite(image_path, opencv_image)
        print(f"Image saved as {image_path}")

    return cliffs, safe_areas


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
    """Immediately stop motion when a cliff is detected, reverse, and process safe zones."""
    print("Cliff detected! Stopping movement immediately...")
    robot.stop_all_motors()

    # Double-check cliff detection before proceeding
    if robot.is_cliff_detected:
        print("Confirmed cliff detection. Reversing...")
        robot.drive_straight(distance_mm(-100), speed_mmps(50)).wait_for_completed()
        print("Reversed successfully.")


        # Raise Cozmo's arms to avoid interference with the camera feed
        robot.set_lift_height(1.0).wait_for_completed()
        print("Arms raised to avoid blocking the camera.")
        

        # Set up for image processing
        robot.set_head_angle(degrees(-5)).wait_for_completed()

        # Process center (cliff) image
        latest_image_center = robot.world.latest_image
        center_cliffs, _ = process_camera_image(latest_image_center, capture_delay=1, save_path="images", filename="center") if latest_image_center else ([], [])
        print("Processed center image for cliffs.")

        # Turn left and capture the left image
        robot.turn_in_place(degrees(10)).wait_for_completed()
        latest_image_left = robot.world.latest_image
        _, left_safe = process_camera_image(latest_image_left, capture_delay=1 ,save_path="images", filename="left") if latest_image_left else ([], [])
        print("Processed left image for safe zones.")

        # Turn right (20 degrees total from left) and capture the right image
        robot.turn_in_place(degrees(-20)).wait_for_completed()
        latest_image_right = robot.world.latest_image
        _, right_safe = process_camera_image(latest_image_right, capture_delay=1,save_path="images", filename="right") if latest_image_right else ([], [])
        print("Processed right image for safe zones.")

        # Return to center
        robot.turn_in_place(degrees(10)).wait_for_completed()

        # Compare safe zones and decide direction
        best_direction = compare_safe_areas(left_safe, right_safe)

        # Lower the arms back to the resting position
        robot.set_lift_height(0.0).wait_for_completed()
        print("Arms returned to resting position.")

        if best_direction == "left":
            print("Turning left and moving to the safe zone...")
            robot.turn_in_place(degrees(45)).wait_for_completed()
        else:
            print("Turning right and moving to the safe zone...")
            robot.turn_in_place(degrees(-45)).wait_for_completed()

        # Drive towards the safe zone
        robot.drive_straight(distance_mm(200), speed_mmps(50)).wait_for_completed()

        return True
    else:
        print("No cliff detected upon recheck. Resuming normal operation.")
        return False


def move_to_coordinates_with_cliff_handling(robot: cozmo.robot.Robot, x, y):
    """Move to specific coordinates while checking for cliffs continuously."""
    print("Moving to target coordinates...")
    move_speed = speed_mmps(30)  # Set constant move speed

    while True:
        robot.drive_wheels(move_speed.speed_mmps, move_speed.speed_mmps)

        # Continuously check for cliffs while driving
        if robot.is_cliff_detected:
            handle_cliff(robot)
            continue

        # Check if target is reached
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

