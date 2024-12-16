import cozmo
import cv2
import numpy as np
import math
from cozmo.util import degrees, distance_mm, speed_mmps

# Global variable to store detected cliffs based on camera feed
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

    # Find contours of dark regions
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Register dark areas as cliffs and visually mark them
    for contour in contours:
        # Approximate the contour area to eliminate noise
        area = cv2.contourArea(contour)
        if area > 300:  # Reduced threshold to include smaller areas
            print("Detected potential cliff in camera feed.")
            detected_cliffs.append(contour)
            # Draw the detected contour on the original image
            cv2.drawContours(lower_half, [contour], -1, (0, 0, 255), 2)  # Red for cliffs

    # Update the lower-half back into the main image
    opencv_image[int(height * 0.5):, :] = lower_half

    # Debugging: Display the final image with cliffs marked
    cv2.imshow("Cliff Detection - Visual Representation", opencv_image)
    cv2.waitKey(1)


def handle_cliff(robot: cozmo.robot.Robot):
    """Handle cliff detection using Cozmo's cliff sensors and camera."""
    if robot.is_cliff_detected:
        print("Cliff detected via sensors. Reversing and adjusting for analysis...")

        # Reverse to avoid the cliff
        robot.drive_straight(distance_mm(-100), speed_mmps(50)).wait_for_completed()
        print("Reversing complete.")

        # Raise Cozmo's arms to avoid interference with the camera feed
        robot.set_lift_height(1.0).wait_for_completed()
        print("Arms raised to avoid blocking the camera.")

        # Adjust the head angle to look at the ground
        robot.set_head_angle(degrees(-5)).wait_for_completed()
        print("Head angle adjusted to -5°.")

        # Capture the camera feed for processing
        latest_image = robot.world.latest_image
        if latest_image:
            process_camera_image(latest_image)
        else:
            print("No camera image available for analysis.")

        # Lower the arms back to the resting position
        robot.set_lift_height(0.0).wait_for_completed()
        print("Arms returned to resting position.")

        return True  # Indicate cliff was handled
    return False

def move_to_coordinates_with_cliff_handling(robot: cozmo.robot.Robot, x, y):
    """Move to specific coordinates while handling cliffs visually and via sensors."""
    global detected_cliffs

    while True:
        # Get current robot position
        robot_x = robot.pose.position.x
        robot_y = robot.pose.position.y
        dx = x - robot_x
        dy = y - robot_y
        distance = math.sqrt(dx**2 + dy**2)

        # Stop when within 80 mm of the target
        if distance <= 80:
            print(f"Reached target at x={x:.2f}, y={y:.2f}")
            break

        # Check for cliffs via sensors and camera
        if handle_cliff(robot):
            print("Restarting due to detected cliff.")
            return  # Exit the function, assuming the movement is restarted

        # Calculate angle to the target
        target_angle = math.degrees(math.atan2(dy, dx))
        angle_to_turn = target_angle - robot.pose.rotation.angle_z.degrees
        angle_to_turn = (angle_to_turn + 180) % 360 - 180  # Normalize angle to [-180, 180]

        # Turn to face the target
        if abs(angle_to_turn) > 5:
            print(f"Turning {angle_to_turn:.2f}° to face the target.")
            robot.turn_in_place(degrees(angle_to_turn)).wait_for_completed()

        # Move in small increments towards the target
        move_distance = min(30, distance)
        print(f"Driving forward by {move_distance:.2f} mm towards x={x:.2f}, y={y:.2f}")
        robot.drive_straight(distance_mm(move_distance), speed_mmps(30)).wait_for_completed()

        # Update current position after moving
        robot_x = robot.pose.position.x
        robot_y = robot.pose.position.y
        dx = x - robot_x
        dy = y - robot_y
        distance = math.sqrt(dx**2 + dy**2)

    print(f"Precisely reached target at x={x:.2f}, y={y:.2f}")

def cozmo_program(robot: cozmo.robot.Robot):
    """Cozmo main program to test cliff handling and coordinate movement."""
    try:
        robot.set_head_angle(degrees(5)).wait_for_completed()
        print("Starting Cozmo's movement test with cliff handling...")

        # Test moving to a target coordinate with cliff handling
        move_to_coordinates_with_cliff_handling(robot, 1000, 0)  # Example target
        print("Successfully reached the target!")

        # Keep displaying the cliff detection visualization for a few seconds
        print("Keeping the visualization open for 100 seconds...")
        for _ in range(1000):
            cv2.waitKey(100)  # Allow time for visualization (100 seconds)

    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        print("Shutting down...")
        print("Detected cliffs:", detected_cliffs)
        cv2.destroyAllWindows()

cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)

