import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
import math
from cozmo.objects import CustomObject, CustomObjectMarkers, CustomObjectTypes
import threading
import cv2
import numpy as np
import time
import os 

spin_increment = 20 
search_distance = 200  
safety_distance = 100  
slow_approach_speed = 30  


path_taken = []  
robot_frames = []  
object_frames = {}  
detected_cliffs = []
starting_position = (0, 0)  

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def record_robot_pose(robot):
    global path_taken
    x, y, rotation = robot.pose.position.x, robot.pose.position.y, robot.pose.rotation.angle_z.degrees
    if not robot.is_cliff_detected:
        path_taken.append((x, y, rotation))
        print(f"Path recorded: x={x:.2f}, y={y:.2f}, rotation={rotation:.2f}")
    else:
        print("Cliff detected at current position. Path not recorded.")

def create_cozmo_walls(robot: cozmo.robot.Robot):
    types = [CustomObjectTypes.CustomType01,
             CustomObjectTypes.CustomType02,
             CustomObjectTypes.CustomType03,
             CustomObjectTypes.CustomType04,
             CustomObjectTypes.CustomType05,
             CustomObjectTypes.CustomType06,
             CustomObjectTypes.CustomType07,
             CustomObjectTypes.CustomType08,
             CustomObjectTypes.CustomType09,
             CustomObjectTypes.CustomType10,
             CustomObjectTypes.CustomType11,
             CustomObjectTypes.CustomType12,
             CustomObjectTypes.CustomType13,
             CustomObjectTypes.CustomType14,
             CustomObjectTypes.CustomType15,
             CustomObjectTypes.CustomType16]
    markers = [CustomObjectMarkers.Circles2,
               CustomObjectMarkers.Diamonds2,
               CustomObjectMarkers.Hexagons2,
               CustomObjectMarkers.Triangles2,
               CustomObjectMarkers.Circles3,
               CustomObjectMarkers.Diamonds3,
               CustomObjectMarkers.Hexagons3,
               CustomObjectMarkers.Triangles3,
               CustomObjectMarkers.Circles4,
               CustomObjectMarkers.Diamonds4,
               CustomObjectMarkers.Hexagons4,
               CustomObjectMarkers.Triangles4,
               CustomObjectMarkers.Circles5,
               CustomObjectMarkers.Diamonds5,
               CustomObjectMarkers.Hexagons5,
               CustomObjectMarkers.Triangles5]
    
    for i in range(len(types)):
        try:
            wall = robot.world.define_custom_wall(
                types[i], markers[i], 200, 60, 50, 50, is_unique=True)
            if wall:
                print(f"Defined wall: {wall}")
            else:
                print(f"Failed to define wall for type={types[i]} and marker={markers[i]}")
        except Exception as e:
            print(f"Error defining wall type={types[i]}, marker={markers[i]}: {e}")

def process_camera_image(image, capture_delay=3, save_path=None, filename="image"):
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

def calculate_total_cliff_area(cliffs):
    total_area = 0
    for x, y, w, h in cliffs:
        total_area += w * h  # Area of the bounding box
    return total_area

def compare_cliff_areas(left_cliffs, right_cliffs):
    left_cliff_area = calculate_total_cliff_area(left_cliffs)
    right_cliff_area = calculate_total_cliff_area(right_cliffs)

    print(f"Cliff area - Left: {left_cliff_area}, Right: {right_cliff_area}")

    if left_cliff_area < right_cliff_area:
        print("Left direction has the smallest cliff area. Choosing left.")
        return "left"
    else:
        print("Right direction has the smallest cliff area. Choosing right.")
        return "right"


def handle_cliff(robot: cozmo.robot.Robot):
    print("Cliff detected! Stopping movement immediately...")
    robot.stop_all_motors()

    # Double-check cliff detection before proceeding
    if robot.is_cliff_detected:
        print("Confirmed cliff detection. Reversing...")
        robot.drive_straight(distance_mm(-100), speed_mmps(50)).wait_for_completed()

        # Record the safe reversed position
        record_robot_pose(robot)
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
        left_cliffs, _ = process_camera_image(latest_image_left, capture_delay=1, save_path="images", filename="left") if latest_image_left else ([], [])
        print("Processed left image for cliffs.")

        # Turn right (20 degrees total from left) and capture the right image
        robot.turn_in_place(degrees(-20)).wait_for_completed()
        latest_image_right = robot.world.latest_image
        right_cliffs, _ = process_camera_image(latest_image_right, capture_delay=1, save_path="images", filename="right") if latest_image_right else ([], [])
        print("Processed right image for cliffs.")

        # Return to center
        robot.turn_in_place(degrees(10)).wait_for_completed()

        # Compare cliff areas and decide direction
        best_direction = compare_cliff_areas(left_cliffs, right_cliffs)

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
        robot.drive_straight(distance_mm(200), speed_mmps(30)).wait_for_completed()

        # Record the safe zone position
        record_robot_pose(robot)

        return True
    else:
        print("No cliff detected upon recheck. Resuming normal operation.")
        return False

def retrace_path(robot):
    global path_taken
    reversed_path = list(reversed(path_taken))  # Reverse the path
    print(f"Retracing filtered path: {reversed_path}")

    for x, y, rotation in reversed_path:
        move_to_coordinates(robot, x, y)
        print(f"Retraced to x={x:.2f}, y={y:.2f}, rotation={rotation:.2f}")

    print("Back at approximate starting position. Fine-tuning to exact (0, 0).")
    # Perform final fine-tuning to ensure exact (0, 0) position
    move_to_coordinates(robot, 0, 0)
    print("Precisely at starting position.")
    path_taken.clear()

def move_to_coordinates(robot: cozmo.robot.Robot, x, y):
    robot_x = robot.pose.position.x
    robot_y = robot.pose.position.y
    dx = x - robot_x
    dy = y - robot_y
    distance = math.sqrt(dx**2 + dy**2)

    while distance > 80:  # Stop when within 80 mm of the target
        # Check for cliffs
        if robot.is_cliff_detected:
            print("Cliff detected while moving to the object!")
            handle_cliff(robot)
            continue  # Restart navigation after handling the cliff

        # Calculate the angle to the target
        target_angle = math.degrees(math.atan2(dy, dx))
        angle_to_turn = target_angle - robot.pose.rotation.angle_z.degrees
        angle_to_turn = (angle_to_turn + 180) % 360 - 180  # Normalize angle to [-180, 180]

        # Turn to face the target
        robot.turn_in_place(degrees(angle_to_turn)).wait_for_completed()

        # Move in small increments for precise adjustment
        move_distance = min(30, distance)
        robot.drive_straight(distance_mm(move_distance), speed_mmps(30)).wait_for_completed()

        # Record the position only if no cliffs are detected
        record_robot_pose(robot)

        # Recalculate the remaining distance
        robot_x = robot.pose.position.x
        robot_y = robot.pose.position.y
        dx = x - robot_x
        dy = y - robot_y
        distance = math.sqrt(dx**2 + dy**2)

    print(f"Precisely reached target at x={x:.2f}, y={y:.2f}")


def move_to_object(robot: cozmo.robot.Robot, target_object):
    if target_object:
        # Extract target coordinates
        target_x = target_object.pose.position.x
        target_y = target_object.pose.position.y

        print(f"Moving towards object at x={target_x:.2f}, y={target_y:.2f}")

        while True:
            # Update robot's current coordinates
            robot_x = robot.pose.position.x
            robot_y = robot.pose.position.y
            robot_angle = robot.pose.rotation.angle_z.degrees

            # Calculate movement details
            dx = target_x - robot_x
            dy = target_y - robot_y
            distance = math.sqrt(dx**2 + dy**2)
            target_angle = math.degrees(math.atan2(dy, dx))
            angle_to_turn = target_angle - robot_angle

            # Normalize angle to the range [-180, 180]
            angle_to_turn = (angle_to_turn + 180) % 360 - 180

            # Debugging output
            print(f"Current position: x={robot_x:.2f}, y={robot_y:.2f}")
            print(f"Distance to target: {distance:.2f} mm")
            print(f"Angle to target: {angle_to_turn:.2f}°")

            # Check for cliffs
            if robot.is_cliff_detected:
                print("Cliff detected while moving to the object!")
                handle_cliff(robot)
                continue  # Restart navigation after handling the cliff


            # Check if we've reached the target
            if distance <= safety_distance:
                print("Reached target within safe distance.")
                break

            # Turn to face the target if the angle is significant
            if abs(angle_to_turn) > 5:
                print(f"Turning {angle_to_turn:.2f}° to face the target...")
                robot.turn_in_place(degrees(angle_to_turn)).wait_for_completed()

            # Move straight toward the target
            move_distance = min(30, distance - safety_distance)
            print(f"Driving forward by {move_distance:.2f} mm")
            robot.drive_straight(distance_mm(move_distance), speed_mmps(slow_approach_speed)).wait_for_completed()

            # Update the robot's pose dynamically
            record_robot_pose(robot)

        # Record the robot's final position after reaching the target
        record_robot_pose(robot)
    else:
        print("No target object specified.")



def pick_and_deposit_cube(robot, cube):
    print("Picking up the cube...")

    while True:
        if robot.is_cliff_detected:
            print("Cliff detected during approach to cube! Handling...")
            handle_cliff(robot)

            # Recalculate the cube's position and ensure the robot reorients itself
            print("Recalculating path to cube after handling cliff...")
            cube_x = cube.pose.position.x
            cube_y = cube.pose.position.y
            robot_x = robot.pose.position.x
            robot_y = robot.pose.position.y
            dx = cube_x - robot_x
            dy = cube_y - robot_y
            target_angle = math.degrees(math.atan2(dy, dx))
            angle_to_turn = target_angle - robot.pose.rotation.angle_z.degrees
            angle_to_turn = (angle_to_turn + 180) % 360 - 180  # Normalize to [-180, 180]
            print(f"Reorienting towards cube: Turning {angle_to_turn:.2f}°")
            robot.turn_in_place(degrees(angle_to_turn)).wait_for_completed()
            continue  # Restart the loop to resume approaching the cube

        else:
            try:
                cube_x = cube.pose.position.x
                cube_y = cube.pose.position.y
                robot_x = robot.pose.position.x
                robot_y = robot.pose.position.y
                dx = cube_x - robot_x
                dy = cube_y - robot_y
                target_angle = math.degrees(math.atan2(dy, dx))
                angle_to_turn = target_angle - robot.pose.rotation.angle_z.degrees
                angle_to_turn = (angle_to_turn + 180) % 360 - 180  # Normalize to [-180, 180]
                print(f"Reorienting towards cube: Turning {angle_to_turn:.2f}°")
                robot.turn_in_place(degrees(angle_to_turn)).wait_for_completed()            

                # Calculate distance to the cube
                dx = cube_x - robot_x
                dy = cube_y - robot_y
                distance_to_cube = math.sqrt(dx**2 + dy**2)

                # Slow down as Cozmo gets closer to the cube
                if distance_to_cube > 300:  # If farther than 50 mm, approach normally
                    print(f"Approaching cube... Distance: {distance_to_cube:.2f} mm")
                    robot.drive_straight(distance_mm(50), speed_mmps(50)).wait_for_completed()
                else:  # Within 70 mm, approach more slowly
                    print(f"Fine-tuning approach... Distance: {distance_to_cube:.2f} mm")
                    robot.drive_straight(distance_mm(30), speed_mmps(20)).wait_for_completed()

                # Break the loop when close enough
                if distance_to_cube <= 150:  # Stop the approach when within 30 mm
                    print("Close enough to pick up the cube.")
                    break
            except cozmo.exceptions.RobotBusy:
                print("Robot is busy, retrying approach...")

    # Attempt to pick up the cube
    while True:
        if robot.is_cliff_detected:
            print("Cliff detected during pickup! Handling...")
            handle_cliff(robot)
            continue  # Restart the pickup attempt after handling the cliff

        try:
            robot.pickup_object(cube, num_retries=3).wait_for_completed()
            print("Cube picked up successfully.")
            break  # Exit the loop once the cube is picked up
        except cozmo.exceptions.RobotBusy:
            print("Robot is busy, retrying pickup...")

    print("Returning to the starting position to deposit the cube...")
    retrace_path(robot)

    # Attempt to deposit the cube
    print("Depositing the cube...")
    while True:
        if robot.is_cliff_detected:
            print("Cliff detected during deposit! Handling...")
            handle_cliff(robot)

            # Recalculate path to deposit location
            print("Recalculating path to deposit location after handling cliff...")
            continue
        else:
            try:
                robot.place_object_on_ground_here(cube).wait_for_completed()
                robot.set_lift_height(0.0).wait_for_completed()
                print("Cube deposited successfully.")
                break  # Exit the loop once the cube is deposited
            except cozmo.exceptions.RobotBusy:
                print("Robot is busy, retrying deposit...")



def search_360(robot: cozmo.robot.Robot):
    print("Starting a 360-degree search...")
    all_visible_objects = []
    cubes_seen = []
    walls_seen = []

    for _ in range(360 // spin_increment):
        robot.turn_in_place(degrees(spin_increment)).wait_for_completed()
        visible_objects = robot.world.visible_objects

        for obj in visible_objects:
            if isinstance(obj, cozmo.objects.LightCube):
                # Add cube to list of cubes seen
                cubes_seen.append(obj)
            elif isinstance(obj, CustomObject):
                # Add wall to list of walls seen
                walls_seen.append(obj)

        # Add all objects to the global list
        all_visible_objects.extend(visible_objects)

    # Consolidate results after full spin
    if cubes_seen:
        # Choose the closest cube
        closest_cube = min(
            cubes_seen,
            key=lambda cube: calculate_distance(
                robot.pose.position.x, robot.pose.position.y, cube.pose.position.x, cube.pose.position.y
            )
        )
        print(f"Target object (cube): id={closest_cube.object_id}, x={closest_cube.pose.position.x:.2f}, y={closest_cube.pose.position.y:.2f}")
        return closest_cube

    if walls_seen:
        # Choose the furthest wall from Cozmo's current position
        furthest_wall = max(
            walls_seen,
            key=lambda wall: calculate_distance(
                robot.pose.position.x, robot.pose.position.y, wall.pose.position.x, wall.pose.position.y
            )
        )
        print(f"Target object (wall): id={furthest_wall.object_id}, x={furthest_wall.pose.position.x:.2f}, y={furthest_wall.pose.position.y:.2f}")
        return furthest_wall

    print("No objects detected during the 360-degree search.")
    return None


def log_cozmo_pose(robot):
    while True:
        pose = robot.pose
        print(f"Cozmo's current pose: x={pose.position.x:.2f}, y={pose.position.y:.2f}, rotation={pose.rotation.angle_z.degrees:.2f}°")
        threading.Event().wait(2)


def main(robot: cozmo.robot.Robot):
    pose_logger = threading.Thread(target=log_cozmo_pose, args=(robot,))
    pose_logger.start()

    # Counter to track the number of cube pickups and deposits
    cube_pickup_count = 0
    max_cubes = 3 

    try:
        robot.set_head_angle(degrees(5)).wait_for_completed()
        robot.set_lift_height(0.0).wait_for_completed()

        # Create walls in Cozmo's environment
        create_cozmo_walls(robot)

        # Record the initial robot frame
        record_robot_pose(robot)

        while cube_pickup_count < max_cubes:  # Limit the number of cube pickups
            target_object = search_360(robot)

            if target_object:
                if isinstance(target_object, cozmo.objects.LightCube):
                    # Perform the pickup and deposit process
                    pick_and_deposit_cube(robot, target_object)
                    cube_pickup_count += 1  # Increment the counter
                    print(f"Cube picked up and deposited. Total completed: {cube_pickup_count}/{max_cubes}")
                else:
                    # Otherwise, move to the target object
                    move_to_object(robot, target_object)
                print("Target reached. Restarting search.")
            else:
                print("No objects detected. Continuing to spin indefinitely.")

        print("All cubes have been picked up and deposited. Stopping the program.")
    finally:
        pose_logger.join()



cozmo.run_program(main, use_3d_viewer=True, use_viewer=True)

