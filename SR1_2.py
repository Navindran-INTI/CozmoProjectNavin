import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
import math
from cozmo.objects import CustomObject, CustomObjectMarkers, CustomObjectTypes
import threading

# Parameters
spin_increment = 20  # Increment angle for 360-degree spin
search_distance = 200  # Distance to drive towards walls if no cube is found
safety_distance = 100  # Minimum distance to maintain from objects
slow_approach_speed = 30  # Speed when approaching objects
cliff_buffer_distance = 50  # Buffer distance (in mm) to avoid cliffs
cliff_reverse_distance = 100  # Distance to reverse when a cliff is detected (in mm)


# Global variables to store object and robot data
robot_frames = []  # List to store robot positions
object_frames = {}  # Dictionary to store detected objects' frames
cliff_positions = []  # List to store cliff coordinates
starting_position = (0, 0)  # Cozmo's starting position

def calculate_distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def record_robot_frame(robot):
    """Record the current robot frame."""
    global robot_frames
    robot_frames.append((robot.pose.position.x, robot.pose.position.y))
    print(f"Recorded robot frame: x={robot.pose.position.x:.2f}, y={robot.pose.position.y:.2f}")

def is_cliff_nearby(robot):
    """Check if Cozmo's cliff sensors detect a drop-off."""
    return robot.is_cliff_detected

def record_cliff_position(robot):
    """Record the current position as a cliff position."""
    global cliff_positions
    cliff_x = robot.pose.position.x
    cliff_y = robot.pose.position.y
    cliff_positions.append((cliff_x, cliff_y))
    print(f"Cliff detected! Recorded cliff position: x={cliff_x:.2f}, y={cliff_y:.2f}")

def is_position_safe(x, y):
    """Check if a position is safe by ensuring it is not near a cliff."""
    global cliff_positions
    for cliff_x, cliff_y in cliff_positions:
        distance_to_cliff = calculate_distance(x, y, cliff_x, cliff_y)
        if distance_to_cliff < cliff_buffer_distance:
            print(f"Unsafe position near cliff detected at x={cliff_x:.2f}, y={cliff_y:.2f}. Distance: {distance_to_cliff:.2f}")
            return False
    return True



def create_cozmo_walls(robot: cozmo.robot.Robot):
    """Define custom walls and add them to Cozmo's world."""
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


def move_to_object(robot: cozmo.robot.Robot, target_object):
    """Move the robot towards the target object."""
    if target_object:
        # Extract target coordinates
        target_x = target_object.pose.position.x
        target_y = target_object.pose.position.y

        # Extract robot's current coordinates
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

        print(f"Moving towards object at x={target_x:.2f}, y={target_y:.2f}")
        print(f"Movement details: dx={dx:.2f}, dy={dy:.2f}, distance={distance:.2f}, angle_to_turn={angle_to_turn:.2f}°")

        # Turn to face the target
        robot.turn_in_place(degrees(angle_to_turn)).wait_for_completed()

        # Drive straight to the target, checking for cliffs
        while distance > safety_distance:
            if is_cliff_nearby(robot):
                print("Cliff detected! Reversing and avoiding.")
                record_cliff_position(robot)
                robot.drive_straight(distance_mm(-cliff_reverse_distance), speed_mmps(50)).wait_for_completed()
                return  # Resume searching after avoiding the cliff
            robot.drive_straight(distance_mm(min(50, distance)), speed_mmps(slow_approach_speed)).wait_for_completed()
            distance -= 50  # Update remaining distance

        # Record the robot's new position after movement
        record_robot_frame(robot)
    else:
        print("No target object specified.")

def pick_and_deposit_cube(robot: cozmo.robot.Robot, cube):
    """Pick up the cube and deposit it at the starting position."""
    print("Picking up the cube...")
    robot.pickup_object(cube, num_retries=3).wait_for_completed()
    
    print("Returning to the starting position to deposit the cube...")
    move_to_coordinates(robot, starting_position[0], starting_position[1])
    
    print("Depositing the cube...")
    robot.place_object_on_ground_here(cube).wait_for_completed()
    robot.set_lift_height(0.0).wait_for_completed()
    print("Cube deposited.")
    
def move_to_coordinates(robot: cozmo.robot.Robot, x, y):
    """Move the robot to specified (x, y) coordinates."""
    robot_x = robot.pose.position.x
    robot_y = robot.pose.position.y
    dx = x - robot_x
    dy = y - robot_y
    distance = math.sqrt(dx**2 + dy**2)
    target_angle = math.degrees(math.atan2(dy, dx))
    angle_to_turn = target_angle - robot.pose.rotation.angle_z.degrees
    angle_to_turn = (angle_to_turn + 180) % 360 - 180
    robot.turn_in_place(degrees(angle_to_turn)).wait_for_completed()
    robot.drive_straight(distance_mm(distance), speed_mmps(50)).wait_for_completed()
    


def search_360(robot: cozmo.robot.Robot):
    """
    Perform a 360-degree spin to detect objects.
    Consolidate all objects seen during the spin and prioritize cubes.
    If no cubes are detected, prioritize the furthest wall.
    """
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
    """Log Cozmo's current pose every 2 seconds."""
    while True:
        pose = robot.pose
        print(f"Cozmo's current pose: x={pose.position.x:.2f}, y={pose.position.y:.2f}, rotation={pose.rotation.angle_z.degrees:.2f}°")
        threading.Event().wait(2)


def main(robot: cozmo.robot.Robot):
    """Main program to execute Cozmo's object detection and movement behavior."""
    pose_logger = threading.Thread(target=log_cozmo_pose, args=(robot,))
    pose_logger.start()

    try:
        robot.set_head_angle(degrees(5)).wait_for_completed()
        robot.set_lift_height(0.0).wait_for_completed()

        # Create walls in Cozmo's environment
        create_cozmo_walls(robot)

        # Record the initial robot frame
        record_robot_frame(robot)

        while True:
            target_object = search_360(robot)

            if target_object:
                if isinstance(target_object, cozmo.objects.LightCube):
                    pick_and_deposit_cube(robot, target_object)
                else:
                    # Otherwise, move to the target object
                    move_to_object(robot, target_object)
                print("Target reached. Restarting search.")
            else:
                print("No objects detected. Continuing to spin indefinitely.")
    finally:
        pose_logger.join()


cozmo.run_program(main, use_3d_viewer=True, use_viewer=True)
