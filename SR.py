import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
import math
from frame2d import Frame2D
from map2_coordinates import robotFrames, cubeFrames, wallFrames,cliffSensor
import sys
import numpy as np
import heapq

# Parameters
wall_buffer = 40  # Minimum distance to walls MAP 2 40
proximity_threshold = 80  # Proximity to cube MAP

# Validate arguments
if len(sys.argv) < 2:
    print("Please provide the cube you are looking for. Example: python program.py cube1")
    sys.exit(1)

# Get the target cube from the arguments
target_cube = sys.argv[1]
print(f"Searching for: {target_cube}")

# Mapping cube names to Cozmo cube IDs
first_cube_id = min(cube_id for cube_id, _ in cubeFrames)
cube_mapping = {
    "cube1": first_cube_id,
    "cube2": first_cube_id + 1,
    "cube3": first_cube_id + 2
}

if target_cube not in cube_mapping:
    print(f"Invalid cube name: {target_cube}. Use: cube1, cube2, or cube3.")
    sys.exit(1)

target_cube_id = cube_mapping[target_cube]

# Walls list from wallFrames
wall_frames = [f for walls in wallFrames.values() for f in walls]
cliff_frames = [frame for t, cliff in cliffSensor if cliff and t in robotFrames]
obstacles = wall_frames + cliff_frames
designated_cube_frames = [frame for cube_id, frame in cubeFrames if cube_id == target_cube_id]

# Average coordinates of the designated cube frames
avg_x = sum(frame.x() for frame in designated_cube_frames) / len(designated_cube_frames)
avg_y = sum(frame.y() for frame in designated_cube_frames) / len(designated_cube_frames)

# A* Algorithm
def heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
def is_position_safe(x, y, walls, cliffs, buffer_distance):
    """Check if a position is safe (not too close to walls or cliffs)."""
    # Check walls
    for wall in walls:
        if np.hypot(wall.x() - x, wall.y() - y) < buffer_distance:
            return False
    
    # Check cliffs
    for cliff in cliffs:
        if np.hypot(cliff.x() - x, cliff.y() - y) < buffer_distance:
            return False
    
    return True

    
def is_trajectory_safe(start, end, walls, cliffs, buffer_distance):
    """Check if the trajectory between two points avoids walls and cliffs."""
    for obstacle in walls + cliffs:
        obs_x, obs_y = obstacle.x(), obstacle.y()
        start_to_end = np.array([end[0] - start[0], end[1] - start[1]])
        start_to_obs = np.array([obs_x - start[0], obs_y - start[1]])
        projection = np.dot(start_to_obs, start_to_end) / np.linalg.norm(start_to_end)**2
        closest_point = (
            start[0] + projection * start_to_end[0],
            start[1] + projection * start_to_end[1]
        )
        distance_to_obstacle = np.linalg.norm([closest_point[0] - obs_x, closest_point[1] - obs_y])
        if 0 <= projection <= 1 and distance_to_obstacle < buffer_distance:
            return False  # Trajectory intersects or is too close to an obstacle
    return True


def a_star_pathfinding_with_safety(start, goal, robot_frames, walls, cliffs):
    """Improved A* algorithm with trajectory validation."""
    valid_nodes = [
        (frame.x(), frame.y()) for _, frame in robot_frames
        if is_position_safe(frame.x(), frame.y(), walls, cliffs, wall_buffer)  # Pasa los cliffs aquí
    ]
    valid_nodes.append(start)
    valid_nodes.append(goal)

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            break

        for neighbor in valid_nodes:
            if neighbor == current or not is_position_safe(neighbor[0], neighbor[1], walls, cliffs, wall_buffer):
                continue

            # Validate trajectory safety
            if not is_trajectory_safe(current, neighbor, walls, cliffs, wall_buffer):  # Incluye cliffs aquí también
                continue

            distance = heuristic(current, neighbor)
            if distance > 300:  # Threshold for valid connections
                continue

            new_cost = cost_so_far[current] + distance
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current

    # Reconstruct path
    path = []
    current = goal
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    second_last = path[-2]
    last = path[-1]
    averaged_point = ((second_last[0] + last[0]) / 2, (second_last[1] + last[1]) / 2)
    path[-1] = averaged_point
    return path


def retrace_path(robot: cozmo.robot.Robot, path):
    """Function to retrace the path back to the start."""
    reversed_path = list(reversed(path))  # Reverse the path
    print(f"Retracing path: {reversed_path}")
    follow_path(robot, reversed_path)

# Function to move the robot along the path
def follow_path(robot: cozmo.robot.Robot, path):
    for point in path[1:]:  # Skip the first point (starting position)
        dx = point[0] - robot.pose.position.x
        dy = point[1] - robot.pose.position.y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        true_distance = distance_to_target - 10
        angle_to_target = math.degrees(math.atan2(dy, dx)) - robot.pose.rotation.angle_z.degrees

        # Turn towards the target
        robot.turn_in_place(degrees(angle_to_target)).wait_for_completed()

        # Drive straight to the target
        robot.drive_straight(distance_mm(true_distance), speed_mmps(50)).wait_for_completed()

def drop_cube(robot: cozmo.robot.Robot):
    """Drop the cube currently held by Cozmo."""
    # Check for a cube being carried
    cube = robot.world.light_cubes.get(robot.carrying_object_id, None)
    if cube is not None or cube.is_visible:
        print("Dropping the cube...")
        robot.place_object_on_ground_here(cube).wait_for_completed()
        print("Cube placed on the ground.")
    else:
        print("No cube is currently being carried or it is not visible.")




def main(robot: cozmo.robot.Robot):
    # Define a list of cubes to search for
    cube_order = ["cube1", "cube2", "cube3"]
    
    # Initial starting position (could be set differently based on your environment)
    current_position = (0, 0)

    # Loop through cubes in the specified order
    for target_cube in cube_order:
        # Calculate the goal position for the current target cube
        designated_cube_frames = [frame for cube_id, frame in cubeFrames if cube_id == cube_mapping[target_cube]]
        
        if designated_cube_frames:
            avg_x = sum(frame.x() for frame in designated_cube_frames) / len(designated_cube_frames)
            avg_y = sum(frame.y() for frame in designated_cube_frames) / len(designated_cube_frames)
            goal_position = (avg_x, avg_y)
        else:
            print(f"No designated frames found for {target_cube}.")
            continue  # Skip to the next cube if no frames are found

        # Calculate path using A*
        path = a_star_pathfinding_with_safety(current_position, goal_position, robotFrames, wall_frames, cliff_frames)
 
        if not path:
            print(f"No path found to {target_cube}.")
            continue

        print(f"Path to {target_cube}: {path}")

        # Follow the calculated path
        follow_path(robot, path)

        # Find all visible cubes
        cubes = robot.world.wait_until_observe_num_objects(num=3, object_type=cozmo.objects.LightCube, timeout=60)

        # Attempt to pick up the target cube
        max_dst, targ = 0, None
        for cube in cubes:
            translation = robot.pose - cube.pose
            dst = translation.position.x ** 2 + translation.position.y ** 2
            if dst > max_dst:
                max_dst, targ = dst, cube

        if targ:
            print(f"Picking up the target cube: {target_cube}...")
            robot.pickup_object(targ, num_retries=3).wait_for_completed()
            print("Cube picked up successfully!")

            # Retrace the path to return with the cube
            retrace_path(robot, path)

            # Drop the cube
            robot.place_object_on_ground_here(targ).wait_for_completed()
            print("Cube dropped on the ground.")

            # Perform a specific turn after processing cube2 or cube3
            if target_cube == "cube2":
                robot.turn_in_place(degrees(200)).wait_for_completed()
            elif target_cube == "cube3":
                robot.turn_in_place(degrees(300)).wait_for_completed()

            # Update the current position for the next cube search
            current_position = robot.pose.position.x, robot.pose.position.y
        else:
            print(f"No target cube {target_cube} found. Exiting.")
            return

cozmo.run_program(main, use_3d_viewer=True, use_viewer=True)


