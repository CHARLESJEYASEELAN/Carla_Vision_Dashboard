import carla
import pygame
import numpy as np
import random



# ==============================
# CONFIG
# ==============================
WIDTH, HEIGHT = 500, 400
WINDOW_WIDTH = WIDTH * 2    # 800
WINDOW_HEIGHT = HEIGHT * 2  # 600
SCALE = 5
MAX_DISTANCE = 50

# ==============================
# CONNECT TO CARLA
# ==============================
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.load_world('Town05')
blueprints = world.get_blueprint_library()

# ==============================
# SPAWN VEHICLES
# ==============================
spawn_points = world.get_map().get_spawn_points()

ego_bp = blueprints.filter('vehicle.tesla.model3')[0]
ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))

actors_list = [ego_vehicle]

traffic_manager = client.get_trafficmanager()
tm_port = traffic_manager.get_port()

ego_vehicle.set_autopilot(True, tm_port)

for _ in range(20):
    bp = random.choice(blueprints.filter('vehicle.*'))
    spawn = random.choice(spawn_points)
    try:
        v = world.spawn_actor(bp, spawn)
        v.set_autopilot(True, tm_port)
        actors_list.append(v)
    except:
        pass

print("Ego vehicle spawned!")

# ==============================
# SEMANTIC CAMERA (TESLA STYLE)
# ==============================
camera_bp = blueprints.find('sensor.camera.semantic_segmentation')
camera_bp.set_attribute('image_size_x', str(WIDTH))
camera_bp.set_attribute('image_size_y', str(HEIGHT))
camera_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(
    carla.Location(x=-10, z=5),    # Lower z from 10 to 5
    carla.Rotation(pitch=-10)      # Reduce pitch from -25 to -10
)

camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
actors_list.append(camera)

camera_image = None

def process_image(image):
    global camera_image

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))

    labels = array[:, :, 2]  # Use red channel for tag values

    output = np.zeros((image.height, image.width, 3), dtype=np.uint8)

    # Roads
    output[labels == 1] = [128, 64, 128]    # purple

    # RoadLines
    output[labels == 24] = [157, 234, 50]   # green-yellow

    # Poles
    output[labels == 6] = [255, 255, 255]   # white

    # Vehicles (all categories)
    output[labels == 14] = [0, 0, 142]      # car (blue)
    output[labels == 15] = [0, 0, 70]       # truck (dark blue)
    output[labels == 16] = [0, 60, 100]     # bus (teal)
    output[labels == 17] = [0, 80, 100]     # train (teal)
    output[labels == 18] = [0, 0, 230]      # motorcycle (bright blue)
    output[labels == 19] = [119, 11, 32]    # bicycle (red)

    camera_image = np.rot90(output)

camera.listen(lambda image: process_image(image))

# RGB CAMERA SETUP
rgb_camera_bp = blueprints.find('sensor.camera.rgb')
rgb_camera_bp.set_attribute('image_size_x', str(WIDTH))
rgb_camera_bp.set_attribute('image_size_y', str(HEIGHT))
rgb_camera_bp.set_attribute('fov', '90')

rgb_camera_transform = carla.Transform(
    carla.Location(x=-10, z=5),    # Lower z from 10 to 5
    carla.Rotation(pitch=-10)      # Reduce pitch from -25 to -10
)

rgb_camera = world.spawn_actor(rgb_camera_bp, rgb_camera_transform, attach_to=ego_vehicle)
actors_list.append(rgb_camera)

rgb_camera_image = None

def process_rgb_image(image):
    global rgb_camera_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    # Convert BGRA to RGB
    rgb_array = array[:, :, [2, 1, 0]]  # R, G, B
    rgb_camera_image = np.rot90(rgb_array)

rgb_camera.listen(lambda image: process_rgb_image(image))

# ==============================
# PYGAME INIT
# ==============================
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Semantic Dashboard")

clock = pygame.time.Clock()

# ==============================
# UTILS
# ==============================
def get_nearby_vehicles(ego, world):
    vehicles = world.get_actors().filter('vehicle.*')
    ego_tf = ego.get_transform()

    nearby = []
    for v in vehicles:
        if v.id == ego.id:
            continue

        dist = ego_tf.location.distance(v.get_transform().location)
        if dist < MAX_DISTANCE:
            nearby.append((v, dist))

    return nearby


def to_ego_frame(ego_tf, target_tf):
    dx = target_tf.location.x - ego_tf.location.x
    dy = target_tf.location.y - ego_tf.location.y

    yaw = np.radians(ego_tf.rotation.yaw)

    x = dx * np.cos(-yaw) - dy * np.sin(-yaw)
    y = dx * np.sin(-yaw) + dy * np.cos(-yaw)

    return x, y


def compute_ttc(ego, target):
    ego_v = ego.get_velocity()
    tar_v = target.get_velocity()

    rel_vx = ego_v.x - tar_v.x
    rel_vy = ego_v.y - tar_v.y

    rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)

    dist = ego.get_transform().location.distance(target.get_transform().location)

    if rel_speed < 0.1:
        return float('inf')

    return dist / rel_speed


def draw_vehicle(surface, x, y, yaw, color):
    rect = pygame.Surface((10, 20))
    rect.fill(color)

    rotated = pygame.transform.rotate(rect, yaw)
    rect_rect = rotated.get_rect(center=(x, y))

    surface.blit(rotated, rect_rect.topleft)


def draw_lanes(surface, ego_tf, world):
    carla_map = world.get_map()

    waypoint = carla_map.get_waypoint(
        ego_tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    # Get more waypoints forward
    next_wps = []
    wp = waypoint
    for _ in range(50):
        wps = wp.next(2.0)
        if not wps:
            break
        wp = wps[0]
        next_wps.append(wp)

    # Get more waypoints backward
    prev_wps = []
    wp = waypoint
    for _ in range(50):
        wps = wp.previous(2.0)
        if not wps:
            break
        wp = wps[0]
        prev_wps.append(wp)

    all_wps = prev_wps[::-1] + [waypoint] + next_wps

    lane_points = []
    for wp in all_wps:
        x, y = to_ego_frame(ego_tf, wp.transform)
        screen_x = int(WIDTH // 2 + x * SCALE)
        screen_y = int((HEIGHT // 4) - y * SCALE)
        lane_points.append((screen_x, screen_y))

    if len(lane_points) > 1:
        pygame.draw.lines(surface, (255, 255, 255), False, lane_points, 2)

    # boundaries + direction markers
    for i, wp in enumerate(all_wps):
        right = wp.transform.get_right_vector()
        lane_width = wp.lane_width / 2

        left_loc = wp.transform.location - right * lane_width
        right_loc = wp.transform.location + right * lane_width

        lx, ly = to_ego_frame(ego_tf, carla.Transform(left_loc))
        rx, ry = to_ego_frame(ego_tf, carla.Transform(right_loc))

        lx = int(WIDTH // 2 + lx * SCALE)
        ly = int((HEIGHT // 4) - ly * SCALE)
        rx = int(WIDTH // 2 + rx * SCALE)
        ry = int((HEIGHT // 4) - ry * SCALE)

        pygame.draw.circle(surface, (100, 100, 100), (lx, ly), 2)
        pygame.draw.circle(surface, (100, 100, 100), (rx, ry), 2)

        # direction markers
        if i % 5 == 0:
            pygame.draw.circle(surface, (0, 255, 0), (lx, ly), 3)

        # --- INTERSECTION BRANCHES ---
        branches = wp.next(2.0)
        if len(branches) > 1:
            for branch_wp in branches:
                branch_points = []
                b_wp = branch_wp
                for _ in range(10):  # Draw branch for 10 steps
                    b_next = b_wp.next(2.0)
                    if not b_next:
                        break
                    b_wp = b_next[0]
                    bx, by = to_ego_frame(ego_tf, b_wp.transform)
                    b_screen_x = int(WIDTH // 2 + bx * SCALE)
                    b_screen_y = int((HEIGHT // 4) - by * SCALE)
                    branch_points.append((b_screen_x, b_screen_y))
                if len(branch_points) > 1:
                    pygame.draw.lines(surface, (255, 0, 255), False, branch_points, 2)  # Magenta for intersection

# ==============================
# MAIN LOOP
# ==============================
running = True
try:
    while running:
        clock.tick(20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ego_tf = ego_vehicle.get_transform()

        # Top-left: Semantic camera
        if camera_image is not None:
            cam_surface = pygame.surfarray.make_surface(camera_image)
            cam_surface = pygame.transform.scale(cam_surface, (WIDTH, HEIGHT))
            screen.blit(cam_surface, (0, 0))

        # Top-right: RGB camera
        if rgb_camera_image is not None:
            rgb_surface = pygame.surfarray.make_surface(rgb_camera_image)
            rgb_surface = pygame.transform.scale(rgb_surface, (WIDTH, HEIGHT))
            screen.blit(rgb_surface, (WIDTH, 0))

        # Bottom-left: Dashboard
        dashboard = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT // 2))
        dashboard.fill((10, 10, 10))
        draw_lanes(dashboard, ego_tf, world)
        ego_x, ego_y = WIDTH // 2, HEIGHT // 4
        draw_vehicle(dashboard, ego_x, ego_y, 0, (0, 255, 0))
        pygame.draw.line(dashboard, (0, 255, 0), (ego_x, ego_y), (ego_x, ego_y - 20), 3)
        pygame.draw.circle(dashboard, (255, 0, 0), (ego_x, ego_y - 50), 30, 2)
        nearby = get_nearby_vehicles(ego_vehicle, world)
        for v, dist in nearby:
            v_tf = v.get_transform()
            x, y = to_ego_frame(ego_tf, v_tf)
            screen_x = int(WIDTH // 2 + x * SCALE)
            screen_y = int((HEIGHT // 4) - y * SCALE)
            if not (0 < screen_x < WIDTH and 0 < screen_y < HEIGHT // 2):
                continue
            ttc = compute_ttc(ego_vehicle, v)
            if ttc < 2:
                color = (255, 0, 0)
            elif ttc < 5:
                color = (255, 165, 0)
            else:
                color = (0, 0, 255)
            rel_yaw = v_tf.rotation.yaw - ego_tf.rotation.yaw
            draw_vehicle(dashboard, screen_x, screen_y, rel_yaw, color)
            vel = v.get_velocity()
            vx, vy = vel.x, vel.y
            yaw = np.radians(ego_tf.rotation.yaw)
            vx_ego = vx * np.cos(-yaw) - vy * np.sin(-yaw)
            vy_ego = vx * np.sin(-yaw) + vy * np.cos(-yaw)
            end_x = int(screen_x + vx_ego * 2)
            end_y = int(screen_y - vy_ego * 2)
            pygame.draw.line(dashboard, (0, 255, 255), (screen_x, screen_y), (end_x, end_y), 2)
            for t in range(1, 6):
                fx = screen_x + vx_ego * t * 2
                fy = screen_y - vy_ego * t * 2
                pygame.draw.circle(dashboard, (255, 255, 0), (int(fx), int(fy)), 2)
        # Center the dashboard in the bottom row
        screen.blit(dashboard, (0, HEIGHT))

        flipped_screen = pygame.transform.flip(screen, True, False)
        pygame.display.get_surface().blit(flipped_screen, (0, 0))

        pygame.display.flip()
finally:
    print("Cleaning up actors...")
    for actor in actors_list:
        actor.destroy()
    pygame.quit()