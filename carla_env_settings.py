# 23319/usr/bin/env python
# Edited by Yuan Zheng
# Carla environment settings
CARLA_PATH = 'D:/Carla/CARLA911/WindowsNoEditor'  # Path to Carla root folder
CARLA_HOSTS_TYPE = 'local'  # 'local' or 'remote', 'local' means that script can start and restart Carla Simulator
CARLA_HOSTS_NO = 1
CARLA_HOSTS = [['localhost', 2000, 10], ['localhost', 2002, 10]]  # List of hosts and ports and worlds to use, at least 2 ports of difference as Carla uses N and N+1 port, Town01 to Town97 for world currently, Town01 to Town07 for world are currently available, int number instead - random world change interval in minutes
SECONDS_PER_EPISODE = 10
EPISODE_FPS = 60  # Desired
IMG_WIDTH = 480
IMG_HEIGHT = 270
CAR_NPCS = 50
RESET_CAR_NPC_EVERY_N_TICKS = 1  # Resets one car NPC every given number of ticks, tick is about a second
ACTIONS = ['left-action', 'straight-action', 'right-action']  # ['forward', 'left', 'right', 'forward_left', 'forward_right', 'backwards', 'backwards_left', 'backwards_right']
WEIGHT_REWARDS_WITH_EPISODE_PROGRESS = False  # Linearly weights rewards from 0 to 1 with episode progress (from 0 up to SECONDS_PER_EPISODE)
WEIGHT_REWARDS_WITH_SPEED = 'linear'  # 'discrete': -1 < 50kmh, 1 otherwise, 'linear': -1..1 with 0..100kmh, 'quadratic': -1..1 with 0..100kmh with formula: (speed / 100) ** 1.3 * 2 - 1
SPEED_MIN_REWARD = -1
SPEED_MAX_REWARD = 1
PREVIEW_CAMERA_RES = [[640, 400, -5, 0, 2.5], [1280, 800, -5, 0, 2.5]]  # Available resolutions from "above the car" preview camera [width, height, x, y, z], where x, y and z are related to car position
COLLISION_FILTER = [['static.sidewalk', -1], ['static.road', -1], ['vehicle.', 500]]  # list of pairs: agent id (can be part of the name) and impulse value allowed (-1 - disable collision detection entirely)
