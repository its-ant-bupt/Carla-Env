import carla
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import math
import networkx as nx

import os
import pygame
from PIL import Image

import carla_env_settings as settings
import Controller2D
import cutils
import event_handle
import path_optimizer
import SyncMode
import utils
import utils_for_waypoints
import argparse
from matplotlib import cm



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

control = carla.VehicleControl()

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def send_control_command(_actor, _throttle, _steer, _brake):
    control.throttle = _throttle
    control.steer = _steer
    control.brake = _brake
    control.hand_brake = False
    control.reverse = False
    control.manual_gear_shift = False
    # control.gear = 0
    _actor.apply_control(control)


def get_position(frame_id, vehicle_list, snapshot):
    assert (frame_id == snapshot.frame)
    transforms = []
    _yaws = []
    _velocitys = []
    for vehicle in vehicle_list:
        transforms.append(snapshot.find(vehicle.id).get_transform())
        _rotation = snapshot.find(vehicle.id).get_transform().rotation
        _yaws.append(_rotation.yaw)  # [degrees]
        _velocitys.append(math.sqrt(
            math.pow(snapshot.find(vehicle.id).get_velocity().x, 2) + math.pow(
                snapshot.find(vehicle.id).get_velocity().y,
                2)))  # [m / s]
    return transforms, _yaws, _velocitys, snapshot.timestamp.elapsed_seconds


def get_transform(vehicle_location):
    # location = carla.Location(-5.5, 0, 2.8) + vehicle_location
    # return carla.Transform(location, carla.Rotation(pitch=-15))
    location = carla.Location(0, 0, 50) + vehicle_location  # 正上方视角
    return carla.Transform(location, carla.Rotation(pitch=-90))


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def draw_image_array(surface, image_array, blend=False):
    image_surface = pygame.surfarray.make_surface(image_array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def env_args():
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor sync and projection tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='680x420',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=500,
        type=int,
        help='number of frames to record (default: 500)')
    argparser.add_argument(
        '-d', '--dot_extent',
        metavar='SIZE',
        default=2,
        type=int,
        help='visualization dot extent in pixels (Recomended [1-4]) (default: 2)')
    argparser.add_argument(
        '--no_noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--upper_fov',
        metavar='F',
        default=30.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower_fov',
        metavar='F',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '-c', '--channels',
        metavar='C',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '-r', '--range',
        metavar='R',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points_per_second',
        metavar='N',
        default='400000',
        type=int,
        help='lidar points per second (default: 100000)')
    argparser.add_argument(
        '--show_image_type',
        default = 1,
        type = int,
        help='Mean 1:Only show the camera image 2:Only show the lidar point 3:Show both camera image and lidar')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.dot_extent -= 1
    return args


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def get_next_point(current_location, current_yaw, degree_delta, radius):
    '''
    Take a point from a circle centered on the vehicle with radius and degree delta
    :param current_location: The current location of vehicle
    :param current_yaw: The current yaw of vehicle driving [0,360]
    :param degree_delta: The delta degree from the current yaw [-180,180]
    :param radius: The radius for taking point [m]
    :return: next point
    '''
    _yaw = math.radians(current_yaw + degree_delta)
    next_x = current_location.x + radius * np.cos(_yaw)
    next_y = current_location.y + radius * np.sin(_yaw)
    return next_x, next_y


class CarlaEnv:
    # 方向盘转动率
    STEER_AMT = 1.0

    # 图像尺寸
    im_width = settings.IMG_WIDTH
    im_height = settings.IMG_HEIGHT

    def __init__(self, args, run_type="train", upload_map_layer=False):
        if run_type == "test":
            random.seed(13)
        self.args = args

        self.clock = pygame.time.Clock()
        # set client attribute
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        # set world attribute
        self.world = self.client.load_world('Town03')
        if upload_map_layer:
            self.world.unload_map_layer(carla.MapLayer.All)
        # get map information
        self.map = self.world.get_map()
        # select the point in map with the interval=50m
        self.map_waypoints = self.map.generate_waypoints(50)
        self.spawn_points = self.map.get_spawn_points()
        # the Digraph attribute
        self.G = nx.DiGraph()
        self.waypoints_info = {}
        self.get_topology()
        # set the world debug helper
        self.debug = self.world.debug
        # set the spectator
        self.spectator = self.world.get_spectator()
        # get the traffic manager
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        # get the blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        # get the vehicle model and set the attribute
        self.model_3 = self.blueprint_library.filter('model3')[0]
        self.model_3.set_attribute('color', '200, 0, 0')
        self.police = self.blueprint_library.filter('vehicle.chargercop2020.*')[0]

        # set the vehicles information , different environments have different settings
        self.main_vehicles = []
        self.main_vehicles_obstacle_detection = []
        self.equipment_actors = []
        self.background_vehicles = []
        self.front_camera = None
        self.preview_camera = None

        # set the attribute of some blueprint
        # 1.camera
        self.camera_rgb_bp = self.blueprint_library.filter("sensor.camera.rgb")[0]
        self.camera_rgb_bp.set_attribute("image_size_x", str(self.args.width))
        self.camera_rgb_bp.set_attribute("image_size_y", str(self.args.height))
        # 2.lidar
        self.lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        if self.args.no_noise:
            self.lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            self.lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            self.lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        self.lidar_bp.set_attribute('upper_fov', str(self.args.upper_fov))
        self.lidar_bp.set_attribute('lower_fov', str(self.args.lower_fov))
        self.lidar_bp.set_attribute('channels', str(self.args.channels))
        self.lidar_bp.set_attribute('range', str(self.args.range))
        self.lidar_bp.set_attribute('points_per_second', str(self.args.points_per_second))
        self.lidar_bp.set_attribute('rotation_frequency', '30')
        self.lidar_bp.set_attribute('points_per_second', '112000')
        # 3.obstacle sensor
        self.sensor_obstacle_detector_bp = self.world.get_blueprint_library().find('sensor.other.obstacle')
        self.sensor_obstacle_detector_bp.set_attribute('distance', '200')
        self.sensor_obstacle_detector_bp.set_attribute('hit_radius', '12')
        self.sensor_obstacle_detector_bp.set_attribute('only_dynamics', 'True')
        self.sensor_obstacle_detector_bp.set_attribute('debug_linetrace', 'True')
        self.sensor_obstacle_detector_bp.set_attribute('sensor_tick', '0.5')

        # obstacle information
        self.obstacle_actors = []

        # the control information
        self.main_vehicles_control_route_info = []
        self.all_stop_flag = False

        # confirm running
        self.last_cam_update = time.time()

        # 计算reward用到的参数，用于记录前一步中的位置，index表示
        self.last_index = 0

        self.Start = False

    def reset(self, main_vehicle_nums, background_vehicle_nums):
        # set the vehicles information , different environments have different settings
        self.Start = False
        self.main_vehicles = []
        self.main_vehicles_obstacle_detection = []
        self.equipment_actors = [[] for _ in range(main_vehicle_nums)]
        self.background_vehicles = []
        self.main_vehicles_strat_points = []
        spawn_start = time.time()
        random.shuffle(self.spawn_points)
        try:
            if main_vehicle_nums + background_vehicle_nums > len(self.spawn_points):
                background_vehicle_nums = len(self.spawn_points) - main_vehicle_nums
            for i in range(main_vehicle_nums):
                main_vehicle = self.world.spawn_actor(self.model_3, self.spawn_points[i])
                self.main_vehicles_strat_points.append(self.spawn_points[i])
                self.main_vehicles.append(main_vehicle)

                camera_rgb = self.world.spawn_actor(
                    self.camera_rgb_bp,
                    carla.Transform(carla.Location(x = -5.5, z = 2.8), carla.Rotation(pitch = -15)),
                    attach_to = main_vehicle
                )
                self.equipment_actors[i].append(camera_rgb)
                # camera_semseg = self.world.spawn_actor(
                #     self.blueprint_library.find('sensor.camera.semantic_segmentation'),
                #     carla.Transform(carla.Location(x = -5.5, z = 2.8), carla.Rotation(pitch = -15)),
                #     attach_to = main_vehicle
                # )
                # self.equipment_actors[i].append(camera_semseg)

                # add lidar sensor
                lidar_sensor = self.world.spawn_actor(
                    self.lidar_bp,
                    carla.Transform(carla.Location(x = 1.0, z = 1.8)),
                    attach_to = main_vehicle
                )
                self.equipment_actors[i].append(lidar_sensor)
                # add obstacle sensor
                sensor_obstacle_detector = self.world.spawn_actor(
                    self.sensor_obstacle_detector_bp,
                    carla.Transform(carla.Location(x = 1.6, z = 1.7), carla.Rotation(yaw = 0)),
                    attach_to = main_vehicle
                )
                sensor_obstacle_detector.listen(self._obstacle_detector)
                self.main_vehicles_obstacle_detection.append(sensor_obstacle_detector)
        except:
            print("Error")
            time.sleep(0.01)
        # if time.time() > spawn_start + 10:
        #     raise  Exception('Can\'t spawn a car')
        # add background vehicle
        # number_of_spawn_points = len(self.spawn_points) - main_vehicle_nums
        #
        # if background_vehicle_nums > number_of_spawn_points:
        #     background_vehicle_nums = number_of_spawn_points
        vehicle_bps = self.blueprint_library.filter('vehicle.*.*')
        vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels'))==4]

        for i in range(background_vehicle_nums):
            point = self.spawn_points[i+main_vehicle_nums]
            temp_vehicle_bp = np.random.choice(vehicle_bps)
            try:
                bg_vehicle = self.world.spawn_actor(temp_vehicle_bp, point)
                self.background_vehicles.append(bg_vehicle)
                print("spawn the actor : %s wait 0.1s..." % bg_vehicle.id)
                time.sleep(0.1)
            except:
                print('spawn failed')
                continue

        traffic_manager_port = self.traffic_manager.get_port()
        for v in self.background_vehicles:
            try:
                v.set_autopilot(True, traffic_manager_port)
                self.traffic_manager.ignore_lights_percentage(v, 100)
                # traffic_manager.distance_to_leading_vehicle(v, 0)
                self.traffic_manager.vehicle_percentage_speed_difference(v, 5)
                print("Set the manager : %s wait 0.1s..." % v.id)
                time.sleep(0.1)
            except:
                print('manager failed')
                continue

        for i in range(main_vehicle_nums):
            temp_start_waypoint = self.map.get_waypoint(self.spawn_points[i].location)
            init_list, waypoint_list = utils.get_next_dis_no_turn(temp_start_waypoint, self.map, dis=50, interval=0.25)
            wp_distance, wp_interp, wp_interp_hash, wp_interp_state, wp_interp_state_hash = \
                utils_for_waypoints.liner_interpolation(init_list)
            speed_list = utils.init_desired_speed(init_list, 60, 0)
            temp_controller = Controller2D.Controller2D(init_list, 'MPC')
            self.main_vehicles_control_route_info.append({
                "controller": temp_controller,
                "next_list": init_list,
                "wp_distance": wp_distance,
                "wp_interp": wp_interp,
                "wp_interp_hash": wp_interp_hash,
                "wp_interp_state": wp_interp_state,
                "wp_interp_state_hash": wp_interp_state_hash,
                "speed_list": speed_list,
                "Need_update": False,
                "route_length": len(init_list),
                "id": self.main_vehicles[i].id
            })

        # quick start
        for i in range(main_vehicle_nums):
            send_control_command(self.main_vehicles[i], 1.0, 0, 1.0)
        time.sleep(4.0)

        self.Start = True
        return time.time() - spawn_start

    def step_control(self, sync_frame, snapshot, frame):
        vehicles_transform, vehicles_yaw, vehicles_speed, curr_timestamp = \
            get_position(sync_frame, self.main_vehicles, snapshot)
        if curr_timestamp <= 1:
            self.all_stop_flag = True
        else:
            curr_timestamp = curr_timestamp - 1
            self.all_stop_flag = False
        for i in range(len(self.main_vehicles)):
            x = vehicles_transform[i].location.x
            y = vehicles_transform[i].location.y
            # todo test radius next point
            # next_x, next_y = get_next_point(vehicles_transform[i].location, vehicles_yaw[i], -5, 50)
            # distance = utils.distance_to_point(x, y, next_x, next_y)
            # next_points = utils.inter_two_point([x, y, 0], [next_x, next_y, 0], distance, 0.5)
            # for point in next_points:
            #     self.debug.draw_point(
            #         location=carla.Location(x=point[0], y=point[1], z=1),
            #         color=carla.Color(0, 255, 255),
            #         life_time=0.1
            #     )

            # 更新车辆的位置状态

            yaw = math.radians(vehicles_yaw[i])
            current_x, current_y = self.main_vehicles_control_route_info[i]["controller"].get_shifted_coordinate(x, y, yaw
                                                                                                         , length = 1.5)
            new_waypoints, closest_index, closest_distance, new_waypoints_state = \
                utils_for_waypoints.get_new_waypoints_state(
                self.main_vehicles_control_route_info[i]["next_list"],
                self.main_vehicles_control_route_info[i]["wp_distance"],
                self.main_vehicles_control_route_info[i]["wp_interp"],
                self.main_vehicles_control_route_info[i]["wp_interp_state"],
                self.main_vehicles_control_route_info[i]["wp_interp_hash"],
                self.main_vehicles_control_route_info[i]["wp_interp_state_hash"],
                current_x,
                current_y,
                LOOKAHEAD_DISTANCE = 10)
            for _way in range(len(new_waypoints_state)):
                self.debug.draw_point(
                    location=carla.Location(x=new_waypoints_state[_way][0], y=new_waypoints_state[_way][1],
                                            z=0),
                    color=carla.Color(255, 0, 0), life_time=0.1)
            desired_speed = utils.get_desired_speed(self.main_vehicles_control_route_info[i]["speed_list"], x, y)
            self.main_vehicles_control_route_info[i]["controller"].update_waypoints(new_waypoints)
            self.main_vehicles_control_route_info[i]["controller"].update_values(current_x, current_y, yaw, vehicles_speed[i],
                                                                         curr_timestamp, frame, closest_distance,
                                                                         desired_speed)
            self.main_vehicles_control_route_info[i]["controller"].update_controls()
            cmd_throttle, cmd_steer, cmd_brake = self.main_vehicles_control_route_info[i]["controller"].get_commands()
            if self.all_stop_flag:
                cmd_throttle = 0
                cmd_brake = 1
                cmd_steer = 0
            else:
                if frame < 100:
                    cmd_throttle = 0
                elif frame < 200:
                    cmd_throttle = 1
            assert (self.main_vehicles[i].id == self.main_vehicles_control_route_info[i]["id"])
            send_control_command(self.main_vehicles[i], cmd_throttle, cmd_steer, cmd_brake)
            print("vehicle :{} The throttle is :{}, steer is :{}, brake is :{}".format(self.main_vehicles[i].id, cmd_throttle,
                                                                                       cmd_steer, cmd_brake))
            if closest_index > self.main_vehicles_control_route_info[i]["route_length"] - 40:
                self.main_vehicles_control_route_info[i]["Need_update"] = True
        return vehicles_transform, vehicles_speed

    def step_update(self, vehicles_transform, vehicles_speed):
        update = False
        for i in range(len(self.main_vehicles)):
            if self.main_vehicles_control_route_info[i]["Need_update"]:
                update = True
        if update:
            for i in range(len(self.main_vehicles)):
                if self.main_vehicles_control_route_info[i]["Need_update"]:
                    temp_now_waypoint = self.map.get_waypoint(vehicles_transform[i].location)
                    # go along straight
                    next_list, waypoint_list = utils.get_next_dis_no_turn(temp_now_waypoint,
                                                                 self.map,
                                                                 dis=50,
                                                                 interval=0.25)
                    self.main_vehicles_control_route_info[i]["next_list"] = next_list
                    self.main_vehicles_control_route_info[i]["route_length"] = len(next_list)
                    wp_distance, wp_interp, wp_interp_hash, wp_interp_state, wp_interp_state_hash = \
                        utils_for_waypoints.liner_interpolation(next_list)
                    self.main_vehicles_control_route_info[i]["wp_distance"] = wp_distance
                    self.main_vehicles_control_route_info[i]["wp_interp"] = wp_interp
                    self.main_vehicles_control_route_info[i]["wp_interp_hash"] = wp_interp_hash
                    self.main_vehicles_control_route_info[i]["wp_interp_state"] = wp_interp_state
                    self.main_vehicles_control_route_info[i]["wp_interp_state_hash"] = wp_interp_state_hash
                    speed_list = utils.init_desired_speed(next_list, 60, vehicles_speed[i])
                    self.main_vehicles_control_route_info[i]["speed_list"] = speed_list
                    self.main_vehicles_control_route_info[i]["Need_update"] = False
        return True

    def destroy_agents(self):
        for equips in self.equipment_actors:
            print("Destroy equip")
            for actor in equips:
                # 如果actor存在回调函数，先去除
                if hasattr(actor, 'is_listening') and actor.is_listening:
                    actor.stop()
                if actor.is_alive:
                    print("Destory {}".format(actor.id))
                    actor.destroy()
        for actor in self.main_vehicles_obstacle_detection:
            print("Destroy obstacle")
            # 如果actor存在回调函数，先去除
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            if actor.is_alive:
                print("Destory {}".format(actor.id))
                actor.destroy()
        for actor in self.main_vehicles:
            print("Destroy main vehicle")
            # 如果actor存在回调函数，先去除
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            if actor.is_alive:
                print("Destory {}".format(actor.id))
                actor.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.background_vehicles])
        # for actor in self.background_vehicles:
        #     print("Destroy background")
        #     # 如果actor存在回调函数，先去除
        #     if hasattr(actor, 'is_listening') and actor.is_listening:
        #         actor.stop()
        #     if actor.is_alive:
        #         print("Destory {}".format(actor.id))
        #         actor.destroy()

    def _obstacle_detector(self, obstacle_event):
        if self.Start and 'vehicle.' in obstacle_event.other_actor.type_id:
            if obstacle_event.other_actor.id in self.obstacle_actors:
                print("Haven found the vehicle,the obstacle actors length is %s" % len(self.obstacle_actors))
            else:
                self.obstacle_actors.append(obstacle_event.other_actor.id)
                print("The vehicle %s at the distance %s" % (obstacle_event.other_actor.id, obstacle_event.distance))

    def get_topology(self):
        '''
        :return: 当前地图固定 返回当前地图下每一条路径的之间的拓扑结构
        '''
        topology = self.map.get_topology()
        # 新建一个空地字典用于查找waypoint
        self.waypoints_info = {}
        for i in range(len(topology)):
            waypoint1 = topology[i][0]
            waypoint2 = topology[i][1]
            waypoint1_info = "%s-%s-%s" % (waypoint1.road_id, waypoint1.section_id, waypoint1.lane_id)
            waypoint2_info = "%s-%s-%s" % (waypoint2.road_id, waypoint2.section_id, waypoint2.lane_id)
            if waypoint1_info not in self.waypoints_info:
                self.waypoints_info[waypoint1_info] = waypoint1
            if waypoint2_info not in self.waypoints_info:
                self.waypoints_info[waypoint2_info] = waypoint2
            distance = math.sqrt(math.pow((waypoint2.transform.location.x - waypoint1.transform.location.x), 2) +
                                 math.pow((waypoint2.transform.location.y - waypoint1.transform.location.y), 2))
            # 带有权重的边
            self.G.add_weighted_edges_from([(waypoint1_info, waypoint2_info, distance)])

    def lidar_data(self, lidar_image, lidar_measurement, lidar_sensor, lidar_camera):
        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        image_w = self.camera_rgb_bp.get_attribute("image_size_x").as_int()
        image_h = self.camera_rgb_bp.get_attribute("image_size_y").as_int()
        fov = self.camera_rgb_bp.get_attribute("fov").as_float()  # Horizontal filed of view in degrees
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0

        im_array = np.copy(np.frombuffer(lidar_image.raw_data, dtype = np.dtype("uint8")))
        im_array = np.reshape(im_array, (lidar_image.height, lidar_image.width, 4))
        im_array = im_array[:, :, :3][:, :, ::-1]
        if self.args.show_image_type == 2:
            im_array = np.zeros(im_array.shape)
        # Get the lidar data and convert it to a numpy array.
        p_cloud_size = len(lidar_measurement)
        p_cloud = np.copy(np.frombuffer(lidar_measurement.raw_data, dtype = np.dtype('f4')))
        p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))
        # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
        # focus on the 3D points.
        intensity = np.array(p_cloud[:, 3])

        # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
        local_lidar_points = np.array(p_cloud[:, :3]).T

        # Add an extra 1.0 at the end of each 3d point so it becomes of
        # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
        local_lidar_points = np.r_[
            local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
        # This (4, 4) matrix transforms the points from lidar space to world space.
        lidar_2_world = lidar_sensor.get_transform().get_matrix()

        # Transform the points from lidar space to world space.
        world_points = np.dot(lidar_2_world, local_lidar_points)
        # This (4, 4) matrix transforms the points from world to sensor coordinates.
        world_2_camera = np.array(lidar_camera.get_transform().get_inverse_matrix())

        # Transform the points from world space to camera space.
        sensor_points = np.dot(world_2_camera, world_points)
        # New we must change from UE4's coordinate system to an "standard"
        # camera coordinate system (the same used by OpenCV):

        # ^ z                       . z
        # |                        /
        # |              to:      +-------> x
        # | . x                   |
        # |/                      |
        # +-------> y             v y

        # This can be achieved by multiplying by the following matrix:
        # [[ 0,  1,  0 ],
        #  [ 0,  0, -1 ],
        #  [ 1,  0,  0 ]]

        # Or, in this case, is the same as swapping:
        # (x, y ,z) -> (y, -z, x)
        point_in_camera_coords = np.array([
            sensor_points[1],
            sensor_points[2] * -1,
            sensor_points[0]])

        # Finally we can use our K matrix to do the actual 3D -> 2D.
        points_2d = np.dot(K, point_in_camera_coords)

        # Remember to normalize the x, y values by the 3rd value.
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])

        # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
        # contains all the y values of our points. In order to properly
        # visualize everything on a screen, the points that are out of the screen
        # must be discarted, the same with points behind the camera projection plane.
        points_2d = points_2d.T
        intensity = intensity.T
        points_in_canvas_mask = \
            (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
            (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
            (points_2d[:, 2] > 0.0)
        points_2d = points_2d[points_in_canvas_mask]
        intensity = intensity[points_in_canvas_mask]

        # Extract the screen coords (uv) as integers.
        u_coord = points_2d[:, 0].astype(np.int)
        v_coord = points_2d[:, 1].astype(np.int)

        # Since at the time of the creation of this script, the intensity function
        # is returning high values, these are adjusted to be nicely visualized.
        intensity = 4 * intensity - 3
        color_map = np.array([
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T

        if self.args.dot_extent <= 0:
            # Draw the 2d points on the image as a single pixel using numpy.
            im_array[v_coord, u_coord] = color_map
        else:
            # Draw the 2d points on the image as squares of extent args.dot_extent.
            for i in range(len(points_2d)):
                # I'm not a NumPy expert and I don't know how to set bigger dots
                # without using this loop, so if anyone has a better solution,
                # make sure to update this script. Meanwhile, it's fast enough :)
                im_array[
                v_coord[i] - self.args.dot_extent: v_coord[i] + self.args.dot_extent,
                u_coord[i] - self.args.dot_extent: u_coord[i] + self.args.dot_extent] = color_map[i]
        return im_array

    def lidar_matrix(self, lidar_measurement):
        inter = 5
        width = int(self.args.range/inter)
        _lidar_matrix = [[0 for _ in range(2*width)] for _ in range(2*width)]
        for location in lidar_measurement:
            _x = location.point.x/inter
            _y = location.point.y/inter
            _z = round(location.point.z, 1)
            X = abs(int(_x))
            Y = abs(int(_y))
            if _x >= 0:
                if _y >= 0:
                    _lidar_matrix[width + X][width - 1 - Y] = max(_lidar_matrix[width + X][width - 1 - Y], _z)
                else:
                    _lidar_matrix[width + X][width + Y] = max(_lidar_matrix[width + X][width + Y], _z)
            else:
                if _y >= 0:
                    _lidar_matrix[width - 1 - X][width - 1 - Y] = max(_lidar_matrix[width - 1 - X][width - 1 - Y], _z)
                else:
                    _lidar_matrix[width - 1 - X][width + Y] = max(_lidar_matrix[width - 1 - X][width + Y], _z)
        _lidar_matrix = np.array(_lidar_matrix)
        # high_list = []
        # for i in range(20):
        #     high_list.append(0.1 * i)
        # for i in range(len(high_list)-1):
        #     _lidar_matrix[high_list[i + 1] > _lidar_matrix > high_list[i]] = high_list[i + 1]
        return _lidar_matrix

    def run(self, display):
        font = get_font()
        with SyncMode.CarlaSyncMode(self.world, self.equipment_actors[0][0], self.equipment_actors[0][1], fps=30) as sync_mode:
            frame = -1
            done = False
            while True:
                if frame == 0:
                    self.spectator.set_transform(get_transform(self.main_vehicles[0].get_location()))
                if should_quit():
                    return
                self.clock.tick()

                snapshot, image_rgb_look, lidar_measurement = sync_mode.tick(timeout = 2.0)
                lidar_matrix = self.lidar_matrix(lidar_measurement)
                # todo matplot test
                start = int(-1 * self.args.range/5)
                end = int(self.args.range/5)
                _x = np.arange(start, end, 1)
                _y = np.arange(start, end, 1)
                _x, _y = np.meshgrid(_x, _y)
                _z = lidar_matrix
                plt.figure()
                plt.contourf(_x, _y, _z)
                plt.contour(_x, _y, _z)
                plt.savefig('./matplot/%s.jpg' % frame)
                # plt.show()
                vehicles_transform, vehicles_speed = self.step_control(sync_mode.frame, snapshot, frame = frame)
                frame += 1
                result = self.step_update(vehicles_transform, vehicles_speed)
                if not result:
                    return
                if self.args.show_image_type == 1:
                    draw_image(display, image_rgb_look)
                else:
                    image_array = self.lidar_data(image_rgb_look, lidar_measurement,
                                                  self.equipment_actors[0][1], self.equipment_actors[0][0])
                    image_array_save = Image.fromarray(np.uint8(image_array))
                    image_array_save.save("out/%08d.png" % frame)
                    draw_image_array(display, image_array)
                display.blit(
                    font.render('%s frame' % frame, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()


def main():
    args = env_args()
    pygame.init()
    pygame_width = args.width  # default 680 800
    pygame_height = args.height  # default 420 600
    display = pygame.display.set_mode(
        (pygame_width, pygame_height),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    try:
        env = CarlaEnv(args = args)
        build_time = env.reset(main_vehicle_nums = 1, background_vehicle_nums = 50)
        print("The reset time is %s" % build_time)
        env.run(display)
    finally:
        print('destroying actors.')
        env.destroy_agents()
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    main()






