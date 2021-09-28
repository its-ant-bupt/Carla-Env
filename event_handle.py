import math
import numpy as np

import carla
import path_optimizer
import bezier
import matplotlib.pyplot as plt
from utils import inter_two_point, findClosestPoint, distance_to_point, get_bezier_curve, judge_vehicle
import utils


def Judge_junction(start_waypoint, interval=0.25, dis=15):
    junction_dis = -1
    add_dis = 0
    temp_next = start_waypoint
    for i in range(int(dis / interval)):
        temp_next = temp_next.next(interval)[0]
        add_dis += interval
        if temp_next.is_junction:
            junction_dis = add_dis
            return junction_dis, temp_next
    return junction_dis, temp_next


def Judge_junction_left_turn(all_waypoints_pairs, now_waypoints):
    for i in range(len(all_waypoints_pairs)):
        dis2First = distance_to_point(all_waypoints_pairs[i][0].transform.location.x
                                      , all_waypoints_pairs[i][0].transform.location.y
                                      , now_waypoints.transform.location.x
                                      , now_waypoints.transform.location.y)
        if dis2First < 0.3:
            # 判断是否是左转
            if 80 < all_waypoints_pairs[i][0].transform.rotation.yaw - \
                    all_waypoints_pairs[i][1].transform.rotation.yaw < 100:
                return True
    return False


def Judge_junction_right_turn(all_waypoints_pairs, now_waypoints):
    for i in range(len(all_waypoints_pairs)):
        dis2First = distance_to_point(all_waypoints_pairs[i][0].transform.location.x
                                      , all_waypoints_pairs[i][0].transform.location.y
                                      , now_waypoints.transform.location.x
                                      , now_waypoints.transform.location.y)
        if dis2First < 0.3:
            # 判断是否是左转
            if -100 < all_waypoints_pairs[i][0].transform.rotation.yaw - \
                    all_waypoints_pairs[i][1].transform.rotation.yaw < -80:
                return True
    return False


def get_next_dis_no_turn(start_waypoint, _map, dis=50, interval=0.25):
    '''
    返回从当前位置开始前方dis距离内的路径点，默认以0.25米为间隔，且路径点为直行，即通过交叉口时为直行通过，若存在交叉口实际距离会比dis大
    :param start_waypoint: 起始点 carla.waypoint
    :param _map: 地图信息
    :param dis: 前方的距离 单位：m
    :param interval: 两个路径点之间的间隔 单位:m
    :return: 路径点坐标[[x1,y1], [x2,y2],...] (x,y 单位：m) Waypoint列表 [waypoint1, waypoint2,...] (waypoint:carla.waypoint)
    '''
    routeList = []
    waypointList = []
    routeList.append([start_waypoint.transform.location.x, start_waypoint.transform.location.y])
    waypointList.append(start_waypoint)
    now_waypoint = start_waypoint
    for i in range(int(dis / interval)):
        next_waypoint = now_waypoint.next(interval)
        if next_waypoint[0].is_junction:
            temp_junction = next_waypoint[0].get_junction()
            all_waypoint = temp_junction.get_waypoints(carla.LaneType.Driving)
            print("Length junction is %s" % len(all_waypoint))
            # if len(all_waypoint) > 10:
            junction_waypoints, junction_waypoints_list = get_straight_point(all_waypoint, now_waypoint, _map)
            if junction_waypoints:
                waypointList += junction_waypoints
                routeList += junction_waypoints_list
                next_waypoint = waypointList[-1].next(interval)
        if len(next_waypoint) > 1:
            routeList.append([next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
            waypointList.append(next_waypoint[0])
            now_waypoint = next_waypoint[0]
        else:
            routeList.append([next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
            waypointList.append(next_waypoint[0])
            now_waypoint = next_waypoint[0]
    return routeList


def get_next_dis_left_intention(start_waypoint, _map, dis=50, interval=0.25):
    routeList = []
    change_interval = dis / 10
    if str(start_waypoint.lane_change) == "Both" or str(start_waypoint.lane_change) == "Left":
        junction_dis, junction_waypoint = Judge_junction(start_waypoint, 0.25, 15)
        if junction_dis == -1:  # 无交叉口-左变道
            final_path = utils.get_next_dis_left_turn(start_waypoint, _map, dis=10, interval=0.25, vehicle_ahead=True)
            routeList = final_path
            return routeList
        elif junction_dis <= 15:
            all_waypoint = junction_waypoint.get_junction().get_waypoints(carla.LaneType.Driving)
            if Judge_junction_left_turn(all_waypoint, junction_waypoint):  # 有交叉口可以左转
                routeList.append([start_waypoint.transform.location.x, start_waypoint.transform.location.y])
                now_waypoint = start_waypoint
                for i in range(int(dis / interval)):
                    next_waypoint = now_waypoint.next(interval)
                    if next_waypoint[0].is_junction:
                        temp_junction = next_waypoint[0].get_junction()
                        all_waypoint = temp_junction.get_waypoints(carla.LaneType.Driving)
                        print("Length junction is %s" % len(all_waypoint))
                        junction_waypoints, junction_waypoints_list = get_left_point(all_waypoint, now_waypoint, _map)
                        if junction_waypoints:
                            routeList += junction_waypoints_list
                            next_waypoint = junction_waypoints[-1].next(interval)
                        if len(next_waypoint) > 1:
                            routeList.append(
                                [next_waypoint[1].transform.location.x, next_waypoint[1].transform.location.y])
                            now_waypoint = next_waypoint[1]
                        else:
                            routeList.append(
                                [next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
                            now_waypoint = next_waypoint[0]
                return routeList
            else:
                if junction_dis >= 10:  # 有交叉口无法左转且超过10m-左变道
                    final_path = utils.get_next_dis_left_turn(start_waypoint, _map, dis=10, interval=0.25,
                                                              vehicle_ahead=False)
                    routeList = final_path
                    return routeList
                else:  # 有交叉口无法左转且无法左变道-直行
                    routeList = get_next_dis_no_turn(start_waypoint, _map, dis, interval)
                    return routeList

    else:  # 无法左变道，以左转弯为主
        routeList.append([start_waypoint.transform.location.x, start_waypoint.transform.location.y])
        now_waypoint = start_waypoint
        for i in range(int(dis / interval)):
            next_waypoint = now_waypoint.next(interval)
            if next_waypoint[0].is_junction:
                temp_junction = next_waypoint[0].get_junction()
                all_waypoint = temp_junction.get_waypoints(carla.LaneType.Driving)
                print("Length junction is %s" % len(all_waypoint))
                junction_waypoints, junction_waypoints_list = get_left_point(all_waypoint, now_waypoint, _map)
                if junction_waypoints:
                    routeList += junction_waypoints_list
                    next_waypoint = junction_waypoints[-1].next(interval)
            if len(next_waypoint) > 1:
                routeList.append([next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
                now_waypoint = next_waypoint[0]
            else:
                routeList.append([next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
                now_waypoint = next_waypoint[0]
        return routeList


def get_next_dis_right_intention(start_waypoint, _map, dis=50, interval=0.25):
    routeList = []
    change_interval = dis / 10
    if str(start_waypoint.lane_change) == "Both" or str(start_waypoint.lane_change) == "Right":
        junction_dis, junction_waypoint = Judge_junction(start_waypoint, 0.25, 15)
        if junction_dis == -1:  # 无交叉口-右变道
            final_path = utils.get_next_dis_right_turn(start_waypoint, _map, dis=10, interval=0.25, vehicle_ahead=True)
            routeList = final_path
            return routeList
        elif junction_dis <= 15:
            all_waypoint = junction_waypoint.get_junction().get_waypoints(carla.LaneType.Driving)
            if Judge_junction_right_turn(all_waypoint, junction_waypoint):  # 有交叉口可以右转
                routeList.append([start_waypoint.transform.location.x, start_waypoint.transform.location.y])
                now_waypoint = start_waypoint
                for i in range(int(dis / interval)):
                    next_waypoint = now_waypoint.next(interval)
                    if next_waypoint[0].is_junction:
                        temp_junction = next_waypoint[0].get_junction()
                        all_waypoint = temp_junction.get_waypoints(carla.LaneType.Driving)
                        print("Length junction is %s" % len(all_waypoint))
                        junction_waypoints, junction_waypoints_list = get_right_point(all_waypoint, now_waypoint, _map)
                        if junction_waypoints:
                            routeList += junction_waypoints_list
                            next_waypoint = junction_waypoints[-1].next(interval)
                        if len(next_waypoint) > 1:
                            routeList.append(
                                [next_waypoint[1].transform.location.x, next_waypoint[1].transform.location.y])
                            now_waypoint = next_waypoint[1]
                        else:
                            routeList.append(
                                [next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
                            now_waypoint = next_waypoint[0]
                return routeList
            else:
                if junction_dis >= 10:  # 有交叉口无法右转且超过10m-右变道
                    final_path = utils.get_next_dis_right_turn(start_waypoint, _map, dis=10, interval=0.25,
                                                              vehicle_ahead=False)
                    routeList = final_path
                    return routeList
                else:  # 有交叉口无法右转且无法右变道-直行
                    routeList = get_next_dis_no_turn(start_waypoint, _map, dis, interval)
                    return routeList

    else:  # 无法左变道，以右转弯为主
        routeList.append([start_waypoint.transform.location.x, start_waypoint.transform.location.y])
        now_waypoint = start_waypoint
        for i in range(int(dis / interval)):
            next_waypoint = now_waypoint.next(interval)
            if next_waypoint[0].is_junction:
                temp_junction = next_waypoint[0].get_junction()
                all_waypoint = temp_junction.get_waypoints(carla.LaneType.Driving)
                print("Length junction is %s" % len(all_waypoint))
                junction_waypoints, junction_waypoints_list = get_right_point(all_waypoint, now_waypoint, _map)
                if junction_waypoints:
                    routeList += junction_waypoints_list
                    next_waypoint = junction_waypoints[-1].next(interval)
            if len(next_waypoint) > 1:
                routeList.append([next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
                now_waypoint = next_waypoint[0]
            else:
                routeList.append([next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
                now_waypoint = next_waypoint[0]
        return routeList


def get_left_point(all_waypoints_pairs, now_waypoints, _map, interval=0.25):
    final_waypoint = []
    final_waypoint_list = []
    for i in range(len(all_waypoints_pairs)):
        dis2First = distance_to_point(all_waypoints_pairs[i][0].transform.location.x
                                      , all_waypoints_pairs[i][0].transform.location.y
                                      , now_waypoints.transform.location.x
                                      , now_waypoints.transform.location.y)
        if dis2First < 0.3:
            # 判断是否是左转
            if 80 < all_waypoints_pairs[i][0].transform.rotation.yaw - all_waypoints_pairs[i][
                1].transform.rotation.yaw < 100:
                final_waypoint_list.append([all_waypoints_pairs[i][0].transform.location.x
                                               , all_waypoints_pairs[i][0].transform.location.y])
                final_waypoint.append(all_waypoints_pairs[i][0])
                temp_road_id = all_waypoints_pairs[i][0].road_id
                temp_lane_id = all_waypoints_pairs[i][0].lane_id
                start_s = all_waypoints_pairs[i][0].s
                s_total = abs(all_waypoints_pairs[i][0].s - all_waypoints_pairs[i][1].s)
                if all_waypoints_pairs[i][0].s - all_waypoints_pairs[i][1].s > 0:
                    symbol = -1
                else:
                    symbol = 1
                for j in range(int(s_total / interval) - 1):
                    temp_waypoint = _map.get_waypoint_xodr(temp_road_id, temp_lane_id,
                                                           start_s + (symbol * (j + 1) * interval))
                    if temp_waypoint is None:
                        continue
                    else:
                        final_waypoint_list.append(
                            [temp_waypoint.transform.location.x, temp_waypoint.transform.location.y])
                        final_waypoint.append(temp_waypoint)
                return final_waypoint, final_waypoint_list
    return False, False


def get_right_point(all_waypoints_pairs, now_waypoints, _map, interval=0.25):
    final_waypoint = []
    final_waypoint_list = []
    for i in range(len(all_waypoints_pairs)):
        dis2First = distance_to_point(all_waypoints_pairs[i][0].transform.location.x
                                      , all_waypoints_pairs[i][0].transform.location.y
                                      , now_waypoints.transform.location.x
                                      , now_waypoints.transform.location.y)
        if dis2First < 0.3:
            # 判断是否是左转
            if -100 < all_waypoints_pairs[i][0].transform.rotation.yaw - all_waypoints_pairs[i][
                1].transform.rotation.yaw < -80:
                final_waypoint_list.append([all_waypoints_pairs[i][0].transform.location.x
                                               , all_waypoints_pairs[i][0].transform.location.y])
                final_waypoint.append(all_waypoints_pairs[i][0])
                temp_road_id = all_waypoints_pairs[i][0].road_id
                temp_lane_id = all_waypoints_pairs[i][0].lane_id
                start_s = all_waypoints_pairs[i][0].s
                s_total = abs(all_waypoints_pairs[i][0].s - all_waypoints_pairs[i][1].s)
                if all_waypoints_pairs[i][0].s - all_waypoints_pairs[i][1].s > 0:
                    symbol = -1
                else:
                    symbol = 1
                for j in range(int(s_total / interval) - 1):
                    temp_waypoint = _map.get_waypoint_xodr(temp_road_id, temp_lane_id,
                                                           start_s + (symbol * (j + 1) * interval))
                    if temp_waypoint is None:
                        continue
                    else:
                        final_waypoint_list.append(
                            [temp_waypoint.transform.location.x, temp_waypoint.transform.location.y])
                        final_waypoint.append(temp_waypoint)
                return final_waypoint, final_waypoint_list
    return False, False


# 根据交叉口中的指向得到当前位置下直行的走向
def get_straight_point(all_waypoints_pairs, now_waypoints, _map, interval=0.25):
    '''
    当需要直行通过交叉口时，根据当前的位置，获得一个直行穿过交叉口的路径点
    :param all_waypoints_pairs: 交叉口中出入口的连接关系[[w1, w2],...] w1：交叉口入口处的carla.waypoint w2:交叉口出口处的carla.waypoint
    :param now_waypoints: 当前点的carla.waypoint
    :param _map:地图信息
    :param interval:两个路径点之间的间隔
    :return:路径点坐标[[x1,y1], [x2,y2],...] (x,y 单位：m) Waypoint列表 [waypoint1, waypoint2,...] (waypoint:carla.waypoint)
    '''
    final_waypoint = []
    final_waypoint_list = []
    for i in range(len(all_waypoints_pairs)):
        dis2First = distance_to_point(all_waypoints_pairs[i][0].transform.location.x
                                      , all_waypoints_pairs[i][0].transform.location.y
                                      , now_waypoints.transform.location.x
                                      , now_waypoints.transform.location.y)
        if dis2First < 0.3:
            # 判断是否是直行
            if abs(all_waypoints_pairs[i][0].transform.rotation.yaw - all_waypoints_pairs[i][
                1].transform.rotation.yaw) < 1:
                final_waypoint_list.append([all_waypoints_pairs[i][0].transform.location.x
                                               , all_waypoints_pairs[i][0].transform.location.y])
                final_waypoint.append(all_waypoints_pairs[i][0])
                temp_road_id = all_waypoints_pairs[i][0].road_id
                temp_lane_id = all_waypoints_pairs[i][0].lane_id
                start_s = all_waypoints_pairs[i][0].s
                s_total = abs(all_waypoints_pairs[i][0].s - all_waypoints_pairs[i][1].s)
                if all_waypoints_pairs[i][0].s - all_waypoints_pairs[i][1].s > 0:
                    symbol = -1
                else:
                    symbol = 1
                for j in range(int(s_total / interval) - 1):
                    temp_waypoint = _map.get_waypoint_xodr(temp_road_id, temp_lane_id,
                                                           start_s + (symbol * (j + 1) * interval))
                    if temp_waypoint is None:
                        continue
                    else:
                        final_waypoint_list.append(
                            [temp_waypoint.transform.location.x, temp_waypoint.transform.location.y])
                        final_waypoint.append(temp_waypoint)
                return final_waypoint, final_waypoint_list
    return False, False
