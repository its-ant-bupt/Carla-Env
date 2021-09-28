# 23319/usr/bin/env python
# Edited by Yuan Zheng
import math
import numpy as np


INTER_MAX_POINTS_PLOT = 10  # number of points used for displaying lookahead path
INTER_LOOKAHEAD_DISTANCE = 20  # lookahead in meters
INTER_DISTANCE_RES = 0.01  # distance between interpolated points
INTER_DISTANCE_STATE = 0.5


def liner_interpolation(waypoints):
    '''
    Preprocessing the waypoints
    :param waypoints: 预先获得的x，y组成的路径点
    :return:
    :param wp_distance: 预设路径点中每个点间的间距
    :param wp_interp: 插值后得到的新的路径点
    :param wp_interp_hash: 预设路径点在对应的插值点list中的位置
    :param wp_interp_state_hash: The position of waypoints in larger interval interpolation points
    '''
    wp_distance = []  # distance array
    for i in range(1, len(waypoints)):
        wp_distance.append(
            np.sqrt((waypoints[i][0] - waypoints[i-1][0])**2 +
                    (waypoints[i][1] - waypoints[i-1][1])**2)
        )
    wp_distance.append(0)

    wp_interp = []
    # 输入状态集的数据 间隔是0.2米
    wp_interp_state = []
    wp_interp_hash = []  # 用这个哈希表原路径上的点在插值后路径点的位置
    wp_interp_state_hash = []

    interp_counter = 0
    interp_state_counter = 0
    for i in range(len(waypoints) - 1):
        wp_interp.append(waypoints[i])
        wp_interp_state.append(waypoints[i])
        wp_interp_hash.append(interp_counter)
        wp_interp_state_hash.append(interp_state_counter)
        interp_counter += 1
        interp_state_counter += 1
        num_pts_to_interp = int(np.floor(wp_distance[i] / float(INTER_DISTANCE_RES)) - 1)
        num_pts_to_interp_state = int(np.floor(wp_distance[i] / float(INTER_DISTANCE_STATE)) - 1)

        wp_vector = [waypoints[i+1][0] - waypoints[i][0], waypoints[i+1][1] - waypoints[i][1]]
        wp_uvector = wp_vector / (np.linalg.norm(wp_vector) + 0.000001)
        for j in range(num_pts_to_interp):
            next_wp_vector = INTER_DISTANCE_RES * float(j + 1) * wp_uvector
            wp_interp.append([waypoints[i][0] + next_wp_vector[0], waypoints[i][1] + next_wp_vector[1]])
            interp_counter += 1
        for j in range(num_pts_to_interp_state):
            next_wp_vector_state = INTER_DISTANCE_STATE * float(j + 1) * wp_uvector
            wp_interp_state.append([waypoints[i][0] + next_wp_vector_state[0],
                                    waypoints[i][1] + next_wp_vector_state[1]])
            interp_state_counter += 1
    wp_interp.append([waypoints[-1][0], waypoints[-1][1]])
    wp_interp_state.append([waypoints[-1][0], waypoints[-1][1]])
    wp_interp_hash.append(interp_counter)
    wp_interp_state_hash.append(interp_state_counter)
    interp_counter += 1
    interp_state_counter += 1
    return wp_distance, wp_interp, wp_interp_hash, wp_interp_state, wp_interp_state_hash


def get_new_waypoints(waypoints, wp_distance, wp_interp, wp_interp_hash,
                      curr_x, curr_y, LOOKAHEAD_DISTANCE):
    '''
    此函数是为了生成所需要观察的路径点，从当前车的位置，一直到前多少米所包含的所有路径点
    :param waypoints: 预先获得的x，y组成的路径点
    :param wp_distance: 预设路径点中每个点间的间距
    :param wp_interp: 插值后得到的新的路径点
    :param wp_interp_hash: 预设路径点在对应的插值点list中的位置
    :param curr_x: 当前坐标x
    :param curr_y: 当前坐标y
    :param LOOKAHEAD_DISTANCE: 前瞻的距离
    :return:
    '''
    closest_index, closest_distance = find_Closest_Index(waypoints, curr_x, curr_y)
    waypoint_subset_first_index = closest_index - 1
    if waypoint_subset_first_index < 0:
        waypoint_subset_first_index = 0

    waypoint_subset_last_index = closest_index
    total_distance_ahead = 0
    while total_distance_ahead < LOOKAHEAD_DISTANCE:
        total_distance_ahead += wp_distance[waypoint_subset_last_index]
        waypoint_subset_last_index += 1
        if waypoint_subset_last_index >= len(waypoints):
            waypoint_subset_last_index = len(waypoints) - 1
            break
    new_waypoints = wp_interp[wp_interp_hash[waypoint_subset_first_index]:
                              wp_interp_hash[waypoint_subset_last_index] + 1]
    return new_waypoints, closest_index, closest_distance


def get_new_waypoints_state(waypoints, wp_distance, wp_interp, wp_interp_state, wp_interp_hash, wp_interp_state_hash,
                            curr_x, curr_y, LOOKAHEAD_DISTANCE):
    '''
    Get the small spacing route list for inputting controller
    :param waypoints: 预先获得的x，y组成的路径点
    :param wp_distance: 预设路径点中每个点间的间距
    :param wp_interp: 插值后得到的新的路径点
    :param wp_interp_state: The new route list interpolation by larger interval
    :param wp_interp_hash: 预设路径点在对应的插值点list中的位置
    :param wp_interp_state_hash: The position of waypoints in larger interval interpolation points
    :param curr_x: 当前坐标x
    :param curr_y: 当前坐标y
    :param LOOKAHEAD_DISTANCE: 前瞻的距离
    :return:
    :param new_waypoints: The route list used by controller
    :param new_waypoints_state: The route list used by adding into the state list for RL training
    '''
    closest_index, closest_distance = find_Closest_Index(waypoints, curr_x, curr_y)
    waypoint_subset_first_index = closest_index - 1
    if waypoint_subset_first_index < 0:
        waypoint_subset_first_index = 0

    waypoint_subset_last_index = closest_index
    total_distance_ahead = 0
    while total_distance_ahead < LOOKAHEAD_DISTANCE:
        total_distance_ahead += wp_distance[waypoint_subset_last_index]
        waypoint_subset_last_index += 1
        if waypoint_subset_last_index >= len(waypoints):
            waypoint_subset_last_index = len(waypoints) - 1
            break
    new_waypoints = wp_interp[wp_interp_hash[waypoint_subset_first_index]:
                              wp_interp_hash[waypoint_subset_last_index] + 1]
    new_waypoints_state = wp_interp_state[wp_interp_state_hash[waypoint_subset_first_index]:
                                          wp_interp_state_hash[waypoint_subset_last_index] + 1]
    return new_waypoints, closest_index, closest_distance, new_waypoints_state


# ******************************************
# 找到给定（x，y）在所给路径点中距离最近的位置和其距离
def find_Closest_Index(waypoints, curr_x, curr_y):
    index = 0
    dis = []
    for i in range(len(waypoints)):
        dis_temp = math.sqrt((math.pow((waypoints[i][0] - curr_x), 2)) + (math.pow((waypoints[i][1] - curr_y), 2)))
        dis.append(dis_temp)
    index = dis.index(min(dis))
    closest_distance = math.sqrt((math.pow((waypoints[index][0] - curr_x), 2)) +
                                 (math.pow((waypoints[index][1] - curr_y), 2)))
    return index, closest_distance
