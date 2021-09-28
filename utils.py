# 23319/usr/bin/env python
# Edited by Yuan Zheng

import math
import numpy as np

import carla
import path_optimizer
import bezier
import matplotlib.pyplot as plt

def get_desired_speed(waypoints, x, y):
    min_idx = 0
    min_dist = float("inf")
    desired_speed = 0
    for i in range(len(waypoints)):
        dist = np.linalg.norm(np.array([
            waypoints[i][0] - x,
            waypoints[i][1] - y]))
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    if min_idx < len(waypoints) - 1:
        desired_speed = waypoints[min_idx][2]
        # print("The waypoint poi is : %s, %s" % (waypoints[min_idx][0], waypoints[min_idx][1]))
    else:
        desired_speed = waypoints[-1][2]
        # print("The waypoint poi is : %s, %s" % (waypoints[-1][0], waypoints[-1][1]))
    # print("The actual poision is : %s, %s" % (x, y))

    return desired_speed


# 根据路径点所在位置的曲率，设置期望速度
# Input
# waypoint : 预设规划好的路径点 Format:{[x[m], y(m)]}
# speed_limit : 直行时的限速 km/h
# curr_v : 车辆当前速度 m/s
# Output
# result_waypoint : 在预设规划好的路径点上所给的速度及其位置
# Format:{[x(m), y(m), v(m/s)],...}
def init_desired_speed(waypoint, speed_limit, curr_v):
    speed_limit_temp = speed_limit / 3.6  # 转为 m/s
    result_waypoint = []
    result_curvature = []
    max_a = 2
    # 初始化第一个点为其当前速度
    result_waypoint.append([waypoint[0][0], waypoint[0][1], curr_v+0.1])
    for i in range(len(waypoint) - 2):
        x = [waypoint[i][0], waypoint[i+1][0], waypoint[i+2][0]]
        y = [waypoint[i][1], waypoint[i+1][1], waypoint[i+2][1]]
        curvature_now = abs(curvature(x, y))
        result_curvature.append(curvature_now)
    # 给中间路径点加上速度(1 ~ n-1)
    for i in range(1, 1+len(waypoint) - 2):
        # 两个路径点间的距离
        dist_pro = np.linalg.norm([waypoint[i][0] - waypoint[i-1][0], waypoint[i][1] - waypoint[i-1][1]])
        # 直线行驶
        # if abs(result_curvature[i-1]) < 0.00001:
        if abs(result_curvature[i - 1]) < 0.02:
            if result_waypoint[i-1][2] > speed_limit_temp:
                desired_sp = calc_final_speed(result_waypoint[i-1][2], -1 * max_a, dist_pro)
            else:
                desired_sp = calc_final_speed(result_waypoint[i-1][2], max_a, dist_pro)
                if desired_sp > speed_limit_temp:
                    desired_sp = speed_limit_temp
            # print("the desired is:%s" % desired_sp)
            result_waypoint.append([waypoint[i][0], waypoint[i][1], desired_sp])
        # 小幅度转弯速度限制为4m/s，需要考虑周围曲度是否较大
        elif abs(result_curvature[i-1]) < 0.07:
            # 查看前面计算出曲度的点的个数
            length = min(len(result_waypoint) - 1, 2)
            temp_cur = []
            temp_cur.append(abs(result_curvature[i-1]))
            if length > 0:
                for index_temp in range(length):
                    if abs(result_curvature[i-1-index_temp-1]) > 0.00001:
                        temp_cur.append(abs(result_curvature[i-1-index_temp-1]))
            if i + 1 < len(result_curvature):
                if abs(result_curvature[i-1+1]) > 0.00001:
                    temp_cur.append(abs(result_curvature[i-1+1]))
            if i + 2 < len(result_curvature):
                if abs(result_curvature[i-1+2]) > 0.00001:
                    temp_cur.append(abs(result_curvature[i-1+2]))
            ave_curve = sum(temp_cur) / len(temp_cur)
            # 正常小幅度弯道
            if ave_curve < 0.05:
                desired_sp = min(15, speed_limit_temp)
                result_waypoint.append([waypoint[i][0], waypoint[i][1], desired_sp])
                # 反向更新速度，使得速度更加平滑
                if result_waypoint[i-1][2] > desired_sp:
                    for index_temp in range(i):
                        dist_pro = np.linalg.norm([waypoint[i-index_temp][0] - waypoint[i-1-index_temp][0],
                                                   waypoint[i-index_temp][1] - waypoint[i-1-index_temp][1]])
                        pro_speed = calc_final_speed(result_waypoint[i-index_temp][2], max_a, dist_pro)
                        if pro_speed >= result_waypoint[i-1-index_temp][2]:
                            break
                        else:
                            result_waypoint[i-1-index_temp][2] = pro_speed
            # 大幅度弯道
            else:
                # desired_sp = 2.8
                desired_sp = min(10, speed_limit_temp)
                result_waypoint.append([waypoint[i][0], waypoint[i][1], desired_sp])
                # 反向更新速度，使得速度更加平滑
                if result_waypoint[i-1][2] > desired_sp:
                    for index_temp in range(i):
                        dist_pro = np.linalg.norm([waypoint[i-index_temp][0] - waypoint[i-1-index_temp][0],
                                                   waypoint[i-index_temp][1] - waypoint[i-1-index_temp][1]])
                        pro_speed = calc_final_speed(result_waypoint[i-index_temp][2], max_a, dist_pro)
                        if pro_speed >= result_waypoint[i-1-index_temp][2]:
                            break
                        else:
                            result_waypoint[i-1-index_temp][2] = pro_speed
        # 大幅度弯道
        else:
            desired_sp = min(10, speed_limit_temp)
            result_waypoint.append([waypoint[i][0], waypoint[i][1], desired_sp])
            # 反向更新速度，使得速度更加平滑
            if result_waypoint[i - 1][2] > desired_sp:
                for index_temp in range(i):
                    dist_pro = np.linalg.norm([waypoint[i - index_temp][0] - waypoint[i - 1 - index_temp][0],
                                               waypoint[i - index_temp][1] - waypoint[i - 1 - index_temp][1]])
                    pro_speed = calc_final_speed(result_waypoint[i - index_temp][2], max_a, dist_pro)
                    if pro_speed >= result_waypoint[i - 1 - index_temp][2]:
                        break
                    else:
                        result_waypoint[i - 1 - index_temp][2] = pro_speed
    result_waypoint.append([waypoint[-1][0], waypoint[-1][1], result_waypoint[-1][2]])
    return result_waypoint


# 更新路径点信息，重新规划路径后使用，需要额外添加当前位置坐标
def update_desired_speed(new_waypoint, speed_limit, curr_v, curr_x, curr_y):
    speed_limit_temp = speed_limit / 3.6  # 转为 m/s
    index, _ = findClosestPoint(curr_x, curr_y, new_waypoint)
    new_curvature = []
    result_waypoint = []
    max_a = 2
    for i in range(len(new_waypoint) - 2):
        x = [new_waypoint[i][0], new_waypoint[i+1][0], new_waypoint[i+2][0]]
        y = [new_waypoint[i][1], new_waypoint[i+1][1], new_waypoint[i+2][1]]
        curvature_now = abs(curvature(x, y))
        new_curvature.append(curvature_now)
    if curr_v >= speed_limit_temp - 0.1:
        result_waypoint.append([new_waypoint[index][0], new_waypoint[index][1], min(curr_v, speed_limit_temp - 0.5)])
    else:
        result_waypoint.append([new_waypoint[index][0], new_waypoint[index][1], curr_v+0.1])
    for i in range(index+1, len(new_waypoint)-1):
        dist_pro = np.linalg.norm([new_waypoint[i][0] - new_waypoint[i-1][0], new_waypoint[i][1] - new_waypoint[i-1][1]])
        # 直线行驶
        # if abs(new_curvature[i-1]) < 0.00001:
        if abs(new_curvature[i - 1]) < 0.02:
            # print("i-1:",i-1)
            if result_waypoint[i-1-index][2] > speed_limit_temp:
                desired_sp = calc_final_speed(result_waypoint[i-1-index][2], -1 * max_a, dist_pro)
            else:
                desired_sp = calc_final_speed(result_waypoint[i-1-index][2], max_a, dist_pro)
                if desired_sp > speed_limit_temp:
                    desired_sp = speed_limit_temp
            # print("the desired is:%s" % desired_sp)
            result_waypoint.append([new_waypoint[i][0], new_waypoint[i][1], desired_sp])
            # 小幅度转弯速度限制为4m/s，需要考虑周围曲度是否较大
        elif abs(new_curvature[i - 1]) < 0.07:
            # 查看前面计算出曲度的点的个数
            length = min(len(result_waypoint) - 1, 2)
            temp_cur = []
            temp_cur.append(abs(new_curvature[i - 1]))
            if length > 0:
                for index_temp in range(length):
                    if abs(new_curvature[i - 1 - index_temp - 1]) > 0.00001:
                        temp_cur.append(abs(new_curvature[i - 1 - index_temp - 1]))
            if i + 1 < len(new_curvature):
                if abs(new_curvature[i - 1 + 1]) > 0.00001:
                    temp_cur.append(abs(new_curvature[i - 1 + 1]))
            if i + 2 < len(new_curvature):
                if abs(new_curvature[i - 1 + 2]) > 0.00001:
                    temp_cur.append(abs(new_curvature[i - 1 + 2]))
            ave_curve = sum(temp_cur) / len(temp_cur)
            # 正常小幅度弯道
            if ave_curve < 0.05:
                # desired_sp = 4
                desired_sp = min(8, speed_limit_temp)
                result_waypoint.append([new_waypoint[i][0], new_waypoint[i][1], desired_sp])
                # 反向更新速度，使得速度更加平滑
                if result_waypoint[i - 1 - index][2] > desired_sp:
                    for index_temp in range(i):
                        dist_pro = np.linalg.norm([new_waypoint[i - index_temp][0] - new_waypoint[i - 1 - index_temp][0],
                                                   new_waypoint[i - index_temp][1] - new_waypoint[i - 1 - index_temp][1]])
                        pro_speed = calc_final_speed(result_waypoint[i - index_temp - index][2], max_a, dist_pro)
                        if pro_speed >= result_waypoint[i - 1 - index_temp - index][2]:
                            break
                        else:
                            result_waypoint[i - 1 - index_temp - index][2] = pro_speed
                # 大幅度弯道
            else:
                desired_sp = min(5, speed_limit_temp)
                result_waypoint.append([new_waypoint[i][0], new_waypoint[i][1], desired_sp])
                # 反向更新速度，使得速度更加平滑
                if result_waypoint[i - 1 - index][2] > desired_sp:
                    for index_temp in range(i):
                        dist_pro = np.linalg.norm([new_waypoint[i - index_temp][0] - new_waypoint[i - 1 - index_temp][0],
                                                   new_waypoint[i - index_temp][1] - new_waypoint[i - 1 - index_temp][1]])
                        pro_speed = calc_final_speed(result_waypoint[i - index_temp - index][2], max_a, dist_pro)
                        if pro_speed >= result_waypoint[i - 1 - index_temp - index][2]:
                            break
                        else:
                            result_waypoint[i - 1 - index_temp - index][2] = pro_speed
        # 大幅度弯道
        else:
            desired_sp = min(5, speed_limit_temp)
            result_waypoint.append([new_waypoint[i][0], new_waypoint[i][1], desired_sp])
            # 反向更新速度，使得速度更加平滑
            if result_waypoint[i - 1 - index][2] > desired_sp:
                for index_temp in range(i):
                    dist_pro = np.linalg.norm([new_waypoint[i - index_temp][0] - new_waypoint[i - 1 - index_temp][0],
                                               new_waypoint[i - index_temp][1] - new_waypoint[i - 1 - index_temp][1]])
                    pro_speed = calc_final_speed(result_waypoint[i - index_temp - index][2], max_a, dist_pro)
                    if pro_speed >= result_waypoint[i - 1 - index_temp - index][2]:
                        break
                    else:
                        result_waypoint[i - 1 - index_temp - index][2] = pro_speed
    result_waypoint.append([new_waypoint[-1][0], new_waypoint[-1][1], result_waypoint[-1][2]])
    # for i in range(len(result_waypoint)):
    #     print("The desired speed is : %s " % result_waypoint[i][2])
    return result_waypoint


# Input : 三个坐标点
# Output: 曲率
def curvature(x, y):
    t_a = np.linalg.norm([x[1] - x[0], y[1] - y[0]])
    t_b = np.linalg.norm([x[2] - x[1], y[2] - y[1]])

    M = np.array([
        [1, -t_a, t_a**2],
        [1, 0, 0],
        [1, t_b, t_b**2]
    ])
    if np.linalg.det(M) == 0.0:
        print("原始矩阵不可逆")
        M = np.array([[1, -0.0001, 0.0001],
                     [1, 0, 0],
                     [1, 0.0001, 0.0001]])
    try:
        a = np.matmul(np.linalg.inv(M), x)
        b = np.matmul(np.linalg.inv(M), y)
    except:
        print("不存在可逆矩阵")
        print(M)


    curvature_out = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1]**2. + b[1]**2.)**1.5
    return curvature_out


def calc_final_speed(v_i, a, d):
    if v_i**2 + 2*a*d >=0:
        v_f = np.sqrt(v_i**2 + 2*a*d)
    else:
        v_f = 0
    return v_f
    # if v_f >= 0:
    #     return v_f
    # else:
    #     return 0


def inter_two_point(curr_point, next_point, distance, INTER_DIS):
    """
    给两点之间插值
    :param curr_point: 当前点的坐标[x, y, z]
    :param next_point: 下一个的的坐标[x, y, z]
    :param distance: 两个点之间的距离
    :param INTER_DIS: 插值点之间的距离
    :return: 从当前点到下一个点之前的所有插值点
    """
    result = [[curr_point[0], curr_point[1]]]
    num2Interp = int(np.floor(distance / float(INTER_DIS)) - 1)
    wp_vector = [next_point[0] - curr_point[0], next_point[1] - curr_point[1]]
    wp_uvector = wp_vector / np.linalg.norm(wp_vector)
    for j in range(num2Interp):
        next_wp_vector = INTER_DIS * float(j + 1) * wp_uvector
        result.append([curr_point[0] + next_wp_vector[0], curr_point[1] + next_wp_vector[1], 0])
    # result.append([next_point[0], next_point[1]])
    return result


# 找当前点距离RouteList最近的位置，并返回index and distance
def findClosestPoint(waypoints, curr_x, curr_y):
    index = 0
    dis = []
    for i in range(len(waypoints)):
        dis_temp = math.sqrt((math.pow((waypoints[i][0] - curr_x), 2)) + (math.pow((waypoints[i][1] - curr_y), 2)))
        dis.append(dis_temp)
    index = dis.index(min(dis))
    closest_distance = math.sqrt((math.pow((waypoints[index][0] - curr_x), 2)) +
                                 (math.pow((waypoints[index][1] - curr_y), 2)))
    return index, closest_distance


def distance_to_point(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))


def get_bezier_curve(x_list, y_list, num=30, degree=5):
    '''
    使用贝塞尔函数拟合曲线,默认使用5次贝塞尔曲线拟合轨迹，因此需要6个固定点来拟合。
    :param x_list: 固定点的x坐标
    :param y_list: 固定点的y坐标
    :param num: 需要生成路劲点的个数默认是30个
    :param degree: 贝塞尔的次数默认是5次
    :return: 路径点的坐标 [[x1, y1], [x2, y2],...]
    '''
    assert len(x_list) == len(y_list), "The length of x_list and y_list is not same"
    assert len(x_list) == degree + 1, "The degree and the length of list is not match, witch should degree + 1 = length"
    nodes = np.asfortranarray([[temp_x for temp_x in x_list], [temp_y for temp_y in y_list]])
    curve = bezier.Curve(nodes, degree=degree)
    s_vals = np.linspace(0, 1, num)
    data = curve.evaluate_multi(s_vals)
    route = []
    for i in range(len(data[0])):
        route.append([data[0][i], data[1][i]])
    return route


def get_next_200_point_from_route(routeList, now_waypoint):
    temp_index, _ = findClosestPoint(now_waypoint.transform.location.x, now_waypoint.transform.location.y, routeList)
    if temp_index+200 >= len(routeList):
        return routeList[temp_index:-1]
    else:
        return routeList[temp_index:temp_index+200]


def get_next_dis_random(start_waypoint, dis=50, interval=0.25):
    """
    返回从当前位置开始前方dis距离内的路径点，默认以0.25米为间隔
    :param interval: 两个路径点之间的间隔 单位:m
    :param start_waypoint: 起始点 carla.waypoint
    :param dis: 前方的距离 单位：m
    :return: 路径点坐标[[x1,y1], [x2,y2],...] (x,y 单位：m) Waypoint列表 [waypoint1, waypoint2,...] (waypoint:carla.waypoint)
    """
    routeList = []
    waypointList = []
    routeList.append([start_waypoint.transform.location.x, start_waypoint.transform.location.y])
    waypointList.append(start_waypoint)
    now_waypoint = start_waypoint
    for i in range(int(dis/interval)):
        next_waypoint = now_waypoint.next(interval)
        routeList.append([next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
        waypointList.append(next_waypoint[0])
        now_waypoint = next_waypoint[0]
    return routeList, waypointList


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
    for i in range(int(dis/interval)):
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
            routeList.append([next_waypoint[1].transform.location.x, next_waypoint[1].transform.location.y])
            waypointList.append(next_waypoint[1])
            now_waypoint = next_waypoint[1]
        else:
            routeList.append([next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
            waypointList.append(next_waypoint[0])
            now_waypoint = next_waypoint[0]
    return routeList, waypointList


def get_next_dis_left_turn(start_waypoint, _map, dis=50, interval=0.25, vehicle_ahead=False):
    change_interval = dis / 10
    # 获取当前点对应于左方的点
    final_path = []
    left_counterpart = start_waypoint.get_left_lane()
    # if not left_counterpart:
    #     print("Not find the left counterpart point")
    # else:
    #     left_counterpart = start_waypoint
    left_routeList, left_waypointList = get_next_dis_no_turn(left_counterpart, _map, dis, interval)
    now_routeList, now_waypointList = get_next_dis_no_turn(start_waypoint, _map, dis, interval)

    # left_x = [temp[0] for temp in left_routeList]
    # left_y = [temp[1] for temp in left_routeList]
    # now_x = [temp[0] for temp in now_routeList]
    # now_y = [temp[1] for temp in now_routeList]
    # plt.plot(left_x, left_y, linewidth='4')
    # plt.plot(now_x, now_y, linewidth='4')

    inflexion_point_start = now_routeList[int(2*change_interval/interval)]
    inflexion_point_end = left_routeList[int(5*change_interval/interval)]
    x_delta = inflexion_point_end[0] - inflexion_point_start[0]
    y_delta = inflexion_point_end[1] - inflexion_point_start[1]
    middle_first_point = [inflexion_point_start[0] + x_delta/3, inflexion_point_start[1] + y_delta/3]
    middle_second_point = [inflexion_point_end[0] - x_delta/3, inflexion_point_end[1] - y_delta/3]

    x_list = [now_routeList[0][0], inflexion_point_start[0], middle_first_point[0], middle_second_point[0]
              , inflexion_point_end[0], left_routeList[int(7*change_interval/interval)][0]]
    y_list = [now_routeList[0][1], inflexion_point_start[1], middle_first_point[1], middle_second_point[1]
              , inflexion_point_end[1], left_routeList[int(7*change_interval/interval)][1]]
    # plt.plot(x_list, y_list, color='red', linewidth='4')
    final_path = get_bezier_curve(x_list, y_list, num=int(70/interval))

    # final_x = [temp[0] for temp in final_path]
    # final_y = [temp[1] for temp in final_path]
    # plt.plot(final_x, final_y, linewidth='4')
    # plt.show()
    # path_change = path_optimizer.PathOptimizer()
    # x_delta = left_routeList[160][0] - now_routeList[80][0]
    # y_delta = left_routeList[160][1] - now_routeList[80][1]
    # extra_path = path_change.optimize_spiral(abs(x_delta), abs(y_delta), 0)
    # print("delta is {} {}".format(left_routeList[160][0] - now_routeList[80][0], left_routeList[160][1] - now_routeList[80][1]))
    # plt.plot(extra_path[0], extra_path[1])
    # plt.show()
    # for i in range(80):
    #     final_path.append([now_routeList[i][0], now_routeList[i][1]])
    # for i in range(len(extra_path[0])):
    #     final_path.append([now_routeList[80][0] + (x_delta * extra_path[0][i]) / abs(x_delta)
    #                           , now_routeList[80][1] + (y_delta * extra_path[1][i]) / abs(y_delta)])
    # final_path_append = get_bezier_curve([left_routeList[161][0], left_routeList[-1][0]]
    #                                      , [left_routeList[161][1], left_routeList[-1][1]], num=80, degree=1)
    for i in range(len(left_routeList[int(7*change_interval/interval):])):
        final_path.append([left_routeList[i+int(7*change_interval/interval)][0]
                              , left_routeList[i+int(7*change_interval/interval)][1]])
    # final_path += final_path_append
    if vehicle_ahead:
        extra_waypoints = left_waypointList[-1].next_until_lane_end(0.25)
        for i in range(len(extra_waypoints)):
            final_path.append([extra_waypoints[i].transform.location.x, extra_waypoints[i].transform.location.y
                                  , extra_waypoints[i].transform.location.z])
    return final_path


def get_next_dis_right_turn(start_waypoint, _map, dis=50, interval=0.25, vehicle_ahead=False):
    change_interval = dis / 10
    # 获取当前点对应于左方的点
    final_path_head = []
    right_counterpart = start_waypoint.get_right_lane()
    if right_counterpart is None:
        next_waypoint = start_waypoint.next(interval)
        right_counterpart = next_waypoint[0].get_right_lane()
        final_path_head.append([next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
        while right_counterpart is None:
            next_waypoint = next_waypoint[0].next(interval)
            right_counterpart = next_waypoint[0].get_right_lane()
            final_path_head.append([next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
    # if not right_counterpart:
    #     print("Not find the right counterpart point")
    # else:
    #     right_counterpart = start_waypoint
    right_routeList, right_waypointList = get_next_dis_no_turn(right_counterpart, _map, dis, interval)
    now_routeList, now_waypointList = get_next_dis_no_turn(start_waypoint, _map, dis, interval)

    inflexion_point_start = now_routeList[int(2*change_interval/interval)]
    inflexion_point_end = right_routeList[int(5*change_interval/interval)]
    x_delta = inflexion_point_end[0] - inflexion_point_start[0]
    y_delta = inflexion_point_end[1] - inflexion_point_start[1]
    middle_first_point = [inflexion_point_start[0] + x_delta/3, inflexion_point_start[1] + y_delta/3]
    middle_second_point = [inflexion_point_end[0] - x_delta/3, inflexion_point_end[1] - y_delta/3]

    x_list = [now_routeList[0][0], inflexion_point_start[0], middle_first_point[0], middle_second_point[0]
              , inflexion_point_end[0], right_routeList[int(7*change_interval/interval)][0]]
    y_list = [now_routeList[0][1], inflexion_point_start[1], middle_first_point[1], middle_second_point[1]
              , inflexion_point_end[1], right_routeList[int(7*change_interval/interval)][1]]
    final_path = get_bezier_curve(x_list, y_list, num=int(70/interval))
    final_path = final_path_head + final_path

    # path_change = path_optimizer.PathOptimizer()
    # x_delta = right_routeList[160][0] - now_routeList[80][0]
    # y_delta = right_routeList[160][1] - now_routeList[80][1]
    # extra_path = path_change.optimize_spiral(abs(x_delta), abs(y_delta), 0)
    # print("delta is {} {}".format(right_routeList[160][0] - now_routeList[80][0], right_routeList[160][1] - now_routeList[80][1]))
    # plt.plot(extra_path[0], extra_path[1])
    # plt.show()
    # for i in range(80):
    #     final_path.append([now_routeList[i][0], now_routeList[i][1]])
    # for i in range(len(extra_path[0])):
    #     final_path.append([now_routeList[80][0] + (x_delta * extra_path[0][i]) / abs(x_delta)
    #                           , now_routeList[80][1] + (y_delta * extra_path[1][i]) / abs(y_delta)])
    # final_path_append = get_bezier_curve([right_routeList[161][0], right_routeList[-1][0]]
    #                                      , [right_routeList[161][1], right_routeList[-1][1]], num=80, degree=1)
    for i in range(len(right_routeList[int(7*change_interval/interval):])):
        final_path.append([right_routeList[i+int(7*change_interval/interval)][0]
                              , right_routeList[i+int(7*change_interval/interval)][1]])
    # final_path += final_path_append
    if vehicle_ahead:
        extra_waypoints = right_waypointList[-1].next_until_lane_end(0.25)
        for i in range(len(extra_waypoints)):
            final_path.append([extra_waypoints[i].transform.location.x, extra_waypoints[i].transform.location.y])
    return final_path


# vehicles_info [{}...], route_list[[]...], cur_info[]
def judge_vehicle(vehicles_info, route_list, cur_info):
    for i in range(len(vehicles_info)):
        temp_location = [vehicles_info[i]["x"], vehicles_info[i]["y"]]
        dis = distance_to_point(temp_location[0], temp_location[1], cur_info[0], cur_info[1])
        if dis < 50:
            closest_index, closest_distance = findClosestPoint(route_list, temp_location[0], temp_location[1])
            if closest_distance < 0.2:
                return True


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
            if abs(all_waypoints_pairs[i][0].transform.rotation.yaw - all_waypoints_pairs[i][1].transform.rotation.yaw) < 1:
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
                for j in range(int(s_total/interval)-1):
                    temp_waypoint = _map.get_waypoint_xodr(temp_road_id, temp_lane_id, start_s+(symbol*(j+1)*interval))
                    if temp_waypoint is None:
                        continue
                    else:
                        final_waypoint_list.append([temp_waypoint.transform.location.x, temp_waypoint.transform.location.y])
                        final_waypoint.append(temp_waypoint)
                return final_waypoint, final_waypoint_list
    return False, False


def speed_control(Route_list, cur_pos, speed, yaw):
    find_end = False
    now_index, _ = findClosestPoint(cur_pos[0], cur_pos[1], Route_list)
    speed_ref_dis = 45 + speed * 3
    if speed_ref_dis > 70:
        speed_ref_dis = 70
    if speed_ref_dis < 45:
        speed_ref_dis = 45
    delta_index = int(speed_ref_dis/0.25)
    target_index = now_index + delta_index
    if target_index >= len(Route_list):
        target_index = len(Route_list) - 1
        find_end = True
    target_x = Route_list[target_index][0]
    target_y = Route_list[target_index][1]
    curr_x = cur_pos[0]
    curr_y = cur_pos[1]
    # 速度末端方向
    v_end = [curr_x + math.cos(math.radians(yaw)), curr_y + math.sin((math.radians(yaw)))]
    # 挡墙速度向量
    v = np.array([v_end[0] - curr_x, v_end[1] - curr_y, 0.])
    # 目标点向量
    w = np.array([target_x - curr_x, target_y - curr_y, 0.])
    # 计算夹角的弧度
    res = np.dot(w, v) / np.linalg.norm(w) * np.linalg.norm(v)
    res = math.acos(res)
    # print("The speed anchor's radian: %s" % res)
    # 将夹角固定在-1到1的范围
    # res = math.acos(np.clip(res, -1., 1.))
    # res = np.clip(abs(res), 0., 1.)
    if res > math.pi:
        if res < 0:
            res += math.pi * 2
        else:
            res -= math.pi * 2
    abs_res = abs(res)
    speed_limit = 50
    if find_end:
        speed_limit = 30
    if abs_res < 0.03:
        except_speed = speed_limit - abs_res * 30 * 3.6
    elif abs_res < 0.3:
        except_speed = speed_limit - abs_res * 5 * 3.6
    elif abs_res < 0.6:
        except_speed = speed_limit - abs_res * 10 * 3.6
    elif abs_res < 1.2:
        except_speed = speed_limit - abs_res * 5 * 3.6
    else:
        except_speed = speed_limit - abs_res * 5 * 3.6
    if except_speed < 5*3.6:
        except_speed = 5 * 3.6
    if except_speed > speed_limit:
        except_speed = speed_limit
    if except_speed < 0:
        except_speed = 3 * 3.6
    except_acc = ((except_speed - speed)/3.6)**2
    if except_acc > 5:
        except_acc = 5
    if except_acc < 1:
        except_acc = 1
    return except_acc






