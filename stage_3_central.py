#!/usr/bin/env python3

"""
Forest traversal mission — centralised swarm controller.
"""

import argparse
import sys
import math
import time
from typing import List, Optional, Tuple

import rclpy
import yaml
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler


# different formations to try out

FORMATIONS_5 = {
    "line": [
        (0.0,  1.0),
        (0.0,  0.5),
        (0.0,  0.0),
        (0.0, -0.5),
        (0.0, -1.0),
    ],
    "v": [
        (-0.8,  1.0),
        ( 0.0,  0.5),
        ( 0.8,  0.0),
        ( 0.0, -0.5),
        (-0.8, -1.0),
    ],
    "column": [
        (0.0,  0.0),
        (0.0, -0.6),
        (0.0, -1.2),
        (0.0, -1.8),
        (0.0, -2.4),
    ],
    "diamond": [
        ( 0.0,  0.7),
        ( 0.7,  0.0),
        ( 0.0,  0.0),
        (-0.7,  0.0),
        ( 0.0, -0.7),
    ],
}

def get_formation_offsets(name: str, num_drones: int) -> List[Tuple[float, float]]:
    return FORMATIONS_5[name]

# math helpers

def rotate_offset(dx: float, dy: float, heading: float) -> Tuple[float, float]:
    c, s = math.cos(heading), math.sin(heading)
    return (dx * c - dy * s, dx * s + dy * c)


def dist2d(a, b) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def segment_circle_intersects(p1, p2, centre, radius) -> bool:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    fx, fy = p1[0] - centre[0], p1[1] - centre[1]
    a = dx*dx + dy*dy
    b = 2.0 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - radius*radius
    if a < 1e-12:
        return c <= 0.0
    disc = b*b - 4.0*a*c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t1 <= 1.0 and t2 >= 0.0

# path planning

def densify_path(waypoints: List[Tuple[float, float]],
                 resolution: float = 0.05
                 ) -> List[Tuple[float, float]]:
    result = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        p1, p2 = waypoints[i], waypoints[i + 1]
        d = dist2d(p1, p2)
        n = max(1, int(math.ceil(d / resolution)))
        for j in range(1, n + 1):
            t = j / n
            result.append((p1[0] + t * (p2[0] - p1[0]),
                           p1[1] + t * (p2[1] - p1[1])))
    return result

def pre_deflect(path: List[Tuple[float, float]],
                obstacles: List[Tuple[float, float]],
                clearance: float,
                heading: float,
                deflection: float = 0.1
                ) -> List[Tuple[float, float]]:
    perp_x = -math.sin(heading)
    perp_y =  math.cos(heading)

    pts = [list(p) for p in path]
    side = {}
    for i in range(len(pts)):
        for oi, obs in enumerate(obstacles):
            if oi in side:
                continue
            d = dist2d(pts[i], obs)
            if d < clearance:
                proj = (pts[i][0] - obs[0]) * perp_x + \
                       (pts[i][1] - obs[1]) * perp_y
                side[oi] = 1.0 if proj >= 0 else -1.0

    for i in range(len(pts)):
        for oi, obs in enumerate(obstacles):
            d = dist2d(pts[i], obs)
            if d < clearance:
                s = side.get(oi, 1.0)
                strength = 1.0 - (d / clearance)
                pts[i][0] += perp_x * s * deflection * strength
                pts[i][1] += perp_y * s * deflection * strength

    return [tuple(p) for p in pts]


def push_path_clear(path: List[Tuple[float, float]],
                    obstacles: List[Tuple[float, float]],
                    clearance: float,
                    passes: int = 25
                    ) -> List[Tuple[float, float]]:
    pts = [list(p) for p in path]

    for _ in range(passes):
        changed = False
        for i in range(len(pts)):
            for obs in obstacles:
                dx = pts[i][0] - obs[0]
                dy = pts[i][1] - obs[1]
                d = math.sqrt(dx*dx + dy*dy)
                if d < clearance:
                    if d > 1e-6:
                        pts[i][0] = obs[0] + (dx / d) * clearance
                        pts[i][1] = obs[1] + (dy / d) * clearance
                    else:
                        pts[i][0] = obs[0] + clearance
                    changed = True
        if not changed:
            break

    return [tuple(p) for p in pts]


def resample_path(path: List[Tuple[float, float]],
                  num_points: int) -> List[Tuple[float, float]]:
    if len(path) < 2 or num_points < 2:
        return list(path) if path else []

    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + dist2d(path[i - 1], path[i]))
    total = cum[-1]
    if total < 1e-9:
        return [path[0]] * num_points

    result = []
    for i in range(num_points):
        target = total * i / (num_points - 1)
        for j in range(1, len(cum)):
            if cum[j] >= target - 1e-9:
                seg = cum[j] - cum[j - 1]
                t = (target - cum[j - 1]) / seg if seg > 1e-9 else 0.0
                t = max(0.0, min(1.0, t))
                x = path[j-1][0] + t * (path[j][0] - path[j-1][0])
                y = path[j-1][1] + t * (path[j][1] - path[j-1][1])
                result.append((x, y))
                break
    return result


def compute_altitude_separation(
        paths: List[List[Tuple[float, float]]],
        baselines: List[List[Tuple[float, float]]],
        num_steps: int,
        num_drones: int,
        base_height: float,
        proximity_threshold: float = 0.4,
        deviation_threshold: float = 0.3,
        alt_offset: float = 0.3,
        ramp_steps: int = 2
) -> List[List[float]]:
    """
    Simplified altitude separation.

    At each step, for every close drone pair: only the drone deviating
    more from its baseline shifts up. This ensures exactly one drone of a
    conflicting pair changes altitude.
    """
    altitudes = [[base_height] * num_steps for _ in range(num_drones)]

    deviation = [[dist2d(paths[i][s], baselines[i][s])
                  for s in range(num_steps)]
                 for i in range(num_drones)]

    shift_at = [[False] * num_steps for _ in range(num_drones)]

    for s in range(num_steps):
        for i in range(num_drones):
            for j in range(i + 1, num_drones):
                if dist2d(paths[i][s], paths[j][s]) >= proximity_threshold:
                    continue
                i_dev = deviation[i][s] > deviation_threshold
                j_dev = deviation[j][s] > deviation_threshold
                if not i_dev and not j_dev:
                    continue
                if deviation[i][s] >= deviation[j][s]:
                    shift_at[i][s] = True
                else:
                    shift_at[j][s] = True

    # Pad with ramp
    for i in range(num_drones):
        raw = list(shift_at[i])
        for s in range(num_steps):
            if raw[s]:
                for r in range(1, ramp_steps + 1):
                    if s - r >= 0:
                        shift_at[i][s - r] = True
                    if s + r < num_steps:
                        shift_at[i][s + r] = True

    # Apply with smooth ramp
    for i in range(num_drones):
        zones = []
        in_zone = False
        z_start = 0
        for s in range(num_steps):
            if shift_at[i][s] and not in_zone:
                z_start = s
                in_zone = True
            elif not shift_at[i][s] and in_zone:
                zones.append((z_start, s - 1))
                in_zone = False
        if in_zone:
            zones.append((z_start, num_steps - 1))

        for (zs, ze) in zones:
            for s in range(zs, ze + 1):
                dist_from_edge = min(s - zs, ze - s)
                if ramp_steps > 0:
                    ramp_t = min(1.0, (dist_from_edge + 1) / (ramp_steps + 1))
                else:
                    ramp_t = 1.0
                ramp_t = ramp_t * ramp_t * (3.0 - 2.0 * ramp_t)
                altitudes[i][s] = base_height + alt_offset * ramp_t

    total_shifted = sum(1 for i in range(num_drones)
                        for s in range(num_steps)
                        if altitudes[i][s] != base_height)
    if total_shifted > 0:
        affected = [i for i in range(num_drones)
                    if any(altitudes[i][s] != base_height
                           for s in range(num_steps))]
        print(f"Altitude separation: {total_shifted} step(s) shifted "
              f"for drone(s) {affected}")
    else:
        print("No altitude separation needed")

    return altitudes


def generate_centroid_path(start, end, num_steps):
    heading = math.atan2(end[1] - start[1], end[0] - start[0])
    points = []
    for i in range(num_steps):
        t = i / max(num_steps - 1, 1)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        points.append((x, y, heading))
    return points

# drone wrapper

class FormationDrone(DroneInterface):
    def __init__(self, namespace: str, verbose: bool = False,
                 use_sim_time: bool = False):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)
        self.current_behavior: Optional[BehaviorHandler] = None
        self._speed = 0.7
        self._yaw_mode = YawMode.PATH_FACING
        self._frame_id = "earth"

    def do_behavior(self, beh: str, *args) -> None:
        self.current_behavior = getattr(self, beh)
        self.current_behavior(*args)

    def send_goal(self, x, y, z, speed=None):
        if speed is None:
            speed = self._speed
        self.do_behavior("go_to", x, y, z, speed,
                         self._yaw_mode, 0.0, self._frame_id, False)

    def send_path(self, waypoints, speed=None):
        if speed is None:
            speed = self._speed
        if not waypoints:
            return
        self.do_behavior("follow_path", waypoints, speed,
                         self._yaw_mode, 0.0, self._frame_id, False)

    def goal_reached(self) -> bool:
        if not self.current_behavior:
            return False
        return self.current_behavior.status == BehaviorStatus.IDLE

# controller

class ForestTraversalController:

    def __init__(self, drones_ns, config, verbose=False, use_sim_time=False):
        self.num_drones = len(drones_ns)
        self.drones = []
        for ns in drones_ns:
            self.drones.append(
                FormationDrone(ns, verbose=verbose, use_sim_time=use_sim_time))

        stage_cfg = config.get("stage3", {})
        self.stage_center = tuple(stage_cfg.get("stage_center", [0.0, -6.0]))
        start_rel = stage_cfg.get("start_point", [-4.0, 0.0])
        end_rel   = stage_cfg.get("end_point",   [ 4.0, 0.0])
        self.start_point = (self.stage_center[0] + start_rel[0],
                            self.stage_center[1] + start_rel[1])
        self.end_point   = (self.stage_center[0] + end_rel[0],
                            self.stage_center[1] + end_rel[1])

        self.obstacle_diameter = stage_cfg.get("obstacle_diameter", 0.4)
        self.obstacle_radius   = self.obstacle_diameter / 2.0
        raw_obs = stage_cfg.get("obstacles", [])
        self.obstacles = [
            (self.stage_center[0] + o[0], self.stage_center[1] + o[1])
            for o in raw_obs
        ]

        self.flight_height = 1.0
        self.formation_name = stage_cfg.get("formation", "line")
        self.safety_margin = 0.5
        self.clearance = self.obstacle_radius + self.safety_margin

        traverse_dist = dist2d(self.start_point, self.end_point)
        self.num_steps = max(int(traverse_dist / 0.15), 30)
        self.chunk_size = 8
        self.base_speed = 1 
        self.max_speed  = 2.5

        self.heading = math.atan2(self.end_point[1] - self.start_point[1],
                                  self.end_point[0] - self.start_point[0])

        print(f"Start: {self.start_point}  End: {self.end_point}")
        print(f"Obstacles ({len(self.obstacles)}): {self.obstacles}")
        print(f"Clearance: {self.clearance:.2f} m  "
              f"Formation: {self.formation_name}  Steps: {self.num_steps}")

    # lifecycle

    def get_ready(self):
        ok = True
        for d in self.drones:
            ok = d.arm() and ok
        time.sleep(1.0)
        for d in self.drones:
            ok = d.offboard() and ok
        time.sleep(1.0)
        print(f"Ready: {ok}")
        return ok

    def takeoff(self):
        for d in self.drones:
            d.do_behavior("takeoff", self.flight_height, 0.7, False)
        self._wait_all()
        time.sleep(0.5)

    def go_to_start(self):
        offsets = get_formation_offsets(self.formation_name, self.num_drones)
        for i, drone in enumerate(self.drones):
            dx, dy = offsets[i]
            rx, ry = rotate_offset(dx, dy, self.heading)
            drone.send_goal(self.start_point[0] + rx,
                            self.start_point[1] + ry,
                            self.flight_height, 0.7)
        self._wait_all()
        time.sleep(0.5)

    def land(self):
        for d in self.drones:
            d.do_behavior("land", 0.4, False)
        self._wait_all()

    def shutdown(self):
        for d in self.drones:
            d.shutdown()

    # planning

    def _plan_paths(self):
        offsets = get_formation_offsets(self.formation_name, self.num_drones)
        centroid = generate_centroid_path(
            self.start_point, self.end_point, self.num_steps)

        # abseline paths
        baselines = []
        for i in range(self.num_drones):
            dx, dy = offsets[i]
            rx, ry = rotate_offset(dx, dy, self.heading)
            baselines.append(
                [(cx + rx, cy + ry) for cx, cy, _ in centroid])

        # densify, pre-deflect, push clear, resample
        planned_per_drone = []
        for i in range(self.num_drones):
            dense = densify_path(baselines[i], resolution=0.05)
            deflected = pre_deflect(dense, self.obstacles, self.clearance,
                                    self.heading)
            pushed = push_path_clear(deflected, self.obstacles, self.clearance)
            resampled = resample_path(pushed, self.num_steps)
            resampled = push_path_clear(resampled, self.obstacles,
                                        self.clearance, passes=10)
            planned_per_drone.append(resampled)

        # formation recovery
        result = []
        for s in range(self.num_steps):
            result.append(list(planned_per_drone[i][s]
                               for i in range(self.num_drones)))

        for s in range(self.num_steps):
            for i in range(self.num_drones):
                bx, by = baselines[i][s]
                px, py = result[s][i]

                baseline_clear = all(
                    dist2d((bx, by), obs) >= self.clearance
                    for obs in self.obstacles)
                if not baseline_clear:
                    continue

                min_obs_dist = min(dist2d((bx, by), obs)
                                   for obs in self.obstacles)
                margin = (min_obs_dist - self.clearance) / self.clearance
                alpha = min(1.0, max(0.0, margin * 2.0))

                result[s][i] = (px + alpha * (bx - px),
                                py + alpha * (by - py))

        # final per-drone densify - push - resample
        for i in range(self.num_drones):
            drone_path = [result[s][i] for s in range(self.num_steps)]
            dense = densify_path(drone_path, resolution=0.03)
            deflected = pre_deflect(dense, self.obstacles, self.clearance,
                                    self.heading)
            pushed = push_path_clear(deflected, self.obstacles, self.clearance)
            fixed = resample_path(pushed, self.num_steps)
            fixed = push_path_clear(fixed, self.obstacles,
                                    self.clearance, passes=10)
            for s in range(self.num_steps):
                result[s][i] = fixed[s]

        # altitude separation
        drone_paths_flat = [
            [result[s][i] for s in range(self.num_steps)]
            for i in range(self.num_drones)
        ]
        altitudes = compute_altitude_separation(
            drone_paths_flat, baselines, self.num_steps, self.num_drones,
            base_height=self.flight_height,
            proximity_threshold=0.4,
            alt_offset=0.3,
            ramp_steps=2)

        # verify obstacle segments
        violations = 0
        for s in range(self.num_steps - 1):
            for i in range(self.num_drones):
                for obs in self.obstacles:
                    if segment_circle_intersects(
                            result[s][i], result[s + 1][i],
                            obs, self.obstacle_radius + 0.1):
                        violations += 1
        if violations:
            print(f"{violations} segment near-misses remain")
        else:
            print("All paths obstacle-free")

        return result, baselines, altitudes

    # execution

    def execute_traversal(self):
        print("Planning...")
        sim_start = time.time()
        planned, baselines, altitudes = self._plan_paths()

        total_chunks = math.ceil(self.num_steps / self.chunk_size)

        for c in range(total_chunks):
            s0 = c * self.chunk_size
            s1 = min(s0 + self.chunk_size, self.num_steps)
            if s1 - s0 < 2:
                continue

            actual_dists = []
            baseline_dists = []
            for i in range(self.num_drones):
                ad = sum(dist2d(planned[s][i], planned[s+1][i])
                         for s in range(s0, s1-1))
                bd = sum(dist2d(baselines[i][s], baselines[i][s+1])
                         for s in range(s0, s1-1))
                actual_dists.append(ad)
                baseline_dists.append(bd)

            sorted_bd = sorted(baseline_dists)
            ref_dist = sorted_bd[len(sorted_bd) // 2]
            if ref_dist < 1e-6:
                ref_dist = max(baseline_dists) if max(baseline_dists) > 1e-6 else 1.0
            ref_time = ref_dist / self.base_speed

            speeds = []
            for i, drone in enumerate(self.drones):
                sp = (actual_dists[i] / ref_time
                      if ref_time > 1e-6 and actual_dists[i] > 1e-6
                      else self.base_speed)
                sp = max(0.2, min(sp, self.max_speed))
                speeds.append(sp)
                wps = [[planned[s][i][0], planned[s][i][1],
                         altitudes[i][s]] for s in range(s0, s1)]
                drone.send_path(wps, sp)

            self._wait_all()
            sp_str = " ".join(f"d{i}:{speeds[i]:.2f}"
                              for i in range(self.num_drones))
            print(f"Chunk {c+1}/{total_chunks}  "
                  f"steps {s0+1}–{s1}  [{sp_str}]")

        print("Traversal complete!")
        sim_elapsed = time.time() - sim_start
        print(f"Traversal wall time: {sim_elapsed:.1f}s")

    def _wait_all(self):
        while True:
            if all(d.goal_reached() for d in self.drones):
                break

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def confirm(msg='Continue'):
    return input(f"{msg}? (y/n): ").strip().lower() == 'y'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0','drone1','drone2','drone3','drone4'])
    parser.add_argument('-c', '--config', type=str,
                        default='scenarios/scenario1_stage3.yaml')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True)
    parser.add_argument('-f', '--formation', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.formation:
        config.setdefault("stage3", {})["formation"] = args.formation

    rclpy.init()
    ctrl = ForestTraversalController(
        args.namespaces, config, args.verbose, args.use_sim_time)

    if confirm("Arm & Takeoff"):
        ctrl.get_ready()
        ctrl.takeoff()
        if confirm("Go to start"):
            ctrl.go_to_start()
        if confirm("Start traversal"):
            ctrl.execute_traversal()
        confirm("Land")
        ctrl.land()

    ctrl.shutdown()
    rclpy.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    main()

    
"""
./launch_as2.bash -s scenarios/scenario1_stage3.yaml -w config_sim/config/world_swarm.yaml

./launch_ground_station.bash -n drone0,drone1,drone2,drone3,drone4 -s scenarios/scenario1_stage3.yaml

python3 stage_3_central.py -n drone0 drone1 drone2 drone3 drone4 -c scenarios/scenario1_stage3.yaml -f line
"""