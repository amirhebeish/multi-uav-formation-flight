import argparse
import sys
import math
import time
import random
from typing import List, Optional, Tuple
import rclpy
import yaml
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler

# Formation offsets from centroid
FORMATIONS_5 = {'line': [(0.0, 1.0), (0.0, 0.5), (0.0, 0.0), (0.0, -0.5), (0.0, -1.0)], 'v': [(-0.8, 1.0), (0.0, 0.5), (0.8, 0.0), (0.0, -0.5), (-0.8, -1.0)], 'column': [(0.0, 0.0), (0.0, -0.6), (0.0, -1.2), (0.0, -1.8), (0.0, -2.4)], 'diamond': [(0.0, 0.7), (0.7, 0.0), (0.0, 0.0), (-0.7, 0.0), (0.0, -0.7)]}

# Geometry helpers
def get_formation_offsets(name: str, num_drones: int) -> List[Tuple[float, float]]:
    return FORMATIONS_5[name]

def rotate_offset(dx: float, dy: float, heading: float) -> Tuple[float, float]:
    c, s = (math.cos(heading), math.sin(heading))
    return (dx * c - dy * s, dx * s + dy * c)

def dist2d(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Forward-simulates obstacle motion with elastic wall bouncing
def simulate_obstacle_trajectories(initial_positions: List[Tuple[float, float]], speed: float, field_bounds: Tuple[float, float, float, float], times: List[float], seed: int=42) -> List[List[Tuple[float, float]]]:
    xmin, xmax, ymin, ymax = field_bounds
    rng = random.Random(seed)
    n_obs = len(initial_positions)
    states = []
    for x0, y0 in initial_positions:
        th = rng.uniform(0.0, 2.0 * math.pi)
        vx = speed * math.cos(th)
        vy = speed * math.sin(th)
        states.append([x0, y0, vx, vy])
    internal_dt = 0.02
    if not times:
        return [[] for _ in range(n_obs)]
    t_max = max(times)
    n_internal = int(math.ceil(t_max / internal_dt)) + 1
    hist = [[(s[0], s[1])] for s in states]
    hist_t = [0.0]
    for k in range(1, n_internal):
        t = k * internal_dt
        hist_t.append(t)
        for i, s in enumerate(states):
            s[0] += s[2] * internal_dt
            s[1] += s[3] * internal_dt
            if s[0] < xmin:
                s[0] = xmin + (xmin - s[0])
                s[2] = -s[2]
            elif s[0] > xmax:
                s[0] = xmax - (s[0] - xmax)
                s[2] = -s[2]
            if s[1] < ymin:
                s[1] = ymin + (ymin - s[1])
                s[3] = -s[3]
            elif s[1] > ymax:
                s[1] = ymax - (s[1] - ymax)
                s[3] = -s[3]
            hist[i].append((s[0], s[1]))
    result = []
    for i in range(n_obs):
        samples = []
        for t_q in times:
            idx = min(int(round(t_q / internal_dt)), len(hist[i]) - 1)
            samples.append(hist[i][idx])
        result.append(samples)
    return result

def densify_path(waypoints: List[Tuple[float, float]], resolution: float=0.05) -> List[Tuple[float, float]]:
    result = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        p1, p2 = (waypoints[i], waypoints[i + 1])
        d = dist2d(p1, p2)
        n = max(1, int(math.ceil(d / resolution)))
        for j in range(1, n + 1):
            t = j / n
            result.append((p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])))
    return result

def resample_path(path: List[Tuple[float, float]], num_points: int) -> List[Tuple[float, float]]:
    if len(path) < 2 or num_points < 2:
        return list(path) if path else []
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + dist2d(path[i - 1], path[i]))
    total = cum[-1]
    if total < 1e-09:
        return [path[0]] * num_points
    result = []
    for i in range(num_points):
        target = total * i / (num_points - 1)
        for j in range(1, len(cum)):
            if cum[j] >= target - 1e-09:
                seg = cum[j] - cum[j - 1]
                t = (target - cum[j - 1]) / seg if seg > 1e-09 else 0.0
                t = max(0.0, min(1.0, t))
                x = path[j - 1][0] + t * (path[j][0] - path[j - 1][0])
                y = path[j - 1][1] + t * (path[j][1] - path[j - 1][1])
                result.append((x, y))
                break
    return result

def push_path_clear_dynamic(path: List[Tuple[float, float]], obstacle_positions_per_step: List[List[Tuple[float, float]]], clearance: float, passes: int=25) -> List[Tuple[float, float]]:
    pts = [list(p) for p in path]
    n = min(len(pts), len(obstacle_positions_per_step))
    for _ in range(passes):
        changed = False
        for s in range(n):
            obs_at_s = obstacle_positions_per_step[s]
            for obs in obs_at_s:
                dx = pts[s][0] - obs[0]
                dy = pts[s][1] - obs[1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < clearance:
                    if d > 1e-06:
                        pts[s][0] = obs[0] + dx / d * clearance
                        pts[s][1] = obs[1] + dy / d * clearance
                    else:
                        pts[s][0] = obs[0] + clearance
                    changed = True
        if not changed:
            break
    return [tuple(p) for p in pts]

def generate_centroid_path(start, end, num_steps):
    heading = math.atan2(end[1] - start[1], end[0] - start[0])
    points = []
    for i in range(num_steps):
        t = i / max(num_steps - 1, 1)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        points.append((x, y, heading))
    return points

# Per-drone wrapper around Aerostack2 interface
class FormationDrone(DroneInterface):

    def __init__(self, namespace: str, verbose: bool=False, use_sim_time: bool=False):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)
        self.current_behavior: Optional[BehaviorHandler] = None
        self._speed = 0.7
        self._yaw_mode = YawMode.PATH_FACING
        self._frame_id = 'earth'

    def do_behavior(self, beh: str, *args) -> None:
        self.current_behavior = getattr(self, beh)
        self.current_behavior(*args)

    def send_goal(self, x, y, z, speed=None):
        if speed is None:
            speed = self._speed
        self.do_behavior('go_to', x, y, z, speed, self._yaw_mode, 0.0, self._frame_id, False)

    def send_path(self, waypoints, speed=None):
        if speed is None:
            speed = self._speed
        if not waypoints:
            return
        self.do_behavior('follow_path', waypoints, speed, self._yaw_mode, 0.0, self._frame_id, False)

    def goal_reached(self) -> bool:
        if not self.current_behavior:
            return False
        return self.current_behavior.status == BehaviorStatus.IDLE

# Centralised controller: plans all paths before execution
class DynamicTraversalController:

    def __init__(self, drones_ns, config, verbose=False, use_sim_time=False):
        self.num_drones = len(drones_ns)
        self.drones = []
        for ns in drones_ns:
            self.drones.append(FormationDrone(ns, verbose=verbose, use_sim_time=use_sim_time))
        stage_cfg = config.get('stage4', config)
        self.stage_center = tuple(stage_cfg.get('stage_center', [0.0, 6.0]))
        start_rel = stage_cfg.get('start_point', [4.0, 0.0])
        end_rel = stage_cfg.get('end_point', [-4.0, 0.0])
        self.start_point = (self.stage_center[0] + start_rel[0], self.stage_center[1] + start_rel[1])
        self.end_point = (self.stage_center[0] + end_rel[0], self.stage_center[1] + end_rel[1])
        self.obstacle_diameter = stage_cfg.get('obstacle_diameter', 0.5)
        self.obstacle_radius = max(self.obstacle_diameter / 2.0, 0.5 * math.sqrt(2) / 2.0)
        self.obstacle_speed = stage_cfg.get('obstacle_velocity', 0.5)
        self.num_obstacles = stage_cfg.get('num_obstacles', 5)
        stage_size = stage_cfg.get('stage_size', [10.0, 10.0])
        hx = stage_size[0] / 2.0
        hy = stage_size[1] / 2.0
        self.field_bounds = (self.stage_center[0] - hx, self.stage_center[0] + hx, self.stage_center[1] - hy, self.stage_center[1] + hy)
        self.flight_height = 1.0
        self.formation_name = stage_cfg.get('formation', 'line')
        self.safety_margin = 0.5
        self.clearance = self.obstacle_radius + self.safety_margin
        traverse_dist = dist2d(self.start_point, self.end_point)
        self.num_steps = max(int(traverse_dist / 0.1), 50)
        self.chunk_size = 8
        self.base_speed = 0.5
        self.max_speed = 2.5
        self.heading = math.atan2(self.end_point[1] - self.start_point[1], self.end_point[0] - self.start_point[0])
        self.dt_per_step = traverse_dist / self.base_speed / max(self.num_steps - 1, 1)
        self.step_times = [s * self.dt_per_step for s in range(self.num_steps)]
        raw_obs = stage_cfg.get('obstacles', None)
        if raw_obs:
            self.obstacle_initial_positions = [(self.stage_center[0] + o[0], self.stage_center[1] + o[1]) for o in raw_obs[:self.num_obstacles]]
        else:
            self.obstacle_initial_positions = self._seed_initial_positions()
        self.obstacle_trajectories = simulate_obstacle_trajectories(self.obstacle_initial_positions, self.obstacle_speed, self.field_bounds, self.step_times)
        self.obstacles_per_step = [[self.obstacle_trajectories[k][s] for k in range(self.num_obstacles)] for s in range(self.num_steps)]
        print(f'Start: {self.start_point}  End: {self.end_point}')
        print(f'Obstacles: {self.num_obstacles} @ {self.obstacle_speed:.2f} m/s')
        print(f'Clearance: {self.clearance:.2f} m  Formation: {self.formation_name}  Steps: {self.num_steps}  dt/step: {self.dt_per_step:.3f} s')

    def _seed_initial_positions(self):
        xmin, xmax, ymin, ymax = self.field_bounds
        margin = 1.5
        rng = random.Random(123)
        pts = []
        tries = 0
        while len(pts) < self.num_obstacles and tries < 500:
            x = rng.uniform(xmin + margin, xmax - margin)
            y = rng.uniform(ymin + margin, ymax - margin)
            if all((math.hypot(x - px, y - py) > 1.2 for px, py in pts)):
                pts.append((x, y))
            tries += 1
        if len(pts) < self.num_obstacles:
            while len(pts) < self.num_obstacles:
                pts.append((self.stage_center[0], self.stage_center[1]))
        return pts

    def get_ready(self):
        ok = True
        for d in self.drones:
            ok = d.arm() and ok
        time.sleep(1.0)
        for d in self.drones:
            ok = d.offboard() and ok
        time.sleep(1.0)
        print(f'Ready: {ok}')
        return ok

    def takeoff(self):
        for d in self.drones:
            d.do_behavior('takeoff', self.flight_height, 0.7, False)
        self._wait_all()
        time.sleep(0.5)

    def go_to_start(self):
        offsets = get_formation_offsets(self.formation_name, self.num_drones)
        for i, drone in enumerate(self.drones):
            dx, dy = offsets[i]
            rx, ry = rotate_offset(dx, dy, self.heading)
            drone.send_goal(self.start_point[0] + rx, self.start_point[1] + ry, self.flight_height, 0.7)
        self._wait_all()
        time.sleep(0.5)

    def land(self):
        for d in self.drones:
            d.do_behavior('land', 0.4, False)
        self._wait_all()

    def shutdown(self):
        for d in self.drones:
            d.shutdown()

    def _plan_paths(self):
        offsets = get_formation_offsets(self.formation_name, self.num_drones)
        centroid = generate_centroid_path(self.start_point, self.end_point, self.num_steps)
        # Baseline straight-line paths per drone
        baselines = []
        for i in range(self.num_drones):
            dx, dy = offsets[i]
            rx, ry = rotate_offset(dx, dy, self.heading)
            baselines.append([(cx + rx, cy + ry) for cx, cy, _ in centroid])
        # Iterative push away from predicted obstacle positions
        planned_per_drone = []
        for i in range(self.num_drones):
            path = list(baselines[i])
            path = push_path_clear_dynamic(path, self.obstacles_per_step, self.clearance, passes=40)
            dense = densify_path(path, resolution=0.1)
            dense_schedule = self._schedule_for_dense_path(len(dense), self.obstacles_per_step)
            dense = push_path_clear_dynamic(dense, dense_schedule, self.clearance, passes=15)
            fixed = resample_path(dense, self.num_steps)
            fixed = push_path_clear_dynamic(fixed, self.obstacles_per_step, self.clearance, passes=10)
            planned_per_drone.append(fixed)
        result = []
        for s in range(self.num_steps):
            result.append([planned_per_drone[i][s] for i in range(self.num_drones)])
        # Pull each drone back toward baseline when the baseline is clear
        for s in range(self.num_steps):
            for i in range(self.num_drones):
                bx, by = baselines[i][s]
                px, py = result[s][i]
                baseline_clear = all((dist2d((bx, by), obs) >= self.clearance for obs in self.obstacles_per_step[s]))
                if not baseline_clear:
                    continue
                min_obs_dist = min((dist2d((bx, by), obs) for obs in self.obstacles_per_step[s]))
                margin = (min_obs_dist - self.clearance) / self.clearance
                alpha = min(1.0, max(0.0, margin * 2.0))
                result[s][i] = (px + alpha * (bx - px), py + alpha * (by - py))
        # Verify clearance on final paths
        violations = 0
        for s in range(self.num_steps):
            for i in range(self.num_drones):
                for obs in self.obstacles_per_step[s]:
                    if dist2d(result[s][i], obs) < self.clearance - 0.05:
                        violations += 1
        if violations:
            print(f'Warning: {violations} waypoint/step clearance violations remain after planning.')
        else:
            print('All planned waypoints satisfy obstacle clearance.')
        return (result, baselines)

    def _schedule_for_dense_path(self, dense_len, sparse_schedule):
        n_sparse = len(sparse_schedule)
        out = []
        for j in range(dense_len):
            frac = j / max(dense_len - 1, 1)
            idx = int(frac * (n_sparse - 1))
            out.append(sparse_schedule[idx])
        return out

    def execute_traversal(self):
        print('Planning...')
        plan_start = time.time()
        planned, baselines = self._plan_paths()
        print(f'Planning done in {time.time() - plan_start:.2f}s')
        sim_start = time.time()
        total_chunks = math.ceil(self.num_steps / self.chunk_size)
        # Dispatch waypoints in chunks with per-drone speed compensation
        for c in range(total_chunks):
            s0 = c * self.chunk_size
            s1 = min(s0 + self.chunk_size, self.num_steps)
            if s1 - s0 < 2:
                continue
            # Per-drone actual vs baseline path lengths for this chunk
            actual_dists = []
            baseline_dists = []
            for i in range(self.num_drones):
                ad = sum((dist2d(planned[s][i], planned[s + 1][i]) for s in range(s0, s1 - 1)))
                bd = sum((dist2d(baselines[i][s], baselines[i][s + 1]) for s in range(s0, s1 - 1)))
                actual_dists.append(ad)
                baseline_dists.append(bd)
            # Reference time from the median baseline; scale each drone's speed to match
            sorted_bd = sorted(baseline_dists)
            ref_dist = sorted_bd[len(sorted_bd) // 2]
            if ref_dist < 1e-06:
                ref_dist = max(baseline_dists) if max(baseline_dists) > 1e-06 else 1.0
            ref_time = ref_dist / self.base_speed
            speeds = []
            for i, drone in enumerate(self.drones):
                sp = actual_dists[i] / ref_time if ref_time > 1e-06 and actual_dists[i] > 1e-06 else self.base_speed
                sp = max(0.2, min(sp, self.max_speed))
                speeds.append(sp)
                wps = [[planned[s][i][0], planned[s][i][1], self.flight_height] for s in range(s0, s1)]
                drone.send_path(wps, sp)
            # Block until all drones finish this chunk
            self._wait_all()
            sp_str = ' '.join((f'd{i}:{speeds[i]:.2f}' for i in range(self.num_drones)))
            print(f'Chunk {c + 1}/{total_chunks}  steps {s0 + 1}–{s1}  [{sp_str}]')
        sim_elapsed = time.time() - sim_start
        print(f'Traversal complete! wall time: {sim_elapsed:.1f}s')

    def _wait_all(self):
        while True:
            if all((d.goal_reached() for d in self.drones)):
                break

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def confirm(msg='Continue'):
    return input(f'{msg}? (y/n): ').strip().lower() == 'y'

# Entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--namespaces', nargs='+', default=['drone0', 'drone1', 'drone2', 'drone3', 'drone4'])
    parser.add_argument('-c', '--config', type=str, default='scenarios/scenario1_stage4.yaml')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True)
    parser.add_argument('-f', '--formation', type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.formation:
        config.setdefault('stage4', {})['formation'] = args.formation
    rclpy.init()
    ctrl = DynamicTraversalController(args.namespaces, config, args.verbose, args.use_sim_time)
    # Mission legs
    if confirm('Arm & Takeoff'):
        ctrl.get_ready()
        ctrl.takeoff()
        if confirm('Go to start'):
            ctrl.go_to_start()
        if confirm('Start traversal'):
            ctrl.execute_traversal()
        confirm('Land')
        ctrl.land()
    ctrl.shutdown()
    rclpy.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    main()