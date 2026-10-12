import argparse
import collections
import sys
import time
import math
import threading
import numpy as np
import yaml
import rclpy
from geometry_msgs.msg import Pose
from tf2_msgs.msg import TFMessage
from as2_msgs.msg import YawMode
from as2_python_api.drone_interface import DroneInterface
from rclpy.qos import qos_profile_sensor_data

# Potential field tuning
ATTRACTION = 5.0
OBS_REPULSION_MAX = 15.0
DRONE_REPULSION = 1.5
DRONE_SAFE = 0.5
MAX_SPEED = 2.5
HZ = 10.0
DT = 1.0 / HZ
BRAKE_RADIUS = 1.2
PREDICTION_HORIZON = 0.3
CLOSING_GAIN = 0.5
ATTRACTION_FLOOR = 0.2

# Formation offsets from centroid
FORMATIONS = {'line': [(0.0, 2.4), (0.0, 1.2), (0.0, 0.0), (0.0, -1.2), (0.0, -2.4)], 'v': [(-1.7, 1.7), (-0.85, 0.85), (0.0, 0.0), (-0.85, -0.85), (-1.7, -1.7)], 'column': [(0.0, 0.0), (0.0, -1.2), (0.0, -2.4), (0.0, -3.6), (0.0, -4.8)], 'diamond': [(0.0, 1.2), (1.2, 0.0), (0.0, 0.0), (-1.2, 0.0), (0.0, -1.2)]}

def rotate(dx, dy, h):
    c, s = (math.cos(h), math.sin(h))
    return (dx * c - dy * s, dx * s + dy * c)

# Ground-truth obstacle tracker via bridged Ignition dynamic_pose topic
class ObstacleTracker:
    NAME_PREFIX = 'dynamic_obstacle_'

    def __init__(self, expected_count, nominal_speed):
        self.expected = expected_count
        self.nominal_speed = nominal_speed
        self._obs = {}
        self._lock = threading.Lock()
        self._msg_count = 0
        self._last_update_t = 0.0

    def on_tfmessage(self, msg, now_t):
        with self._lock:
            self._msg_count += 1
            self._last_update_t = now_t
            for tf in msg.transforms:
                name = tf.child_frame_id
                if not name.startswith(self.NAME_PREFIX):
                    continue
                x = tf.transform.translation.x
                y = tf.transform.translation.y
                p = np.array([x, y], dtype=float)
                entry = self._obs.get(name)
                if entry is None:
                    self._obs[name] = {'pos': p, 'vel': np.zeros(2), 'last_t': now_t}
                else:
                    dt = now_t - entry['last_t']
                    if dt > 0.005:
                        raw_v = (p - entry['pos']) / dt
                        rs = np.linalg.norm(raw_v)
                        if rs > 0.001:
                            entry['vel'] = raw_v / rs * min(rs, self.nominal_speed)
                        else:
                            entry['vel'] = np.zeros(2)
                        entry['pos'] = p
                        entry['last_t'] = now_t

    def snapshot(self, now_t):
        with self._lock:
            return [(e['pos'].copy(), e['vel'].copy()) for e in self._obs.values()]

    def diag_info(self):
        with self._lock:
            return {'num_tracked': len(self._obs), 'msg_count': self._msg_count, 'names': sorted(self._obs.keys())}

# Per-drone wrapper around Aerostack2 interface with self-velocity estimation
class Drone:

    def __init__(self, ns):
        self.iface = DroneInterface(ns, use_sim_time=True)
        self._prev_pos = None
        self._prev_t = 0.0
        self._vel = np.zeros(2)

    @property
    def pos(self):
        return np.array(self.iface.position)

    def update_vel(self, now_t):
        p = self.pos[:2]
        if self._prev_pos is not None:
            dt = now_t - self._prev_t
            if dt > 0.001:
                raw = (p - self._prev_pos) / dt
                self._vel = 0.7 * raw + 0.3 * self._vel
        self._prev_pos = p
        self._prev_t = now_t

    @property
    def vel(self):
        return self._vel.copy()

    def cmd(self, pt):
        self.iface.go_to(float(pt[0]), float(pt[1]), float(pt[2]), MAX_SPEED, YawMode.PATH_FACING, 0.0, 'earth', wait=False)

# Swarm controller: coordinates the formation, forces, and mission legs
class Swarm:

    def __init__(self, names, config, args):
        self.drones = [Drone(ns) for ns in names]
        self.n = len(names)
        cfg = config.get('stage4', config.get('stage3', {}))
        sc = cfg.get('stage_center', [0.0, 6.0])
        sr = cfg.get('start_point', [4.0, 0.0])
        er = cfg.get('end_point', [-4.0, 0.0])
        self.start = np.array([sc[0] + sr[0], sc[1] + sr[1]])
        self.end = np.array([sc[0] + er[0], sc[1] + er[1]])
        stage_half_x = cfg.get('stage_size', [10.0, 10.0])[0] / 2.0
        stage_half_y = cfg.get('stage_size', [10.0, 10.0])[1] / 2.0
        field_x_max = sc[0] + stage_half_x
        field_x_min = sc[0] - stage_half_x
        field_y_min = sc[1] - stage_half_y
        stage_clearance = 2.0
        if self.start[0] >= self.end[0]:
            self.pre_stage = np.array([field_x_max + stage_clearance, self.start[1]])
            self.post_stage = np.array([field_x_min - stage_clearance, self.end[1]])
        else:
            self.pre_stage = np.array([field_x_min - stage_clearance, self.start[1]])
            self.post_stage = np.array([field_x_max + stage_clearance, self.end[1]])
        approach_y = -1.5
        self.approach = np.array([self.pre_stage[0], approach_y])
        self.departure = np.array([self.post_stage[0], approach_y])
        self.obs_r = cfg.get('obstacle_diameter', 0.5) / 2.0
        self.obs_r = max(self.obs_r, 0.5 * math.sqrt(2) / 2.0)
        self.clearance = self.obs_r + args.safety_margin
        form = args.formation or cfg.get('formation', 'diamond')
        if form not in FORMATIONS:
            print(f"[WARN] Unknown formation '{form}', falling back to 'diamond'")
            form = 'diamond'
        self.offsets = FORMATIONS[form]
        self.slot_map = list(range(self.n))
        print(f'Formation: {form}')
        diff = self.end - self.start
        self.heading = math.atan2(diff[1], diff[0])
        self.dist = np.linalg.norm(diff)
        self.direction = diff / self.dist
        self.alt = args.altitude
        self.speed = args.speed
        expected = cfg.get('num_obstacles', 5)
        nominal = cfg.get('obstacle_velocity', 0.5)
        self.tracker = ObstacleTracker(expected, nominal)
        self._setup_ros_subscription()
        print(f'Approach:   {self.approach}')
        print(f'Pre-stage:  {self.pre_stage}  Start: {self.start}')
        print(f'End:        {self.end}        Post-stage: {self.post_stage}')
        print(f'Departure:  {self.departure}')
        print(f'Traverse:   {self.dist:.1f}m at {self.speed} m/s')
        print(f'Obstacle clearance: {self.clearance:.2f}m')
        print(f'Waiting for up to {expected} dynamic obstacles on /world/empty/dynamic_pose/info (bridged ground truth)')
        print('  Make sure the ros_gz_bridge is running in another terminal:')
        print('    ros2 run ros_gz_bridge parameter_bridge \\')
        print('        /world/empty/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V')
        self.trajectory_log = []
        self.current_leg = 'init'
        self._log_meta = {'approach': self.approach.tolist(), 'pre_stage': self.pre_stage.tolist(), 'start': self.start.tolist(), 'end': self.end.tolist(), 'post_stage': self.post_stage.tolist(), 'departure': self.departure.tolist(), 'stage_center': list(sc), 'stage_half_x': stage_half_x, 'stage_half_y': stage_half_y, 'clearance': float(self.clearance), 'obs_r': float(self.obs_r), 'drone_names': list(names)}

    def _setup_ros_subscription(self):
        self.tracker_node = rclpy.create_node('dynamic_obstacle_tracker')
        self.tracker_node.create_subscription(TFMessage, '/world/empty/dynamic_pose/info', self._obstacle_callback, qos_profile_sensor_data)
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.tracker_node)
        self.tracker_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.tracker_thread.start()

    def _obstacle_callback(self, msg: 'TFMessage'):
        t = self.tracker_node.get_clock().now().nanoseconds / 1000000000.0
        self.tracker.on_tfmessage(msg, t)

    def target(self, i, centroid):
        slot_idx = self.slot_map[i]
        dx, dy = self.offsets[slot_idx]
        rx, ry = rotate(dx, dy, self.heading)
        return np.array([centroid[0] + rx, centroid[1] + ry, self.alt])

    def forces(self, i, tgt, obstacles):
        pos = self.drones[i].pos[:2]
        drone_vel = self.drones[i].vel
        F_repel = np.zeros(2)
        min_obs_d = float('inf')
        max_penetration = 0.0
        # Obstacle repulsion with lookahead + closing-speed gain
        for obs_pos, obs_vel in obstacles:
            gap_now = pos - obs_pos
            d_now = np.linalg.norm(gap_now)
            if d_now < 1e-05:
                continue
            min_obs_d = min(min_obs_d, d_now)
            pos_future = pos + drone_vel * PREDICTION_HORIZON
            obs_future = obs_pos + obs_vel * PREDICTION_HORIZON
            gap_pred = pos_future - obs_future
            d_pred = np.linalg.norm(gap_pred)
            if d_pred < 1e-05:
                gap_pred = gap_now
                d_pred = d_now
            d_effective = min(d_now, d_pred)
            gap_for_dir = gap_now if d_now <= d_pred else gap_pred
            if d_effective < self.clearance:
                penetration = max(0.0, min(1.0, (self.clearance - d_effective) / (self.clearance - self.obs_r)))
                max_penetration = max(max_penetration, penetration)
                base_mag = OBS_REPULSION_MAX * (0.1 + 0.9 * penetration * penetration)
                rel_vel = drone_vel - obs_vel
                radial_now = gap_now / d_now
                closing_speed = -np.dot(rel_vel, radial_now)
                gain = 1.0 + CLOSING_GAIN * max(0.0, closing_speed)
                mag = base_mag * gain
                radial = gap_for_dir / np.linalg.norm(gap_for_dir)
                tangent = np.array([-radial[1], radial[0]])
                if np.dot(tangent, self.direction) < 0:
                    tangent = -tangent
                w = penetration
                F_repel += mag * (w * radial + (1.0 - w) * tangent)
        # Attraction scaled by deepest obstacle penetration (risk factor R)
        attract_scale = 1.0 - (1.0 - ATTRACTION_FLOOR) * max_penetration
        F_attract = ATTRACTION * attract_scale * (tgt[:2] - pos)
        F = F_attract + F_repel
        # Inter-drone repulsion
        for j, other in enumerate(self.drones):
            if j == i:
                continue
            gap = pos - other.pos[:2]
            d = np.linalg.norm(gap)
            if d > 1e-05 and d < DRONE_SAFE:
                F += DRONE_REPULSION * (1.0 / d - 1.0 / DRONE_SAFE) / (d * d) * gap / d
        return (F, min_obs_d)

    def step(self, targets, ff=None):
        t0 = time.time()
        now_t = self.tracker_node.get_clock().now().nanoseconds / 1000000000.0
        obstacles = self.tracker.snapshot(now_t)
        self._tick = getattr(self, '_tick', 0) + 1
        # Refresh self-velocity estimates before force calc
        for dr in self.drones:
            dr.update_vel(now_t)
        # Compute forces for every drone
        forces_list = []
        global_min_hazard = float('inf')
        for i in range(self.n):
            F, hd = self.forces(i, targets[i], obstacles)
            forces_list.append(F)
            global_min_hazard = min(global_min_hazard, hd)
        # Scale feed-forward by proximity to nearest obstacle
        braking = global_min_hazard < BRAKE_RADIUS
        if ff is not None and braking:
            ff_scale = max(0.0, (global_min_hazard - DRONE_SAFE) / (BRAKE_RADIUS - DRONE_SAFE))
        else:
            ff_scale = 1.0
        # Dispatch next-step commands
        for i, dr in enumerate(self.drones):
            F = forces_list[i]
            if ff is not None:
                F = F + ff[:2] * ff_scale
            mag = np.linalg.norm(F)
            if mag > MAX_SPEED:
                F = F / mag * MAX_SPEED
            nxt = dr.pos.copy()
            nxt[0] += F[0] * DT
            nxt[1] += F[1] * DT
            nxt[2] = targets[i][2]
            dr.cmd(nxt)
        # Throttled logging
        if self._tick % 10 == 0:
            n_obs = len(obstacles)
            if braking and ff is not None:
                brake_str = f' BRAKE({ff_scale:.1f})'
            elif global_min_hazard < BRAKE_RADIUS:
                brake_str = f' close({global_min_hazard:.2f}m)'
            else:
                brake_str = ''
            print(f'  tick {self._tick} obs={n_obs} min_hazard={global_min_hazard:.2f}m{brake_str}')
        if self._tick % 50 == 0:
            diag = self.tracker.diag_info()
            obs_summary = ', '.join((f'({p[0]:+.2f},{p[1]:+.2f} v={np.linalg.norm(v):.2f})' for p, v in obstacles)) or 'none'
            print(f'  [TRACKER] tracked={diag['num_tracked']} msgs={diag['msg_count']} names={diag['names']}')
            print(f'  [TRACKER] obs: {obs_summary}')
        self.trajectory_log.append({'tick': self._tick, 'sim_t': now_t, 'leg': self.current_leg, 'drones': [d.pos[:2].tolist() for d in self.drones], 'drone_slots': list(self.slot_map), 'targets': [t[:2].tolist() for t in targets], 'obstacles': [p.tolist() for p, _v in obstacles], 'obstacle_vels': [v.tolist() for _p, v in obstacles], 'min_hazard': float(global_min_hazard) if global_min_hazard < float('inf') else -1.0, 'braking': bool(braking and ff is not None), 'ff_scale': float(ff_scale)})
        # Maintain 10 Hz loop rate
        left = DT - (time.time() - t0)
        if left > 0.001:
            time.sleep(left)

    def converged(self, targets, thr=0.2):
        return all((np.linalg.norm(self.drones[i].pos[:2] - targets[i][:2]) < thr for i in range(self.n)))

    def max_err(self, targets):
        return max((np.linalg.norm(self.drones[i].pos[:2] - targets[i][:2]) for i in range(self.n)))

    def arm_and_takeoff(self):
        for d in self.drones:
            d.iface.arm()
        time.sleep(0.5)
        for d in self.drones:
            d.iface.offboard()
        time.sleep(0.5)
        for d in self.drones:
            d.iface.takeoff(self.alt, 0.7, wait=False)
        for _ in range(int(6.0 / DT)):
            if all((abs(d.pos[2] - self.alt) < 0.2 for d in self.drones)):
                break
            time.sleep(DT)
        print('Airborne.')
        self.assign_slots_nearest(target_centroid=self.approach)

    def assign_slots_nearest(self, target_centroid):
        import itertools
        current_centroid = np.mean([d.pos[:2] for d in self.drones], axis=0)
        slot_world = []
        for dx, dy in self.offsets:
            rx, ry = rotate(dx, dy, self.heading)
            slot_world.append(np.array([current_centroid[0] + rx, current_centroid[1] + ry]))
        drone_pos = [d.pos[:2] for d in self.drones]
        best_perm = None
        best_cost = float('inf')
        for perm in itertools.permutations(range(self.n)):
            cost = sum((np.linalg.norm(drone_pos[i] - slot_world[perm[i]]) for i in range(self.n)))
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        self.slot_map = list(best_perm)
        print(f'[SLOTS] Assignment: drone_i -> slot_i : {self.slot_map}  (nearest-slot cost: {best_cost:.2f}m at current centroid ({current_centroid[0]:.1f},{current_centroid[1]:.1f}))')

    def _go_to_centroid(self, centroid, label, timeout_s=30.0, thr=0.25, ramp_speed=0.6):
        print(f'Moving to {label} @ {centroid}...')
        self.current_leg = label
        start_centroid = np.mean([d.pos[:2] for d in self.drones], axis=0)
        target_centroid = np.array(centroid, dtype=float)
        travel = np.linalg.norm(target_centroid - start_centroid)
        ramp_ticks = max(1, int(travel / ramp_speed / DT))
        tgts = None
        for tick in range(ramp_ticks + 1):
            a = tick / ramp_ticks
            c = start_centroid + a * (target_centroid - start_centroid)
            tgts = [self.target(i, c) for i in range(self.n)]
            self.step(tgts)
        settle_budget = max(0, int(timeout_s / DT) - ramp_ticks)
        for _ in range(settle_budget):
            tgts = [self.target(i, target_centroid) for i in range(self.n)]
            self.step(tgts)
            if self.converged(tgts, thr):
                break
        print(f'At {label}. err={self.max_err(tgts):.2f}m')

    def go_to_pre_stage(self):
        self._go_to_centroid(self.approach, 'approach', timeout_s=30.0, thr=0.3)
        self._go_to_centroid(self.pre_stage, 'pre-stage', timeout_s=30.0, thr=0.3)

    def enter_field(self):
        print('Entering obstacle field...')
        self.current_leg = 'enter_field'
        entry_dist = np.linalg.norm(self.start - self.pre_stage)
        entry_speed = 0.4
        ticks = int(entry_dist / entry_speed / DT)
        for t in range(ticks + 1):
            a = t / max(ticks, 1)
            centroid = self.pre_stage + a * (self.start - self.pre_stage)
            tgts = [self.target(i, centroid) for i in range(self.n)]
            self.step(tgts)
        for _ in range(int(3.0 / DT)):
            tgts = [self.target(i, self.start) for i in range(self.n)]
            self.step(tgts)
            if self.converged(tgts, 0.2):
                break
        print(f'At start. err={self.max_err(tgts):.2f}m')

    def traverse(self):
        traverse_time = self.dist / self.speed
        total_ticks = int(traverse_time / DT)
        base_ff = np.array([self.direction[0] * self.speed, self.direction[1] * self.speed])
        print(f'Traversing: {total_ticks} ticks ({traverse_time:.1f}s)...')
        self.current_leg = 'traverse'
        sim_start = time.time()
        for tick in range(total_ticks + 1):
            progress = tick / max(total_ticks, 1)
            centroid = self.start + progress * (self.end - self.start)
            tgts = [self.target(i, centroid) for i in range(self.n)]
            self.step(tgts, ff=base_ff)
            if tick % max(total_ticks // 10, 1) == 0:
                print(f'  {int(progress * 100):3d}%  err={self.max_err(tgts):.2f}m')
        for _ in range(int(3.0 / DT)):
            tgts = [self.target(i, self.end) for i in range(self.n)]
            self.step(tgts)
            if self.converged(tgts, 0.15):
                break
        print(f'Traversal done. wall time: {time.time() - sim_start:.1f}s')

    def exit_field(self):
        self._go_to_centroid(self.post_stage, 'post-stage', timeout_s=30.0, thr=0.3)
        self._go_to_centroid(self.departure, 'departure', timeout_s=30.0, thr=0.3)

    def land(self):
        for d in self.drones:
            d.iface.land(0.4, wait=False)
        time.sleep(5.0)

    def shutdown(self):
        for d in self.drones:
            d.iface.shutdown()

    def save_trajectory_log(self, path):
        import json
        import os
        parent = os.path.dirname(os.path.abspath(path))
        if parent and (not os.path.exists(parent)):
            os.makedirs(parent, exist_ok=True)
            print(f'[LOG] Created directory {parent}')
        payload = {'meta': self._log_meta, 'ticks': self.trajectory_log}
        with open(path, 'w') as f:
            json.dump(payload, f)
        print(f'[LOG] Saved {len(self.trajectory_log)} ticks to {path}')

# Entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--namespaces', nargs='+', default=['drone0', 'drone1', 'drone2', 'drone3', 'drone4'])
    parser.add_argument('-c', '--config', type=str, default='scenarios/scenario1_stage4.yaml')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True)
    parser.add_argument('--altitude', type=float, default=1.0)
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--safety-margin', type=float, default=0.5)
    parser.add_argument('--formation', type=str, default='diamond', choices=list(FORMATIONS.keys()), help='Formation shape (default: diamond)')
    parser.add_argument('--log', type=str, default=None, help='Path to save trajectory log (JSON). Default: auto-generated timestamped filename.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    rclpy.init()
    swarm = Swarm(args.namespaces, config, args)
    if input('Start Stage 4 Phase 2 mission? (y/n): ').strip().lower() != 'y':
        rclpy.shutdown()
        sys.exit(0)
    swarm.arm_and_takeoff()
    try:
        # Mission legs
        swarm.go_to_pre_stage()
        swarm.enter_field()
        swarm.traverse()
        swarm.land()
    finally:
        log_path = args.log or f'stage4_run_{int(time.time())}.json'
        try:
            swarm.save_trajectory_log(log_path)
        except Exception as e:
            print(f'[LOG] Failed to save trajectory: {e}')
        swarm.shutdown()
        rclpy.shutdown()
if __name__ == '__main__':
    main()