#!/usr/bin/env python3
"""Stage 2: Window Traversal with gradient-descent MPC, centralised and decentralised variants."""

import argparse
import sys
import time

import numpy as np
import rclpy
import yaml
from as2_msgs.msg import BehaviorStatus, YawMode
from as2_python_api.drone_interface import DroneInterface


# Formation and constants

V_OFFSETS = np.array([
    (0.0,  0.0),
    (0.6,  0.6),
    (-0.6, 0.6),
    (1.2,  1.2),
    (-1.2, 1.2),
])

ALT_DEFAULT = 1.5


# Trajectory recorder - writes per-drone XYZ samples to a CSV and generates plots
class TrajectoryRecorder:
    def __init__(self, n_drones):
        self.samples = []  # list of (t, [(x,y,z) per drone])
        self.t0 = None
        self.n_drones = n_drones

    def start(self):
        self.t0 = time.time()

    def sample(self, drones):
        if self.t0 is None:
            return
        t = time.time() - self.t0
        positions = []
        for d in drones:
            p = d.position if hasattr(d, 'position') else d.pos
            positions.append((float(p[0]), float(p[1]), float(p[2])))
        self.samples.append((t, positions))

    def save(self, path):
        if not self.samples:
            return
        n = len(self.samples[0][1])
        with open(path, 'w') as f:
            header = ['t']
            for i in range(n):
                header += [f'd{i}_x', f'd{i}_y', f'd{i}_z']
            f.write(','.join(header) + '\n')
            for t, positions in self.samples:
                row = [f'{t:.3f}']
                for (x, y, z) in positions:
                    row += [f'{x:.4f}', f'{y:.4f}', f'{z:.4f}']
                f.write(','.join(row) + '\n')
        print(f'Trajectory saved: {path} ({len(self.samples)} samples)')

    def _to_arrays(self):
        """Convert samples to (times, positions) arrays."""
        times = np.array([s[0] for s in self.samples])
        positions = np.zeros((len(self.samples), self.n_drones, 3))
        for ti, (_, plist) in enumerate(self.samples):
            for di, p in enumerate(plist):
                positions[ti, di] = p
        return times, positions

    def plot(self, windows, method_name, out_prefix, centre=None,
             planned_paths=None, method_kind='centralised'):
        """Save report figure: 3-panel for centralised, single-panel for decentralised."""
        if len(self.samples) < 5:
            print('Not enough samples to plot')
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib not available, skipping plot')
            return

        times, positions = self._to_arrays()
        n = self.n_drones
        colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd']

        def draw_walls_and_windows(ax, show_labels=True):
            for idx, w in enumerate(windows):
                hg = w['gap'] / 2
                ax.plot([w['x']-5.0, w['x']-hg], [w['y'], w['y']],
                        'k-', linewidth=3, zorder=3)
                ax.plot([w['x']+hg, w['x']+5.0], [w['y'], w['y']],
                        'k-', linewidth=3, zorder=3)
                ax.plot([w['x']-hg, w['x']+hg], [w['y'], w['y']],
                        color='gray', linestyle='--', linewidth=1,
                        alpha=0.5, zorder=2)
                if show_labels:
                    ax.annotate(f"W{idx+1}",
                                xy=(w['x'], w['y']),
                                xytext=(w['x']+2.8, w['y']+0.25),
                                fontsize=9, ha='left',
                                bbox=dict(boxstyle='round,pad=0.2',
                                          fc='white', ec='gray', alpha=0.8))

        if method_kind == 'centralised' and centre is not None and planned_paths is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))

            # Panel 1: Baseline V-formation
            ax = axes[0]
            draw_walls_and_windows(ax, show_labels=True)
            # Straight line from each drone's V-slot start to end
            # Wings are geometrically behind the leader
            start_y = centre[1] + 4.0
            end_y = centre[1] - 6.0
            for i in range(n):
                ox, oy = V_OFFSETS[i]
                x = centre[0] + ox
                line_start_y = start_y + oy
                line_end_y = end_y + oy
                ax.plot([x, x], [line_start_y, line_end_y],
                        color=colors[i % len(colors)], linewidth=1.8,
                        label=f'Drone {i}', zorder=4)
                ax.plot(x, line_start_y, marker='o', color=colors[i % len(colors)],
                        markersize=6, zorder=5)
                ax.plot(x, line_end_y, marker='s', color=colors[i % len(colors)],
                        markersize=6, zorder=5)
            ax.set_title('Baseline (ideal formation)')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend(loc='lower left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.set_xlim(-4, 4)
            max_offset_y = float(max(V_OFFSETS[:, 1]))
            y_top = start_y + max_offset_y + 1.0
            y_bot = end_y - 1.0
            ax.set_ylim(y_bot, y_top)

            # Panel 2: Planned obstacle-avoiding trajectories
            ax = axes[1]
            draw_walls_and_windows(ax, show_labels=True)
            for i in range(n):
                ax.plot(positions[:, i, 0], positions[:, i, 1],
                        color=colors[i % len(colors)], linewidth=1.8,
                        label=f'Drone {i}', zorder=4)
                ax.plot(positions[0, i, 0], positions[0, i, 1],
                        marker='o', color=colors[i % len(colors)],
                        markersize=6, zorder=5)
                ax.plot(positions[-1, i, 0], positions[-1, i, 1],
                        marker='s', color=colors[i % len(colors)],
                        markersize=6, zorder=5)
            ax.set_title('Planned (obstacle-avoiding)')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend(loc='lower left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.set_xlim(-4, 4)
            max_offset_y = float(max(V_OFFSETS[:, 1]))
            y_top = start_y + max_offset_y + 1.0
            y_bot = end_y - 1.0
            ax.set_ylim(y_bot, y_top)

            # Panel 3: Altitude profile
            ax = axes[2]
            for i in range(n):
                ax.plot(times, positions[:, i, 2],
                        color=colors[i % len(colors)], linewidth=1.5,
                        label=f'Drone {i}')
            ax.set_title('Altitude profile')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Altitude Z (m)')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            p1 = f"{out_prefix}_combined.png"
            fig.savefig(p1, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Plot saved: {p1}')

        else:
            fig, ax = plt.subplots(figsize=(8, 7))
            draw_walls_and_windows(ax, show_labels=True)
            for i in range(n):
                ax.plot(positions[:, i, 0], positions[:, i, 1],
                        color=colors[i % len(colors)], linewidth=1.8,
                        label=f'Drone {i}', zorder=4)
                ax.plot(positions[0, i, 0], positions[0, i, 1],
                        marker='o', color=colors[i % len(colors)],
                        markersize=7, zorder=5)
                ax.plot(positions[-1, i, 0], positions[-1, i, 1],
                        marker='s', color=colors[i % len(colors)],
                        markersize=7, zorder=5)
            ax.set_title('Recorded (actual flight)')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend(loc='lower left', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.set_xlim(-4, 4)
            fig.tight_layout()
            p1 = f"{out_prefix}_recorded.png"
            fig.savefig(p1, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Plot saved: {p1}')


# Geometry

def parse_stage2(cfg):
    s2 = cfg.get('stage2', {})
    centre = np.array(s2.get('stage_center', [0.0, -6.0]), dtype=float)
    windows = []
    for key in sorted(s2.get('windows', {}).keys()):
        w = s2['windows'][key]
        pos = centre + np.array(w['center'], dtype=float)
        z_mid = w['distance_floor'] + w['height'] / 2.0
        hg = w['gap_width'] / 2.0
        windows.append({
            'x': float(pos[0]), 'y': float(pos[1]), 'z': float(z_mid),
            'gap': float(w['gap_width']), 'height': float(w['height']),
            'thickness': float(w.get('thickness', 0.3)),
            'x_left': float(pos[0]) - hg, 'x_right': float(pos[0]) + hg,
        })
    return centre, windows


def wall_segments(windows):
    A, B = [], []
    for w in windows:
        hg = w['gap'] / 2
        A.append([w['x'] - 5.0, w['y']]); B.append([w['x'] - hg, w['y']])
        A.append([w['x'] + hg, w['y']]); B.append([w['x'] + 5.0, w['y']])
    return np.array(A, dtype=float), np.array(B, dtype=float)


def seg_dist_batch(points, A, B):
    P = points[:, None, :]
    AB = (B - A)[None, :, :]
    AP = P - A[None, :, :]
    denom = np.sum(AB * AB, axis=-1)
    denom = np.where(denom < 1e-9, 1.0, denom)
    t = np.clip(np.sum(AP * AB, axis=-1) / denom, 0, 1)
    C = A[None, :, :] + t[:, :, None] * AB
    d = np.linalg.norm(P - C, axis=-1)
    return d, C


def soft_clamp_batch(targets, drone_ys, windows, blend_range=2.0, margin=0.3):
    result = targets.copy()
    for w in windows:
        dy = np.abs(drone_ys - w['y'])
        weight = np.where(dy < blend_range, (1.0 - dy / blend_range) ** 2, 0.0)
        hg = w['gap'] / 2 - margin
        gap_min, gap_max = w['x'] - hg, w['x'] + hg
        clamped = np.clip(result[:, 0], gap_min, gap_max)
        result[:, 0] = result[:, 0] * (1 - weight) + clamped * weight
    return result


def linspace_path(start, end, n):
    return np.array([start + (end - start) * i / (n - 1) for i in range(n)])


def path_altitude_smooth(y, centre, windows, alt=ALT_DEFAULT):
    w1y = windows[0]['y']
    w2y = windows[1]['y'] if len(windows) > 1 else w1y
    z1 = windows[0]['z']
    z2 = windows[1]['z'] if len(windows) > 1 else z1
    def smoothstep(edge0, edge1, x):
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0, 1)
        return t * t * (3 - 2 * t)
    if y >= w1y + 2.0: return alt
    if y >= w1y - 0.3:
        t = smoothstep(w1y + 2.0, w1y + 0.3, y)
        return alt * (1 - t) + z1 * t
    if len(windows) > 1 and y >= w2y + 2.0:
        t = smoothstep(w1y - 0.3, w2y + 2.0, y)
        return z1 * (1 - t) + z2 * t
    if len(windows) > 1 and y >= w2y - 0.3:
        return z2
    if len(windows) > 1 and y >= w2y - 2.3:
        t = smoothstep(w2y - 0.3, w2y - 2.3, y)
        return z2 * (1 - t) + alt * t
    return alt


# MPC planner

def plan_three_centroids(centre, windows):
    """Returns three centroid segments. Each is strictly monotonic in y."""
    w1 = windows[0]
    w2 = windows[1] if len(windows) > 1 else None

    A_start = np.array([w1['x'], centre[1] + 4.0])
    A_end   = np.array([w1['x'], w1['y'] - 1.5])
    A_nc = linspace_path(A_start, A_end, 40)

    if w2 is None:
        B_nc = np.array([A_end])
        C_nc = linspace_path(A_end, np.array([centre[0], centre[1] - 6.0]), 30)
        return A_nc, B_nc, C_nc

    B_start = A_end
    B_end = np.array([w2['x'], w2['y'] + 1.5])
    B_nc = linspace_path(B_start, B_end, 30)

    C_part1 = linspace_path(B_end, np.array([w2['x'], w2['y'] - 1.5]), 25)
    C_part2 = linspace_path(np.array([w2['x'], w2['y'] - 1.5]),
                            np.array([centre[0], centre[1] - 6.0]), 25)
    C_nc = np.vstack([C_part1, C_part2[1:]])
    return A_nc, B_nc, C_nc


def optimise_segment(centroid_seg, offset, windows,
                     iters=400, lr=0.03, clearance=0.35, margin=0.3,
                     w_wall=120.0, w_form=1.5, w_smooth=3.0,
                     max_step=0.015):
    A, B = wall_segments(windows)
    path = centroid_seg + offset
    path = soft_clamp_batch(path, path[:, 1], windows, margin=margin)
    for _ in range(iters):
        grad = np.zeros_like(path)
        d, C = seg_dist_batch(path, A, B)
        viol = np.maximum(clearance - d, 0.0)
        d_safe = np.maximum(d, 1e-6)
        direction = (path[:, None, :] - C) / d_safe[:, :, None]
        g_wall = np.sum(-2 * viol[:, :, None] * direction, axis=1)
        grad += w_wall * g_wall
        vslots = centroid_seg + offset
        vslots = soft_clamp_batch(vslots, path[:, 1], windows, margin=margin)
        grad += w_form * 2 * (path - vslots)
        curv = np.zeros_like(path)
        curv[1:-1] = path[:-2] - 2 * path[1:-1] + path[2:]
        grad += w_smooth * (-2 * curv)
        grad[0] = 0; grad[-1] = 0
        step = lr * grad
        step_mag = np.linalg.norm(step, axis=1, keepdims=True)
        step = np.where(step_mag > max_step, step / np.maximum(step_mag, 1e-6) * max_step, step)
        path -= step

    # Laplacian smoothing with wall push-out
    for _ in range(15):
        new_path = path.copy()
        new_path[1:-1] = 0.7 * path[1:-1] + 0.15 * path[:-2] + 0.15 * path[2:]
        path = new_path
        d, C = seg_dist_batch(path, A, B)
        viol = np.maximum(clearance - d, 0.0)
        for i in range(1, len(path) - 1):
            for k in range(len(A)):
                if viol[i, k] > 0:
                    dd = max(d[i, k], 1e-6)
                    dirv = (path[i] - C[i, k]) / dd
                    path[i] += (clearance - d[i, k]) * dirv

    # Enforce monotonic y (no backtracking)
    for i in range(1, len(path)):
        if path[i, 1] > path[i - 1, 1]:
            path[i, 1] = path[i - 1, 1]
    return path


def plan_all(centre, windows):
    """Returns 3 lists (A, B, C), each with 5 drone paths in xyz."""
    A_nc, B_nc, C_nc = plan_three_centroids(centre, windows)
    A_paths, B_paths, C_paths = [], [], []
    for idx in range(len(V_OFFSETS)):
        offset = V_OFFSETS[idx]
        pA = optimise_segment(A_nc, offset, windows)
        pB = optimise_segment(B_nc, offset, windows) if len(B_nc) > 1 else B_nc + offset
        pC = optimise_segment(C_nc, offset, windows)
        def addz(p):
            z = np.array([path_altitude_smooth(y, centre, windows) for y in p[:, 1]])
            return np.column_stack([p, z])
        A_paths.append(addz(pA))
        B_paths.append(addz(pB))
        C_paths.append(addz(pC))
    return A_paths, B_paths, C_paths


def verify_segments(A_paths, B_paths, C_paths, windows):
    A, B = wall_segments(windows)
    min_wall = float('inf')
    for paths in (A_paths, B_paths, C_paths):
        for p in paths:
            d, _ = seg_dist_batch(p[:, :2], A, B)
            min_wall = min(min_wall, float(d.min()))
    min_drone = float('inf')
    for paths in (A_paths, B_paths, C_paths):
        n = len(paths[0])
        for t in range(n):
            for i in range(5):
                for j in range(i + 1, 5):
                    dd = float(np.linalg.norm(paths[i][t, :2] - paths[j][t, :2]))
                    if dd < min_drone: min_drone = dd
    return min_wall, min_drone


# Centralised: Joint MPC
# A single planner optimises all five drone trajectories jointly. The
# mission is split into three segments (approach W1, transition, exit W2)
# and each segment is dispatched as a separate follow_path call so the
# drones do not oscillate during path transitions.

class CentralDrone(DroneInterface):
    def __init__(self, ns, verbose=False, use_sim_time=True):
        super().__init__(ns, verbose=verbose, use_sim_time=use_sim_time)
        self.beh = None
        self._yaw = YawMode.PATH_FACING
        self._f = "earth"

    def _ex(self, name, *args):
        self.beh = getattr(self, name)
        self.beh(*args)

    def go(self, x, y, z, speed=0.8):
        self._ex("go_to", x, y, z, speed, self._yaw, 0.0, self._f, False)

    def follow(self, wps, speed=0.8):
        if wps:
            self._ex("follow_path", wps, speed, self._yaw, 0.0, self._f, False)

    def idle(self):
        return self.beh is not None and self.beh.status == BehaviorStatus.IDLE


class CentralisedStage2:
    def __init__(self, namespaces, config, verbose=False, use_sim_time=True):
        self.n = len(namespaces)
        self.drones = [CentralDrone(n, verbose, use_sim_time) for n in namespaces]
        self.centre, self.windows = parse_stage2(config)
        self.alt = ALT_DEFAULT
        self.log = {}

    def _log(self, k, v):
        self.log[k] = round(float(v), 3)

    def _wait(self):
        time.sleep(1.5)
        while not all(d.idle() for d in self.drones):
            time.sleep(0.2)

    def ready(self):
        for d in self.drones: d.arm()
        time.sleep(0.5)
        for d in self.drones: d.offboard()
        time.sleep(0.5)

    def takeoff(self):
        for d in self.drones: d._ex("takeoff", self.alt, 0.7, False)
        self._wait()
        time.sleep(1.0)
        print('Airborne')

    def land(self):
        for d in self.drones: d._ex("land", 0.4, False)
        self._wait()

    def shutdown(self):
        for d in self.drones: d.shutdown()

    def _check_collisions(self, thresh=0.25):
        count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = float(np.linalg.norm(np.array(self.drones[i].position[:2]) -
                                         np.array(self.drones[j].position[:2])))
                if d < thresh: count += 1
        return count

    def _passed(self, w):
        return sum(1 for d in self.drones if d.position[1] < w['y'] - 0.2)

    def run(self):
        print('Run Stage 2 centralised')
        t0 = time.time()
        self._log("start", 0.0)
        recorder = TrajectoryRecorder(self.n)

        print('Planning paths')
        tp = time.time()
        A_paths, B_paths, C_paths = plan_all(self.centre, self.windows)
        mw, md = verify_segments(A_paths, B_paths, C_paths, self.windows)
        plan_time = time.time() - tp
        print(f'[PLAN] {plan_time:.2f}s, A={len(A_paths[0])} B={len(B_paths[0])} C={len(C_paths[0])} wpts')
        print(f'[SAFE] wall:{mw:.3f}m inter-drone:{md:.3f}m')
        self._log("plan_time", plan_time)
        self._log("plan_min_wall_clearance", mw)
        self._log("plan_min_drone_spacing", md)

        print('Moving to start')
        for i, d in enumerate(self.drones):
            p0 = A_paths[i][0]
            d.go(float(p0[0]), float(p0[1]), float(p0[2]), speed=0.8)
        self._wait()
        self._log("approach_time", time.time() - t0)

        w1 = self.windows[0]
        w2 = self.windows[1] if len(self.windows) > 1 else None
        collisions = 0

        recorder.start()

        # Segment A: approach through W1
        print('[SEG-A] approach through W1')
        tcA = time.time()
        for i, d in enumerate(self.drones):
            wps = [[float(p[0]), float(p[1]), float(p[2])] for p in A_paths[i]]
            d.follow(wps, speed=0.6)
        while not all(d.idle() for d in self.drones):
            collisions += self._check_collisions()
            recorder.sample(self.drones)
            time.sleep(0.1)
        self._log("segment_a_time", time.time() - tcA)
        self._log("window1_drones_passed", self._passed(w1))
        print(f'[W1] {self._passed(w1)}/{self.n}')

        if w2 is not None:
            # Segment B: between windows (diagonal x-shift, z-change)
            print('[SEG-B] between windows')
            tcB = time.time()
            for i, d in enumerate(self.drones):
                wps = [[float(p[0]), float(p[1]), float(p[2])] for p in B_paths[i]]
                d.follow(wps, speed=0.5)
            while not all(d.idle() for d in self.drones):
                collisions += self._check_collisions()
                recorder.sample(self.drones)
                time.sleep(0.1)
            self._log("segment_b_time", time.time() - tcB)

            # Segment C: through W2 to exit
            print('[SEG-C] through W2 to exit')
            tcC = time.time()
            for i, d in enumerate(self.drones):
                wps = [[float(p[0]), float(p[1]), float(p[2])] for p in C_paths[i]]
                d.follow(wps, speed=0.6)
            while not all(d.idle() for d in self.drones):
                collisions += self._check_collisions()
                recorder.sample(self.drones)
                time.sleep(0.1)
            self._log("segment_c_time", time.time() - tcC)
            self._log("window2_drones_passed", self._passed(w2))
            print(f'[W2] {self._passed(w2)}/{self.n}')

        self._log("total_collisions", collisions)
        w1p = self._passed(w1); w2p = self._passed(w2) if w2 else 0
        total_exp = self.n * (2 if w2 else 1)
        self._log("success_rate_pct", 100 * (w1p + (w2p if w2 else 0)) / total_exp)
        dur = time.time() - t0
        self._log("duration", dur)
        self._log("end", dur)
        print('---------------------------------')
        print(f'Centralised done: {dur:.1f}s, collisions {collisions}, W1 {w1p}/{self.n}'
              + (f', W2 {w2p}/{self.n}' if w2 else ''))
        print('---------------------------------')
        recorder.save("traj_stage2_centralised.csv")
        planned_flat = [np.vstack([A_paths[i], B_paths[i], C_paths[i]]) for i in range(self.n)]
        recorder.plot(self.windows, "Centralised MPC", "stage2_centralised",
                      centre=self.centre, planned_paths=planned_flat,
                      method_kind='centralised')


# Decentralised: Distributed MPC
# Each drone runs its own planner using the same cost function as the
# centralised approach, optimising only its own trajectory. No shared
# state. Coordination is implicit: because all drones read the same YAML
# and use the same V-formation definition, their paths are consistent
# without any communication.

def plan_own_path(drone_idx, centre, windows,
                  iters=400, lr=0.03, clearance=0.35, margin=0.3,
                  w_wall=120.0, w_form=1.5, w_smooth=3.0):
    """Plan one drone's trajectory independently (no state from other drones)."""
    A, B = wall_segments(windows)
    offset = V_OFFSETS[drone_idx]

    # Build shared deterministic centroid path from YAML
    # Every drone builds the same path because they read the same windows
    w1 = windows[0]
    w2 = windows[1] if len(windows) > 1 else None
    wps = [
        np.array([w1['x'], centre[1] + 4.0]),
        np.array([w1['x'], w1['y']]),
        np.array([w1['x'], w1['y'] - 1.5]),
    ]
    if w2 is not None:
        wps += [
            np.array([(w1['x'] + w2['x']) / 2, (w1['y'] + w2['y']) / 2 + 0.5]),
            np.array([w2['x'], w2['y']]),
            np.array([w2['x'], w2['y'] - 1.5]),
        ]
    wps.append(np.array([centre[0], centre[1] - 6.0]))

    # Resample by arclength to fixed number of points
    n = 120
    segs = [float(np.linalg.norm(wps[i+1] - wps[i])) for i in range(len(wps)-1)]
    total = sum(segs)
    CP = []
    for i in range(n):
        s = i / (n - 1) * total
        acc = 0.0
        for j, L in enumerate(segs):
            if acc + L >= s or j == len(segs) - 1:
                t = (s - acc) / L if L > 1e-6 else 0.0
                t = float(np.clip(t, 0, 1))
                CP.append(wps[j] + t * (wps[j+1] - wps[j]))
                break
            acc += L
    CP = np.array(CP)

    # Initial path: V-slot offset from centroid
    path = CP + offset
    path = soft_clamp_batch(path, path[:, 1], windows, margin=margin)

    # Gradient descent: minimise wall violation + formation deviation + smoothness
    for _ in range(iters):
        grad = np.zeros_like(path)
        d, C = seg_dist_batch(path, A, B)
        viol = np.maximum(clearance - d, 0.0)
        d_safe = np.maximum(d, 1e-6)
        direction = (path[:, None, :] - C) / d_safe[:, :, None]
        grad += w_wall * np.sum(-2 * viol[:, :, None] * direction, axis=1)

        vslots = CP + offset
        vslots = soft_clamp_batch(vslots, path[:, 1], windows, margin=margin)
        grad += w_form * 2 * (path - vslots)

        curv = np.zeros_like(path)
        curv[1:-1] = path[:-2] - 2 * path[1:-1] + path[2:]
        grad += w_smooth * (-2 * curv)

        grad[0] = 0; grad[-1] = 0
        step = lr * grad
        max_step = 0.015
        step_mag = np.linalg.norm(step, axis=1, keepdims=True)
        step = np.where(step_mag > max_step, step / np.maximum(step_mag, 1e-6) * max_step, step)
        path -= step

    # Laplacian smoothing with wall push-out
    for _ in range(15):
        new_path = path.copy()
        new_path[1:-1] = 0.7 * path[1:-1] + 0.15 * path[:-2] + 0.15 * path[2:]
        path = new_path
        d, C = seg_dist_batch(path, A, B)
        for i in range(1, len(path) - 1):
            for k in range(len(A)):
                if clearance - d[i, k] > 0:
                    dd = max(d[i, k], 1e-6)
                    path[i] += (clearance - d[i, k]) * (path[i] - C[i, k]) / dd

    # Enforce monotonic y (no backtracking)
    for i in range(1, len(path)):
        if path[i, 1] > path[i-1, 1]:
            path[i, 1] = path[i-1, 1]

    # Attach z from smooth altitude profile
    z = np.array([path_altitude_smooth(y, centre, windows) for y in path[:, 1]])
    return np.column_stack([path, z])


class DistributedDrone(DroneInterface):
    """Each drone runs its own planner and follows its own trajectory."""
    def __init__(self, ns, verbose=False, use_sim_time=True):
        super().__init__(ns, verbose=verbose, use_sim_time=use_sim_time)
        self.beh = None

    def go(self, x, y, z, speed=0.8):
        self.beh = self.go_to
        self.beh(float(x), float(y), float(z), speed,
                 YawMode.PATH_FACING, 0.0, "earth", wait=False)

    def follow(self, wps, speed=0.6):
        self.beh = self.follow_path
        self.beh(wps, speed, YawMode.PATH_FACING, 0.0, "earth", wait=False)

    def idle(self):
        return self.beh is not None and self.beh.status == BehaviorStatus.IDLE


class DecentralisedStage2:
    def __init__(self, namespaces, config, verbose=False, use_sim_time=True):
        self.n = len(namespaces)
        self.drones = [DistributedDrone(ns, verbose, use_sim_time) for ns in namespaces]
        self.centre, self.windows = parse_stage2(config)
        self.alt = ALT_DEFAULT
        self.log = {}

    def _log(self, k, v):
        self.log[k] = round(float(v), 3)

    def ready(self):
        ok = True
        for d in self.drones:
            ok = d.arm() and ok
        time.sleep(0.5)
        for d in self.drones:
            ok = d.offboard() and ok
        time.sleep(0.5)
        print(f'Ready: {ok}')

    def takeoff(self):
        for d in self.drones:
            d.takeoff(self.alt, 0.5, wait=False)
        self._wait()
        print('Airborne')

    def land(self):
        for d in self.drones:
            d.land(0.4, wait=False)
        self._wait()

    def shutdown(self):
        for d in self.drones: d.shutdown()

    def _wait(self):
        while True:
            if all(d.beh is None or d.beh.status == BehaviorStatus.IDLE
                   for d in self.drones):
                return
            time.sleep(0.3)

    def _check_collisions(self, thresh=0.25):
        count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = float(np.linalg.norm(np.array(self.drones[i].position[:2]) -
                                         np.array(self.drones[j].position[:2])))
                if d < thresh: count += 1
        return count

    def _passed(self, w):
        return sum(1 for d in self.drones if d.position[1] < w['y'] - 0.2)

    def run(self):
        print('Run Stage 2 decentralised')
        t0 = time.time()
        self._log("start", 0.0)
        recorder = TrajectoryRecorder(self.n)

        print('Planning paths (5 independent planners)')
        tp = time.time()
        paths = []
        for i in range(self.n):
            p = plan_own_path(i, self.centre, self.windows)
            paths.append(p)
        plan_time = time.time() - tp
        print(f'[PLAN] {plan_time:.2f}s total across {self.n} planners')

        # Offline safety check over the union of all five paths
        A, B = wall_segments(self.windows)
        per_drone_clearance = []
        for i, p in enumerate(paths):
            d, _ = seg_dist_batch(p[:, :2], A, B)
            per_drone_clearance.append(float(d.min()))
        min_wall = min(per_drone_clearance)
        min_drone = float('inf')
        for t in range(len(paths[0])):
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    dd = float(np.linalg.norm(paths[i][t, :2] - paths[j][t, :2]))
                    if dd < min_drone: min_drone = dd
        print(f'[SAFE] wall:{min_wall:.3f}m inter-drone:{min_drone:.3f}m')
        self._log("plan_time", plan_time)
        self._log("plan_min_wall_clearance", min_wall)
        self._log("plan_min_drone_spacing", min_drone)

        print('Moving to start')
        for i, d in enumerate(self.drones):
            p0 = paths[i][0]
            d.go(float(p0[0]), float(p0[1]), float(p0[2]), speed=0.8)
        self._wait()
        self._log("approach_time", time.time() - t0)

        w1 = self.windows[0]
        w2 = self.windows[1] if len(self.windows) > 1 else None
        collisions = 0

        recorder.start()

        print('[EXEC] each drone follows its own trajectory')
        tcF = time.time()
        for i, d in enumerate(self.drones):
            wps = [[float(p[0]), float(p[1]), float(p[2])] for p in paths[i]]
            d.follow(wps, speed=0.6)
        while not all(d.idle() for d in self.drones):
            collisions += self._check_collisions()
            recorder.sample(self.drones)
            time.sleep(0.1)
        self._log("execution_time", time.time() - tcF)
        self._log("window1_drones_passed", self._passed(w1))
        if w2: self._log("window2_drones_passed", self._passed(w2))
        print(f'[W1] {self._passed(w1)}/{self.n}'
              + (f'  [W2] {self._passed(w2)}/{self.n}' if w2 else ''))

        self._log("total_collisions", collisions)
        w1p = self._passed(w1); w2p = self._passed(w2) if w2 else 0
        total_exp = self.n * (2 if w2 else 1)
        success = 100 * (w1p + (w2p if w2 else 0)) / total_exp
        dur = time.time() - t0
        self._log("success_rate_pct", success)
        self._log("duration", dur)
        self._log("end", dur)
        print('---------------------------------')
        print(f'Decentralised done: {dur:.1f}s, collisions {collisions}, W1 {w1p}/{self.n}'
              + (f', W2 {w2p}/{self.n}' if w2 else ''))
        print('---------------------------------')
        recorder.save("traj_stage2_decentralised.csv")
        recorder.plot(self.windows, "Decentralised Distributed MPC",
                      "stage2_decentralised", method_kind='decentralised')


# Main

def ask(msg):
    return input(f"{msg}? (y/n): ").strip().lower() == 'y'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone1', 'drone2', 'drone3', 'drone4'])
    parser.add_argument('-c', '--config', type=str,
                        default='scenarios/scenario1_stage2.yaml')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True)
    parser.add_argument('--method', choices=['centralised', 'decentralised'],
                        default='centralised')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f'[CFG] method:{args.method}')
    print(f'[CFG] drones:{args.namespaces}')
    print(f'[CFG] config:{args.config}')

    rclpy.init()
    if args.method == 'centralised':
        ctrl = CentralisedStage2(args.namespaces, config, args.verbose, args.use_sim_time)
    else:
        ctrl = DecentralisedStage2(args.namespaces, config, args.verbose, args.use_sim_time)

    if ask("Takeoff"):
        ctrl.ready()
        ctrl.takeoff()
        if ask("Run Stage 2"):
            try:
                ctrl.run()
            except KeyboardInterrupt:
                print('Interrupted')

        logfile = f'log_stage2_{args.method}.txt'
        with open(logfile, 'w') as f:
            for k, v in ctrl.log.items():
                f.write(f'  {k}: {v}\n')
        print(f'Log saved: {logfile}')

        if ask("Land"):
            ctrl.land()

    ctrl.shutdown()
    rclpy.shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()
