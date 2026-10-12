#!/usr/bin/env python3

import argparse
import math
import sys
import time
from typing import List, Optional, Tuple

import rclpy
import yaml
from as2_msgs.msg import BehaviorStatus, YawMode
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler


# Formations

FORMATIONS_5 = {
    "line": [
        ( 0.0,  1.0),
        ( 0.0,  0.5),
        ( 0.0,  0.0),
        ( 0.0, -0.5),
        ( 0.0, -1.0),
    ],
    "v": [
        (-0.8,  1.0),
        ( 0.0,  0.5),
        ( 0.8,  0.0),
        ( 0.0, -0.5),
        (-0.8, -1.0),
    ],
    "square": [
        ( 0.5,  0.5),
        (-0.5,  0.5),
        ( 0.0,  0.0),
        (0.5, -0.5),
        ( -0.5, -0.5),
    ],
    "diamond": [
        ( 0.0,  0.7),
        ( 0.7,  0.0),
        ( 0.0,  0.0),
        (-0.7,  0.0),
        ( 0.0, -0.7),
    ],
    "orbit": [
        (0.7 * math.cos(2 * math.pi * i / 5),
         0.7 * math.sin(2 * math.pi * i / 5))
        for i in range(5)
    ],
    "grid": [
        ( 0.5,  0.5),
        (-0.5,  0.5),
        ( 0.0,  0.0),
        (-0.5, -0.5),
        ( 0.5, -0.5),
    ],
    "staggered": [
        (-0.4,  0.8),
        ( 0.4,  0.4),
        (-0.4,  0.0),
        ( 0.4, -0.4),
        (-0.4, -0.8),
    ],
}


def get_formation_offsets(name: str, num_drones: int) -> List[Tuple[float, float]]:
    return FORMATIONS_5[name]


# math helpers

def dist2d(a, b) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def rotate_offset(dx: float, dy: float, heading: float) -> Tuple[float, float]:
    c, s = math.cos(heading), math.sin(heading)
    return (dx * c - dy * s, dx * s + dy * c)


def smooth_ease(t: float) -> float:
    """Smooth-step easing: zero derivative at both ends."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def circular_trajectory(center: Tuple[float, float],
                        radius: float,
                        num_points: int) -> List[Tuple[float, float, float]]:
    """Evenly-spaced waypoints around a circle: (x, y, heading_rad)."""
    points = []
    for i in range(num_points):
        angle = 2.0 * math.pi * i / num_points
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        heading = angle + math.pi / 2.0
        points.append((x, y, heading))
    return points


def interpolate_offsets(offsets_a: List[Tuple[float, float]],
                        offsets_b: List[Tuple[float, float]],
                        t: float) -> List[Tuple[float, float]]:
    """Smooth-eased interpolation between two formation offset sets."""
    t = smooth_ease(t)
    return [(ax + t * (bx - ax), ay + t * (by - ay))
            for (ax, ay), (bx, by) in zip(offsets_a, offsets_b)]


def enforce_separation(offsets: List[Tuple[float, float]],
                       min_dist: float = 0.45) -> List[Tuple[float, float]]:
    """Push apart any pair of offsets closer than min_dist."""
    pts = [list(o) for o in offsets]
    for _ in range(10):
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < min_dist and d > 1e-6:
                    push = (min_dist - d) / 2.0
                    nx, ny = dx / d, dy / d
                    pts[i][0] += nx * push
                    pts[i][1] += ny * push
                    pts[j][0] -= nx * push
                    pts[j][1] -= ny * push
    return [tuple(p) for p in pts]


# altitude separation, only enabled if necessary

def compute_altitude_separation(
        paths: List[List[Tuple[float, float]]],
        baselines: List[List[Tuple[float, float]]],
        num_steps: int,
        num_drones: int,
        base_height: float,
        proximity_threshold: float = 0.4,
        deviation_threshold: float = 0.3,
        alt_offset: float = 0.3,
        ramp_steps: int = 2,
) -> List[List[float]]:
    """
    Per-step altitude offsets.  When two drones are closer than
    proximity_threshold the one deviating more from its baseline
    shifts up.  Smoothly ramped at zone edges.
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

    # Smooth ramp within each zone
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
                edge_dist = min(s - zs, ze - s)
                ramp_t = min(1.0, (edge_dist + 1) / (ramp_steps + 1)) \
                    if ramp_steps > 0 else 1.0
                ramp_t = ramp_t * ramp_t * (3.0 - 2.0 * ramp_t)
                altitudes[i][s] = base_height + alt_offset * ramp_t

    total_shifted = sum(1 for i in range(num_drones)
                        for s in range(num_steps)
                        if altitudes[i][s] != base_height)
    if total_shifted:
        affected = [i for i in range(num_drones)
                    if any(altitudes[i][s] != base_height
                           for s in range(num_steps))]
        print(f"Altitude separation: {total_shifted} step(s) shifted "
              f"for drone(s) {affected}")
    else:
        print("No altitude separation needed")

    return altitudes


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


# virtual structure controller

class VirtualStructureController:
    """
    Central controller that moves a centroid around a circular trajectory
    while switching between formation shapes.
    """

    def __init__(self, drones_ns: List[str], config: dict,
                 verbose: bool = False, use_sim_time: bool = False):

        self.num_drones = len(drones_ns)
        self.drones: List[FormationDrone] = []
        for ns in drones_ns:
            self.drones.append(
                FormationDrone(ns, verbose=verbose, use_sim_time=use_sim_time))

        # configs
        stage_cfg = config.get("stage1", {})
        traj_cfg = stage_cfg.get("trajectory", {})
        self.center = tuple(stage_cfg.get("stage_center", [0.0, 0.0]))
        self.radius = traj_cfg.get("diameter", 3.0) / 2.0
        self.flight_height = 1.0  

        # Speed bounds 
        self.base_speed = 0.7
        self.max_speed = 2.5
        self.min_speed = 0.2

        # Formation schedule
        raw_formations = stage_cfg.get("formations", ["line", "v", "diamond"])
        self.formation_schedule = [
            f for f in raw_formations if f in FORMATIONS_5
        ] or ["line"]

        # timing
        self.transition_steps = 8

        min_steps_per_formation = self.transition_steps + 4
        min_total = len(self.formation_schedule) * min_steps_per_formation
        self.points_per_loop = max(36, min_total)
        self.switch_every = self.points_per_loop // len(self.formation_schedule)

        self.trajectory = circular_trajectory(
            self.center, self.radius, self.points_per_loop)

        print(f"Formations: {self.formation_schedule}")
        print(f"{self.points_per_loop} waypoints/loop, "
              f"switch every {self.switch_every} steps, "
              f"blend over {self.transition_steps} steps, "
              f"hold for {self.switch_every - self.transition_steps} steps")

    # lifecycle

    def get_ready(self) -> bool:
        ok = True
        print("Arming all drones...")
        for d in self.drones:
            ok = d.arm() and ok
        time.sleep(1.0)
        print("Setting offboard mode...")
        for d in self.drones:
            ok = d.offboard() and ok
        time.sleep(1.0)
        print(f"All drones ready: {ok}")
        return ok

    def takeoff(self):
        print("Taking off...")
        for d in self.drones:
            d.do_behavior("takeoff", self.flight_height, 0.7, False)
        self._wait_all()
        time.sleep(0.5)
        print("All drones airborne")

    def land(self):
        for d in self.drones:
            d.do_behavior("land", 0.4, False)
        self._wait_all()

    def shutdown(self):
        for d in self.drones:
            d.shutdown()

    # path precomputation

    def _precompute_loop(self, num_loops: int = 1):
        """
        Pre-compute all waypoints for every drone across the full circuit.
        """
        total_steps = self.points_per_loop * num_loops
        formation_idx = 0
        current_offsets = get_formation_offsets(
            self.formation_schedule[0], self.num_drones)

        paths = [[] for _ in range(self.num_drones)]
        baselines = [[] for _ in range(self.num_drones)]

        # Track per-step phase for building segment boundaries
        step_formation = []   # (formation_name, phase) per step
        is_first_period = True

        for step in range(total_steps):
            steps_since_switch = step % self.switch_every

            # Advance formation at switch boundaries
            if step > 0 and steps_since_switch == 0:
                current_offsets = get_formation_offsets(
                    self.formation_schedule[formation_idx], self.num_drones)
                formation_idx = (formation_idx + 1) % len(self.formation_schedule)
                is_first_period = False

            target_offsets = get_formation_offsets(
                self.formation_schedule[formation_idx], self.num_drones)

            traj_idx = step % self.points_per_loop
            cx, cy, heading = self.trajectory[traj_idx]

            # First period: no previous formation to blend from, so pure hold.
            # Subsequent periods: blend for transition_steps, then hold.
            blending = (not is_first_period
                        and steps_since_switch < self.transition_steps
                        and self.transition_steps > 1)

            if blending:
                t = steps_since_switch / (self.transition_steps - 1)
                t = min(t, 1.0)
                offsets = interpolate_offsets(current_offsets, target_offsets, t)
                offsets = enforce_separation(offsets)
                phase = "BLEND"
            else:
                offsets = target_offsets
                phase = "HOLD"

            step_formation.append(
                (self.formation_schedule[formation_idx], phase))

            for i in range(self.num_drones):
                dx, dy = offsets[i]
                rx, ry = rotate_offset(dx, dy, heading)
                paths[i].append((cx + rx, cy + ry))

                bdx, bdy = target_offsets[i]
                brx, bry = rotate_offset(bdx, bdy, heading)
                baselines[i].append((cx + brx, cy + bry))

        # Build segment list: runs of identical (name, phase)
        segments = []
        seg_start = 0
        for s in range(1, total_steps):
            if step_formation[s] != step_formation[s - 1]:
                name, phase = step_formation[seg_start]
                segments.append((seg_start, s, name, phase))
                seg_start = s
        name, phase = step_formation[seg_start]
        segments.append((seg_start, total_steps, name, phase))

        # Altitude separation
        altitudes = compute_altitude_separation(
            paths, baselines, total_steps, self.num_drones,
            base_height=self.flight_height,
            proximity_threshold=0.4,
            deviation_threshold=0.2,
            alt_offset=0.3,
            ramp_steps=2,
        )

        return paths, baselines, altitudes, segments

    # speed synchronization

    def _sync_speeds(self,
                     paths: List[List[Tuple[float, float]]],
                     baselines: List[List[Tuple[float, float]]],
                     s0: int, s1: int) -> List[float]:
        """
        Compute per-drone speed for a chunk [s0, s1) so that every
        drone finishes in the same wall-clock time.
        """
        actual_dists = []
        baseline_dists = []
        for i in range(self.num_drones):
            ad = sum(dist2d(paths[i][s], paths[i][s + 1])
                     for s in range(s0, s1 - 1))
            bd = sum(dist2d(baselines[i][s], baselines[i][s + 1])
                     for s in range(s0, s1 - 1))
            actual_dists.append(ad)
            baseline_dists.append(bd)

        sorted_bd = sorted(baseline_dists)
        ref_dist = sorted_bd[len(sorted_bd) // 2]
        if ref_dist < 1e-6:
            ref_dist = max(baseline_dists) if max(baseline_dists) > 1e-6 else 1.0
        ref_time = ref_dist / self.base_speed

        speeds = []
        for i in range(self.num_drones):
            if ref_time > 1e-6 and actual_dists[i] > 1e-6:
                sp = actual_dists[i] / ref_time
            else:
                sp = self.base_speed
            speeds.append(max(self.min_speed, min(sp, self.max_speed)))

        return speeds

    def execute_circuit(self, num_loops: int = 1):
        """
        Pre-compute the full circuit, then dispatch per-segment
        (each blend or hold phase) with synchronised speeds.
        """
        print("Pre-computing paths...")
        t0 = time.time()
        paths, baselines, altitudes, segments = self._precompute_loop(num_loops)
        total_steps = len(paths[0])
        print(f"Planning took {time.time() - t0:.2f}s  "
              f"({total_steps} steps, {len(segments)} segments)")

        for seg_idx, (s0, s1, name, phase) in enumerate(segments):
            if s1 - s0 < 2:
                continue

            speeds = self._sync_speeds(paths, baselines, s0, s1)

            for i, drone in enumerate(self.drones):
                wps = [[paths[i][s][0], paths[i][s][1], altitudes[i][s]]
                       for s in range(s0, s1)]
                drone.send_path(wps, speeds[i])

            self._wait_all()

            sp_str = " ".join(f"d{i}:{speeds[i]:.2f}"
                              for i in range(self.num_drones))
            print(f"Seg {seg_idx+1:2d}/{len(segments)}  "
                  f"steps {s0+1:3d}–{s1:3d}/{total_steps}  "
                  f"{name:10s} {phase:5s}  "
                  f"[{sp_str}]")
        
        sim_elapsed = time.time() - t0
        print(f"Traversal wall time: {sim_elapsed:.1f}s")

    def _wait_all(self):
        while True:
            if all(d.goal_reached() for d in self.drones):
                break

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def confirm(msg: str = 'Continue') -> bool:
    return input(f"{msg}? (y/n): ").strip().lower() == 'y'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone1', 'drone2',
                                 'drone3', 'drone4'])
    parser.add_argument('-c', '--config', type=str,
                        default='scenarios/scenario1.yaml')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-s', '--use_sim_time', action='store_true',
                        default=True)
    parser.add_argument('--loops', type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    rclpy.init()

    controller = VirtualStructureController(
        drones_ns=args.namespaces,
        config=config,
        verbose=args.verbose,
        use_sim_time=args.use_sim_time,
    )

    if confirm("Arm & Takeoff"):
        controller.get_ready()
        controller.takeoff()

        if confirm("Start formation circuit"):
            controller.execute_circuit(num_loops=args.loops)

        confirm("Land")
        controller.land()

    print("Shutting down")
    controller.shutdown()
    rclpy.shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()


"""
./launch_as2.bash -s scenarios/scenario1_stage1.yaml -w config_sim/config/world_swarm.yaml

./launch_ground_station.bash -n drone0,drone1,drone2,drone3,drone4

python3 stage_1_central.py -n drone0 drone1 drone2 drone3 drone4 \
    -c scenarios/scenario1_stage1.yaml
"""