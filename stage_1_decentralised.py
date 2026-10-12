#!/usr/bin/env python3

import argparse
import math
import sys
import threading
import time
from typing import List

import numpy as np
import yaml

import rclpy
from as2_msgs.msg import YawMode
from as2_python_api.drone_interface import DroneInterface


# potential field parameters

PF_ATTRACTION    = 4.0     
PF_REPULSION     = 0.4     
PF_SAFE_DIST     = 0.27   
PF_MAX_SPEED     = 3.0     
PF_TANGENT_FORCE = 0.1     
PF_HZ            = 10.0    
PF_DT            = 1.0 / PF_HZ

STRATIFICATION_HEIGHTS = [0.0, 0.15, 0.3, 0.45, 0.6]

# formations

FORMATIONS_5 = {
    "line": [
        [0.0,  1.0, 0.0],
        [0.0,  0.5, 0.0],
        [0.0,  0.0, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, -1.0, 0.0],
    ],
    "v": [
        [-0.8,  1.0, 0.0],
        [ 0.0,  0.5, 0.0],
        [ 0.8,  0.0, 0.0],
        [ 0.0, -0.5, 0.0],
        [-0.8, -1.0, 0.0],
    ],
    "square": [
        [ 0.5,  0.5, 0.0],
        [-0.5,  0.5, 0.0],
        [ 0.0,  0.0, 0.0],
        [0.5, -0.5, 0.0],
        [ -0.5, -0.5, 0.0],
    ],
    "diamond": [
        [ 0.0,  0.7, 0.0],
        [ 0.7,  0.0, 0.0],
        [ 0.0,  0.0, 0.0],
        [-0.7,  0.0, 0.0],
        [ 0.0, -0.7, 0.0],
    ],
    "orbit": [
        [0.7 * math.cos(2 * math.pi * i / 5),
         0.7 * math.sin(2 * math.pi * i / 5), 0.0]
        for i in range(5)
    ],
    "grid": [
        [ 0.5,  0.5, 0.0],
        [-0.5,  0.5, 0.0],
        [ 0.0,  0.0, 0.0],
        [-0.5, -0.5, 0.0],
        [ 0.5, -0.5, 0.0],
    ],
    "staggered": [
        [-0.4,  0.8, 0.0],
        [ 0.4,  0.4, 0.0],
        [-0.4,  0.0, 0.0],
        [ 0.4, -0.4, 0.0],
        [-0.4, -0.8, 0.0],
    ]
}

def orbit_target(t, offset_vec, orbit_radius, omega, altitude, scale):
    """World-frame target for one drone at orbit-time t."""
    cx = orbit_radius * np.cos(omega * t)
    cy = orbit_radius * np.sin(omega * t)
    center = np.array([cx, cy, altitude])

    yaw = omega * t + (np.pi / 2)
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1],
    ])
    return center + R @ (np.array(offset_vec) * scale)


def orbit_tangent(t, omega):
    """Unit tangent vector along the circular path at time t."""
    theta = omega * t
    return np.array([-np.sin(theta), np.cos(theta), 0.0])


def smooth_blend(tau):
    """Quintic smooth-step: zero 1st and 2nd derivatives at endpoints."""
    tau = max(0.0, min(1.0, tau))
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


# drone wrapper

class Agent:
    def __init__(self, namespace):
        self.ns = namespace
        self.drone = DroneInterface(namespace, use_sim_time=True)

    @property
    def pos(self):
        return np.array(self.drone.position)

    def command(self, point):
        self.drone.go_to(
            float(point[0]), float(point[1]), float(point[2]),
            PF_MAX_SPEED, YawMode.PATH_FACING, None, "earth", wait=False)


# swarm controller

class PFSwarm:

    def __init__(self, drone_names, stage_cfg, args):
        self.agents = [Agent(ns) for ns in drone_names]
        self.n = len(drone_names)

        traj_cfg = stage_cfg.get("trajectory", {})
        self.orbit_radius = traj_cfg.get("diameter", 3.0) / 2.0

        raw = stage_cfg.get("formations", ["line", "v", "diamond"])
        self.sequence = [f for f in raw if f in FORMATIONS_5] or ["line"]
        self.formations = {name: FORMATIONS_5[name] for name in self.sequence}
        self.num_formations = len(self.sequence)

        self.omega = args.omega
        self.altitude = args.altitude
        self.scale = args.scale
        self.num_loops = args.loops

        # circle segment timing
        self.loop_time = 2 * math.pi / self.omega
        self.segment_time = self.loop_time / self.num_formations
        self.morph_time = self.segment_time * (2.0 / 3.0)
        self.hold_time = self.segment_time * (1.0 / 3.0)

        self.global_t = 0.0
        self.keep_flying = True

        centroid_speed = self.omega * self.orbit_radius
        ticks_per_seg = int(self.segment_time / PF_DT)
        print(f"Orbit radius: {self.orbit_radius:.1f}m  "
              f"omega={self.omega:.2f} rad/s  "
              f"centroid speed={centroid_speed:.2f} m/s")
        print(f"Formations ({self.num_formations}): {self.sequence}")
        print(f"Loop: {self.loop_time:.1f}s  "
              f"Segment: {self.segment_time:.1f}s "
              f"({ticks_per_seg} ticks)  "
              f"morph {self.morph_time:.1f}s + hold {self.hold_time:.1f}s")
        print(f"Loops: {self.num_loops}")

    # helpers
    def max_err(self, targets):
        return max(np.linalg.norm(self.agents[i].pos - targets[i])
                   for i in range(self.n))

    def min_separation(self):
        min_d = float('inf')
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = np.linalg.norm(self.agents[i].pos - self.agents[j].pos)
                min_d = min(min_d, d)
        return min_d

    def converged(self, targets, thr=0.15):
        return all(
            np.linalg.norm(self.agents[i].pos - targets[i]) < thr
            for i in range(self.n))

    # potential field core methods

    def _compute_forces(self, agent_idx, target):
        agent = self.agents[agent_idx]
        pos = agent.pos

        # Attraction to target
        F_att = PF_ATTRACTION * (target - pos)

        # Repulsion from other drones
        F_rep = np.zeros(3)
        for j, other in enumerate(self.agents):
            if j == agent_idx:
                continue
            gap = pos - other.pos
            dist = np.linalg.norm(gap)
            if 1e-5 < dist < PF_SAFE_DIST:
                mag = PF_REPULSION * (1.0/dist - 1.0/PF_SAFE_DIST) / (dist**3)
                # Emergency multiplier at critical range
                if dist < PF_SAFE_DIST * 0.3:
                    mag *= 5.0
                F_rep += mag * gap

        return F_att + F_rep

    def _pf_step(self, targets, feedforward=None):
        """
        One PF tick.
        Tangent force is always applied for baseline orbit tracking.
        Feedforward velocity is added on top when available.
        """
        tangent = orbit_tangent(self.global_t, self.omega) * PF_TANGENT_FORCE

        for i, agent in enumerate(self.agents):
            F = self._compute_forces(i, targets[i])
            F += tangent

            if feedforward is not None:
                F += feedforward[i]

            # Speed cap
            mag = np.linalg.norm(F)
            if mag > PF_MAX_SPEED:
                F = F / mag * PF_MAX_SPEED

            nxt = agent.pos + F * PF_DT
            nxt[2] = targets[i][2]
            agent.command(nxt)

    # lifecycle

    def get_ready(self):
        ok = True
        for a in self.agents:
            ok = a.drone.arm() and ok
        time.sleep(1.0)
        for a in self.agents:
            ok = a.drone.offboard() and ok
        time.sleep(1.0)
        print(f"Ready: {ok}")
        return ok

    def takeoff(self):
        for a in self.agents:
            a.drone.takeoff(self.altitude, 0.5, wait=False)
        time.sleep(4.0)
        print("Airborne.")

    def land(self):
        for a in self.agents:
            a.drone.land(0.4, wait=False)
        time.sleep(3.0)

    def shutdown(self):
        for a in self.agents:
            a.drone.shutdown()

    # deploy with altitude stratification

    def deploy_to_first(self, formation_id):
        """Safe initial deployment: rise to stratified heights, converge, descend."""
        offsets = self.formations[formation_id]
        print(f"Deploying to [{formation_id}]...")

        # Rise to different heights
        print("  Rising to safe altitudes...")
        for i, agent in enumerate(self.agents):
            safe_z = self.altitude + STRATIFICATION_HEIGHTS[i]
            agent.command([agent.pos[0], agent.pos[1], safe_z])
        time.sleep(3.0)

        # converge to formation at different heights
        print("  Converging to slots...")
        for _ in range(int(12.0 / PF_DT)):
            targets = []
            for i in range(self.n):
                pos = orbit_target(0.0, offsets[i], self.orbit_radius,
                                   self.omega, self.altitude, self.scale)
                safe_z = self.altitude + STRATIFICATION_HEIGHTS[i]
                targets.append(np.array([pos[0], pos[1], safe_z]))
            self._pf_step(targets)
            if self.converged(targets, 0.2):
                print(f"  Converged. err={self.max_err(targets):.2f}m")
                break
            time.sleep(PF_DT)

        #  descent to common altitude
        print("  Descending...")
        for _ in range(int(6.0 / PF_DT)):
            targets = []
            for i in range(self.n):
                targets.append(orbit_target(
                    0.0, offsets[i], self.orbit_radius,
                    self.omega, self.altitude, self.scale))
            self._pf_step(targets)
            if self.converged(targets, 0.15):
                break
            time.sleep(PF_DT)

        print(f"  Deployment complete. err={self.max_err(targets):.2f}m  "
              f"sep={self.min_separation():.2f}m")

    # hold formation while orbiting

    def hold(self, formation_id, duration):
        offsets = self.formations[formation_id]
        start_t = self.global_t
        prev_targets = None
        last_log = time.time()

        while (self.global_t - start_t < duration) and self.keep_flying:
            targets = [orbit_target(self.global_t, offsets[i],
                                    self.orbit_radius, self.omega,
                                    self.altitude, self.scale)
                       for i in range(self.n)]

            ff = None
            if prev_targets is not None:
                ff = [(targets[i] - prev_targets[i]) / PF_DT
                      for i in range(self.n)]

            self._pf_step(targets, feedforward=ff)
            prev_targets = [t.copy() for t in targets]
            self.global_t += PF_DT
            time.sleep(PF_DT)

            # Diagnostic every 3s wall time
            now = time.time()
            if now - last_log >= 3.0:
                angle = math.degrees(self.omega * self.global_t) % 360
                print(f"t={self.global_t:.1f}s  angle={angle:.0f}°  "
                      f"err={self.max_err(targets):.3f}m  "
                      f"sep={self.min_separation():.3f}m")
                last_log = now

    # morph between formations

    def morph(self, old_id, new_id, duration):
        old_off = self.formations[old_id]
        new_off = self.formations[new_id]
        start_t = self.global_t
        prev_targets = None
        last_log = time.time()

        while (self.global_t - start_t < duration) and self.keep_flying:
            elapsed = self.global_t - start_t
            tau = elapsed / duration
            blend = smooth_blend(tau)
            # Height arc during transition (stratification for safety)
            z_arc = 4 * tau * (1 - tau)

            targets = []
            for i in range(self.n):
                off_old = np.array(old_off[i])
                off_new = np.array(new_off[i])
                blended = off_old + (off_new - off_old) * blend
                blended[2] += STRATIFICATION_HEIGHTS[i] * z_arc

                targets.append(orbit_target(
                    self.global_t, blended.tolist(), self.orbit_radius,
                    self.omega, self.altitude, self.scale))

            ff = None
            if prev_targets is not None:
                ff = [(targets[i] - prev_targets[i]) / PF_DT
                      for i in range(self.n)]

            self._pf_step(targets, feedforward=ff)
            prev_targets = [t.copy() for t in targets]
            self.global_t += PF_DT
            time.sleep(PF_DT)

            # Diagnostic every 3s wall time
            now = time.time()
            if now - last_log >= 3.0:
                angle = math.degrees(self.omega * self.global_t) % 360
                print(f"    t={self.global_t:.1f}s  angle={angle:.0f}°  "
                      f"blend={blend:.0%}  "
                      f"err={self.max_err(targets):.3f}m  "
                      f"sep={self.min_separation():.3f}m")
                last_log = now

    def execute(self):
        """
        Circle-segment formation circuit.
        The orbit keeps moving continuously throughout — morphing and
        holding happen while the centroid travels around the circle.
        """
        t0 = time.time()
        total_segments = self.num_formations * self.num_loops
        current = self.sequence[0]
        first_segment = True
        seg_counter = 0

        for loop in range(self.num_loops):
            for idx, formation in enumerate(self.sequence):
                if not self.keep_flying:
                    return
                seg_counter += 1

                if first_segment:
                    print(f"Seg {seg_counter:2d}/{total_segments}  "
                          f"Loop {loop+1}  "
                          f"{formation:10s}  HOLD   "
                          f"({self.segment_time:.1f}s orbit-time)")
                    self.hold(formation, self.segment_time)
                    first_segment = False
                else:
                    print(f"Seg {seg_counter:2d}/{total_segments}  "
                          f"Loop {loop+1}  "
                          f"{current:10s} -> {formation:10s}  MORPH  "
                          f"({self.morph_time:.1f}s orbit-time)")
                    self.morph(current, formation, self.morph_time)

                    if not self.keep_flying:
                        return

                    print(f"Seg {seg_counter:2d}/{total_segments}  "
                          f"Loop {loop+1}  "
                          f"{formation:10s}  HOLD   "
                          f"({self.hold_time:.1f}s orbit-time)")
                    self.hold(formation, self.hold_time)

                current = formation

        angle = math.degrees(self.omega * self.global_t) % 360
        print(f"Circuit complete. {self.num_loops} loop(s), "
              f"orbit time: {self.global_t:.1f}s, "
              f"final angle: {angle:.0f}°")
        sim_elapsed = time.time() - t0
        print(f"Traversal wall time: {sim_elapsed:.1f}s")

def stop_listener(swarm):
    input("Press enter to land")
    swarm.keep_flying = False

def main():
    parser = argparse.ArgumentParser(
        description='Decentralised PF Formation Flight')
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone1', 'drone2',
                                 'drone3', 'drone4'])
    parser.add_argument('-c', '--config', type=str,
                        default='scenarios/scenario1_stage1.yaml')
    parser.add_argument('-s', '--use_sim_time', action='store_true',
                        default=True)
    parser.add_argument('--omega', type=float, default=0.3,
                        help='Angular velocity rad/s (lower = more time per '
                             'formation, default 0.3)')
    parser.add_argument('--altitude', type=float, default=1.5,
                        help='Flight altitude (m)')
    parser.add_argument('--scale', type=float, default=0.7,
                        help='Formation scale factor')
    parser.add_argument('--loops', type=int, default=1,
                        help='Number of full circular loops')
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)

    stage_cfg = config.get("stage1", {})
    rclpy.init()
    swarm = PFSwarm(args.namespaces, stage_cfg, args)

    print(f"Fleet: {args.namespaces}")
    if input("Takeoff? (y/n): ").strip().lower() != 'y':
        rclpy.shutdown()
        sys.exit(0)

    swarm.get_ready()
    swarm.takeoff()
    swarm.deploy_to_first(swarm.sequence[0])

    threading.Thread(target=stop_listener, args=(swarm,),
                     daemon=True).start()

    swarm.execute()

    print("\n>>> Landing...")
    swarm.land()
    swarm.shutdown()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()