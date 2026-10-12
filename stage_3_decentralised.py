#!/usr/bin/env python3

"""
Decentralised potential fields forest traversal.
Drones use potential fields to navigate obstacles while maintaining formation.
"""

import argparse
import sys
import time
import math
import numpy as np
import yaml

import rclpy
from as2_msgs.msg import YawMode
from as2_python_api.drone_interface import DroneInterface

# potential field tuning

ATTRACTION = 5.0
OBS_REPULSION = 0.5
DRONE_REPULSION = 0.4
DRONE_SAFE = 0.15
MAX_SPEED = 2.5
HZ = 10.0
DT = 1.0 / HZ

# formations to try out

FORMATIONS = {
    "line":    [(0.0, 1.0),(0.0, 0.5),(0.0, 0.0),(0.0,-0.5),(0.0,-1.0)],
    "v":       [(-0.8,1.0),(0.0,0.5),(0.8,0.0),(0.0,-0.5),(-0.8,-1.0)],
    "column":  [(0.0,0.0),(0.0,-0.6),(0.0,-1.2),(0.0,-1.8),(0.0,-2.4)],
    "diamond": [(0.0,0.7),(0.7,0.0),(0.0,0.0),(-0.7,0.0),(0.0,-0.7)],
}


def rotate(dx, dy, h):
    c, s = math.cos(h), math.sin(h)
    return dx*c - dy*s, dx*s + dy*c


class Drone:
    def __init__(self, ns):
        self.iface = DroneInterface(ns, use_sim_time=True)

    @property
    def pos(self):
        return np.array(self.iface.position)

    def cmd(self, pt):
        self.iface.go_to(float(pt[0]), float(pt[1]), float(pt[2]),
                         MAX_SPEED, YawMode.PATH_FACING, 0.0, "earth",
                         wait=False)


class Swarm:
    def __init__(self, names, config, args):
        self.drones = [Drone(ns) for ns in names]
        self.n = len(names)
        self.log = [] 

        cfg = config.get("stage3", {})
        sc = cfg.get("stage_center", [0.0, -6.0])
        sr = cfg.get("start_point", [-4.0, 0.0])
        er = cfg.get("end_point", [4.0, 0.0])

        self.start = np.array([sc[0]+sr[0], sc[1]+sr[1]])
        self.end   = np.array([sc[0]+er[0], sc[1]+er[1]])

        obs_d = cfg.get("obstacle_diameter", 0.4)
        self.obs_r = obs_d / 2.0
        self.clearance = self.obs_r + args.safety_margin
        self.obstacles = [np.array([sc[0]+o[0], sc[1]+o[1]])
                          for o in cfg.get("obstacles", [])]

        form = cfg.get("formation", "line")
        self.offsets = FORMATIONS.get(form, FORMATIONS["line"])

        diff = self.end - self.start
        self.heading = math.atan2(diff[1], diff[0])
        self.dist = np.linalg.norm(diff)
        self.direction = diff / self.dist

        self.alt = args.altitude
        self.speed = args.speed

        print(f"Start: {self.start}  End: {self.end}")
        print(f"Obstacles: {len(self.obstacles)}  Clearance: {self.clearance:.2f}m")
        print(f"Traverse: {self.dist:.1f}m at {self.speed} m/s")

    def target(self, i, centroid):
        dx, dy = self.offsets[i]
        rx, ry = rotate(dx, dy, self.heading)
        return np.array([centroid[0]+rx, centroid[1]+ry, self.alt])

    def forces(self, i, tgt):
        pos = self.drones[i].pos[:2]
        F = ATTRACTION * (tgt[:2] - pos)

        for obs in self.obstacles:
            gap = pos - obs
            d = np.linalg.norm(gap)
            if d < 1e-5:
                continue
            if d < self.clearance:
                radial = gap / d
    
                # Tangential: rotate radial 90° in the direction that
                # aligns with forward travel (so drone sweeps around)
                tangent = np.array([-radial[1], radial[0]]) 
                if np.dot(tangent, self.direction) < 0:
                    tangent = -tangent  # flip to align with travel
                
                mag = OBS_REPULSION / (d*d)
                if d < self.obs_r + 0.1:
                    mag *= 5.0
                
                # Blend: mostly tangential near boundary, more radial close to trunk
                radial_weight = (self.clearance - d) / self.clearance  # 0 at edge, 1 at trunk
                F += mag * (radial_weight * radial + (1.0 - radial_weight) * tangent)
            elif d < self.clearance * 1.5:
                fall = (self.clearance*1.5 - d) / (self.clearance*0.5)
                F += OBS_REPULSION * 0.3 * fall * gap / d

        for j, other in enumerate(self.drones):
            if j == i:
                continue
            gap = pos - other.pos[:2]
            d = np.linalg.norm(gap)
            if 1e-5 < d < DRONE_SAFE:
                F += DRONE_REPULSION * (1/d - 1/DRONE_SAFE) / (d*d) * gap/d

        return F

    def step(self, targets, ff=None):
        t0 = time.time()
        for i, dr in enumerate(self.drones):
            F = self.forces(i, targets[i])
            if ff is not None:
                F += ff[:2]
            mag = np.linalg.norm(F)
            if mag > MAX_SPEED:
                F = F / mag * MAX_SPEED
            nxt = dr.pos.copy()
            nxt[0] += F[0] * DT
            nxt[1] += F[1] * DT
            nxt[2] = targets[i][2]
            dr.cmd(nxt)
        # Adaptive sleep
        left = DT - (time.time() - t0)
        if left > 0.001:
            time.sleep(left)

    def converged(self, targets, thr=0.2):
        return all(np.linalg.norm(self.drones[i].pos[:2] - targets[i][:2]) < thr
                   for i in range(self.n))

    def max_err(self, targets):
        return max(np.linalg.norm(self.drones[i].pos[:2] - targets[i][:2])
                   for i in range(self.n))

    # mission phases

    def arm_and_takeoff(self):
        for d in self.drones:
            d.iface.arm()
        time.sleep(0.5)
        for d in self.drones:
            d.iface.offboard()
        time.sleep(0.5)
        for d in self.drones:
            d.iface.takeoff(self.alt, 0.7, wait=False)
        # Wait until airborne
        for _ in range(int(6.0 / DT)):
            if all(abs(d.pos[2] - self.alt) < 0.2 for d in self.drones):
                break
            time.sleep(DT)
        print("Airborne.")

    def go_to_start(self):
        print("Moving to start...")
        for _ in range(int(25.0 / DT)):
            tgts = [self.target(i, self.start) for i in range(self.n)]
            self.step(tgts)
            if self.converged(tgts, 0.25):
                break
        print(f"At start. err={self.max_err(tgts):.2f}m")

    def traverse(self):
        traverse_time = self.dist / self.speed
        total_ticks = int(traverse_time / DT)
        ff = np.array([self.direction[0]*self.speed,
                       self.direction[1]*self.speed])

        print(f"Traversing: {total_ticks} ticks ({traverse_time:.1f}s)...")
        sim_start = time.time()

        for tick in range(total_ticks + 1):
            progress = tick / max(total_ticks, 1)
            centroid = self.start + progress * (self.end - self.start)
            tgts = [self.target(i, centroid) for i in range(self.n)]
            self.step(tgts, ff=ff)
            self.log.append([d.pos.tolist() for d in self.drones])

            if tick % max(total_ticks // 10, 1) == 0:
                print(f"  {int(progress*100):3d}%  err={self.max_err(tgts):.2f}m")

        # Settle
        for _ in range(int(3.0 / DT)):
            tgts = [self.target(i, self.end) for i in range(self.n)]
            self.step(tgts)
            if self.converged(tgts, 0.15):
                break
        print("Traversal done.")

        sim_elapsed = time.time() - sim_start
        print(f"Traversal wall time: {sim_elapsed:.1f}s")

        # Save log
        with open("flight_log.txt", "w") as f:
            f.write(f"{self.n}\n")
            for positions in self.log:
                parts = [f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}" for p in positions]
                f.write(" ".join(parts) + "\n")
        print(f"Saved {len(self.log)} samples to flight_log.txt")

    def land(self):
        for d in self.drones:
            d.iface.land(0.4, wait=False)
        time.sleep(5.0)

    def shutdown(self):
        for d in self.drones:
            d.iface.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0','drone1','drone2','drone3','drone4'])
    parser.add_argument('-c', '--config', type=str,
                        default='scenarios/scenario1_stage3.yaml')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True)
    parser.add_argument('--altitude', type=float, default=1.0)
    parser.add_argument('--speed', type=float, default=0.8)
    parser.add_argument('--safety-margin', type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    rclpy.init()
    swarm = Swarm(args.namespaces, config, args)

    if input("Start mission? (y/n): ").strip().lower() != 'y':
        rclpy.shutdown()
        sys.exit(0)

    swarm.arm_and_takeoff()
    swarm.go_to_start()
    swarm.traverse()
    swarm.land()
    swarm.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
"""
./launch_as2.bash -s scenarios/scenario1_stage3.yaml -w config_sim/config/world_swarm.yaml

./launch_ground_station.bash -n drone0,drone1,drone2,drone3,drone4 -s scenarios/scenario1_stage3.yaml

python3 stage_3_decentralised.py \
    -n drone0 drone1 drone2 drone3 drone4 \
    -c scenarios/scenario1_stage3.yaml \
    --speed 1 --safety-margin 0.4
"""