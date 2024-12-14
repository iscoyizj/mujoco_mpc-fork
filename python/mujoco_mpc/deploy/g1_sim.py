import os
import time
import mujoco
import mujoco.viewer
import numpy as np
import argparse
import matplotlib.pyplot as plt
import scienceplots
from scipy.spatial.transform import Rotation as R
import time
from copy import deepcopy

# Shared Memory
from multiprocessing import shared_memory
import struct

plt.style.use(["science"])


class G1Sim:
    def __init__(self, render=False):
        # Parameters
        self.render = render
        self.mode = "async" # "sync" or "async"
        self.real_time_factor = 1.0
        self.auto_reset = True
        self.ctrl_dt = 0.02
        self.sim_dt = 0.01  # Simulation timestep
        self.n_frame = int(self.ctrl_dt / self.sim_dt)
        self.delay_warning_threshold = 0.03
        self.delay_critical_threshold = 0.1

        # MuJoCo model setup
        self.mj_model = mujoco.MjModel.from_xml_path("/Users/pcy/Desktop/Research/code/mujoco_mpc-fork/mjpc/tasks/g1/stand/task.xml")
        self.mj_model.opt.timestep = self.sim_dt
        self.mj_data = mujoco.MjData(self.mj_model)

        # Initialize state variables
        self.q = self.mj_model.key_qpos[0].copy()
        self.qd = self.mj_model.key_qvel[0].copy()
        self.nu_msg = self.mj_model.nu  # Number of motors in the interface
        self.nq = self.mj_model.nq
        self.nqd = self.mj_model.nv
        self.nu = self.mj_model.nu

        # Shared Memory for control inputs
        self.ctrl_shm_size = 8 + self.nu_msg * 8  # time + q_des
        self.ctrl_shm = shared_memory.SharedMemory(
            name="ctrl_shm", create=True, size=self.ctrl_shm_size
        )
        self.ctrl_buffer = self.ctrl_shm.buf

        # Shared Memory for state variables
        self.state_shm_size = 8 + self.nq * 8 + self.nqd * 8  # time + q + qd
        self.state_shm = shared_memory.SharedMemory(
            name="state_shm", create=True, size=self.state_shm_size
        )
        self.state_buffer = self.state_shm.buf

    def main_loop(self):
        t0 = time.time()
        sim_time = 0.0
        try:
            with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
            ) as viewer:
                while True:
                    t_start = time.time()
                    # Read control inputs from shared memory
                    ctrl_data = self.ctrl_buffer[:]
                    self.ctrl_time = struct.unpack("d", ctrl_data[0:8])[0]
                    self.q_des = np.frombuffer(
                        ctrl_data[8 : 8 + self.nu_msg * 8], dtype=np.float64
                    )
                    self.mj_data.ctrl[:] = self.q_des

                    mujoco.mj_step(self.mj_model, self.mj_data)

                    # Get the state from the MuJoCo model
                    self.q = self.mj_data.qpos.copy()
                    self.qd = self.mj_data.qvel.copy()

                    # Write the state to shared memory
                    # Write directly to shared memory buffer
                    struct.pack_into("d", self.state_buffer, 0, self.mj_data.time)  # Write time
                    struct.pack_into(f"{self.nq}d", self.state_buffer, 8, *self.q)  # Write positions
                    struct.pack_into(f"{self.nqd}d", self.state_buffer, 8 + self.nq * 8, *self.qd)  # Write velocities
                    # state_data = bytearray(self.state_shm_size)
                    # struct.pack_into("d", state_data, 0, self.mj_data.time)
                    # state_data[8 : 8 + self.nq * 8] = self.q.tobytes()
                    # state_data[8 + self.nq * 8 : 8 + self.nq * 8 + self.nqd * 8] = (
                    #     self.qd.tobytes()
                    # )
                    # check two side value structures if match
                    # self.state_buffer[:] = state_data

                    # Check if robot failed, if so, reset the simulation
                    vec_tar = np.array([0.0, 0.0, 1.0])
                    x_rot_torso = self.mj_data.xquat[1]
                    vec = R.from_quat(x_rot_torso, scalar_first=True).apply(vec_tar)
                    d2upright = np.linalg.norm(vec - vec_tar)
                    if d2upright > 0.8:
                        print(f"Warning: Robot is not upright: {d2upright}")
                        if self.auto_reset:
                            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
                            print("Resetting simulation...")

                    if self.render:
                        # Update the viewer
                        viewer.sync()

                    # Sleep to maintain the loop rate
                    if self.mode == "async":
                        t_end = time.time()
                        duration = t_end - t_start
                        if duration < (self.sim_dt / self.real_time_factor):
                            time.sleep(self.sim_dt / self.real_time_factor - duration)
                        else:
                            print("[WARN] Simulation loop overran")
                    elif self.mode == "sync":
                        t_wait_0 = time.time()
                        while True:
                            # wait until the next control is available
                            ctrl_data = self.ctrl_buffer[:]
                            self.ctrl_time = struct.unpack("d", ctrl_data[0:8])[0]
                            if self.ctrl_time >= (sim_time - self.ctrl_dt):
                                delta_t = time.time() - t_wait_0
                                if delta_t > 0.01:
                                    print(f"sync mode real time factor: {self.ctrl_dt / delta_t:.2f}")
                                break
                            else:
                                time.sleep(0.001)

        finally:
            # Clean up shared memory
            self.state_shm.close()
            self.state_shm.unlink()
            self.ctrl_shm.close()
            self.ctrl_shm.unlink()


if __name__ == "__main__":
    sim = G1Sim(render=True)
    sim.main_loop()