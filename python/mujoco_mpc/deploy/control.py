import numpy as np
import struct
from multiprocessing import shared_memory
from loop_rate_limiters import RateLimiter

import mujoco
from mujoco_mpc import agent as agent_lib

from config import G1Config, Go2Config
from utils import pack_control_data, unpack_mocap_data, unpack_state_data, ctrl_sim2real, state_real2sim

class Controller:
    def __init__(self, robot_name="g1"):
        if robot_name == "g1":
            self.config = G1Config()
        elif robot_name == "go2":
            self.config = Go2Config()
        else:
            raise ValueError(f"Robot {robot_name} not supported")

        # State variables read from real robot
        try:
            self.state_shm = shared_memory.SharedMemory(name="state_shm")
            self.state_buffer = self.state_shm.buf
        except FileNotFoundError:
            print("State shared memory 'state_shm' not found.")
            exit()
        # Control variables written to real robot
        try:
            self.ctrl_shm = shared_memory.SharedMemory(name="ctrl_shm")
            self.ctrl_buffer = self.ctrl_shm.buf
        except FileNotFoundError:
            print("Could not create control shared memory 'ctrl_shm'.")
            exit()

        # Controller
        model = mujoco.MjModel.from_xml_path(self.config.xml_path_ctrl)
        self.agent = agent_lib.Agent(task_id=self.config.task_id, model=model)

        # Initialize control variables
        self.last_plan_time = 0.0
        self.state = None  # Will be initialized in main_loop

    def get_action(self, qpos, qvel, t):
        self.agent.set_state(time=t, qpos=qpos, qvel=qvel)
        for _ in range(self.config.num_opt_steps):
            self.agent.planner_step()
        ctrl = self.agent.get_action()
        return ctrl

    def main_loop(self):
        rate_limiter = RateLimiter(frequency=1 / self.config.dt_ctrl)
        try:
            while True:
                t_real, q_real, qd_real = unpack_state_data(self.state_buffer, self.config.nq_real, self.config.nqd_real)
                q_sim, qd_sim = state_real2sim(q_real, qd_real, self.config.locked_joint_idx, self.config.nq_ctrl, self.config.nqd_ctrl, self.config.nq_real, self.config.nqd_real)
                ctrl = self.get_action(q_sim, qd_sim, t_real)
                ctrl_real = ctrl_sim2real(ctrl, self.config.locked_joint_idx, self.config.nu_real)
                self.ctrl_buffer[:] = pack_control_data(self.ctrl_buffer, t_real, ctrl_real)
                rate_limiter.sleep()

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting...")
        finally:
            self.state_shm.close()
            self.ctrl_shm.close()


if __name__ == "__main__":
    controller = Controller(robot_name="go2")
    controller.main_loop()

