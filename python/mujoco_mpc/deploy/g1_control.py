import numpy as np
import struct
from multiprocessing import shared_memory
from loop_rate_limiters import RateLimiter

import mujoco
from mujoco_mpc import agent as agent_lib

class G1Controller:
    def __init__(self):
        self.nq = 28
        self.nqd = 27
        self.nu = 21
        self.ctrl_dt = 0.02

        # Shared memory
        try:
            self.state_shm = shared_memory.SharedMemory(name="state_shm")
            self.state_buffer = self.state_shm.buf
        except FileNotFoundError:
            print("State shared memory 'state_shm' not found.")
            exit()

        try:
            self.ctrl_shm = shared_memory.SharedMemory(name="ctrl_shm")
            self.ctrl_buffer = self.ctrl_shm.buf
        except FileNotFoundError:
            print("Could not create control shared memory 'ctrl_shm'.")
            exit()

        # Controller
        model = mujoco.MjModel.from_xml_path("/Users/pcy/Desktop/Research/code/mujoco_mpc-fork/mjpc/tasks/g1/stand/task.xml")
        self.agent = agent_lib.Agent(task_id="G1 Stand", model=model)
        self.num_opt_steps = 1

        # Initialize control variables
        self.last_plan_time = 0.0
        self.state = None  # Will be initialized in main_loop

    def get_action(self, qpos, qvel, t):
        self.agent.set_state(
            time=t,
            qpos=qpos,
            qvel=qvel
        )
        for _ in range(self.num_opt_steps):
            self.agent.planner_step()
        ctrl = self.agent.get_action()
        return ctrl
    
    def get_state(self):
        state_data = self.state_buffer[:]
        t_real = struct.unpack("d", state_data[0:8])[0]
        q = np.frombuffer(state_data[8 : 8 + self.nq * 8], dtype=np.float64)
        qd = np.frombuffer(
            state_data[8 + self.nq * 8 : 8 + self.nq * 8 + self.nqd * 8],
            dtype=np.float64,
        )
        return t_real, q, qd

    def main_loop(self):
        rate_limiter = RateLimiter(frequency=1/self.ctrl_dt)
        try:
            while True:
                t_real, q, qd = self.get_state()
                ctrl = self.get_action(q, qd, t_real)
                struct.pack_into("d", self.ctrl_buffer, 0, t_real)
                struct.pack_into(f"{self.nu}d", self.ctrl_buffer, 8, *ctrl)
                rate_limiter.sleep()

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting...")
        finally:
            self.state_shm.close()
            self.ctrl_shm.close()

def main():
    controller = G1Controller()
    controller.main_loop()


if __name__ == "__main__":
    main()