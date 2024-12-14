import time
import mujoco
import mujoco.viewer
import numpy as np
import argparse
import matplotlib.pyplot as plt
import scienceplots
from scipy.spatial.transform import Rotation as R
import time
from loop_rate_limiters import RateLimiter

# Shared Memory
from multiprocessing import shared_memory
import struct

# Unitree SDK2
import sys
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
    unitree_hg_msg_dds__MotorCmd_,
)
from unitree_sdk2py.utils.crc import CRC

plt.style.use(["science"])


def unpack_mocap_data(buffer):
    # Unpack all data at once
    unpacked_data = struct.unpack_from("q13d", buffer, 0)

    timestamp = unpacked_data[0]  # first element is the timestamp (microseconds)
    position = unpacked_data[1:4]
    quaternion = unpacked_data[4:8]
    linear_velocity = unpacked_data[8:11]
    angular_velocity = unpacked_data[11:14]

    q = np.concatenate([position, quaternion])
    qd = np.concatenate([linear_velocity, angular_velocity])

    return q, qd


class G1Real:
    def __init__(self, record=False):
        self.record_data = record

        # Parameters
        self.real_dt = 0.005
        self.delay_warning_threshold = 0.02
        self.delay_critical_threshold = 0.1
        self.mocap_offset = np.array([0.0, 0.0, 0.065])
        self.nq_robot = 29
        self.nu_robot = 29
        self.locked_joint_idx = 12 + 3 + np.array([3, 4, 5, 6, 10, 11, 12, 13])
        self.kp = np.array(
            [
                100,
                100,
                100,
                200,
                20,
                20,
                100,
                100,
                100,
                200,
                20,
                20,
                400,
                400,
                400,
                90,
                60,
                20,
                60,
                4,
                4,
                4,
                90,
                60,
                20,
                60,
                4,
                4,
                4,
            ]
        )
        self.kd = np.array(
            [
                2.5,
                2.5,
                2.5,
                5,
                0.2,
                0.1,
                2.5,
                2.5,
                2.5,
                5,
                0.2,
                0.1,
                5.0,
                5.0,
                5.0,
                2.0,
                1.0,
                0.4,
                1.0,
                0.2,
                0.2,
                0.2,
                2.0,
                1.0,
                0.4,
                1.0,
                0.2,
                0.2,
                0.2,
            ]
        )
        assert (
            len(self.kp) == len(self.kd) == self.nq_robot
        ), f"kp ({len(self.kp)}) and kd ({len(self.kd)}) must have the same length as nq_robot ({self.nq_robot})"

        # Mujoco related
        self.mj_model = mujoco.MjModel.from_xml_path(
            "/Users/pcy/Desktop/Research/code/mujoco_mpc-fork/mjpc/tasks/g1/stand/task.xml"
        )
        self.mj_model.opt.timestep = 0.02
        self.mj_data = mujoco.MjData(self.mj_model)
        self.q = self.mj_model.key_qpos[0].copy()
        self.qd = self.mj_model.key_qvel[0].copy()
        self.nq = self.mj_model.nq
        self.nqd = self.mj_model.nv
        self.nu = self.mj_model.nu
        assert (
            self.nq + len(self.locked_joint_idx) - 7
        ) == self.nq_robot, f"nq ({self.nq}) + len(locked_joint_idx) ({len(self.locked_joint_idx)}) - 7 must equal nq_robot ({self.nq_robot})"

        # Control message (send to robot)
        self.ctrl_msg = unitree_hg_msg_dds__LowCmd_()
        self.ctrl_msg.mode_pr = 0
        self.ctrl_msg.mode_machine = 3

        # Shared Memory for control inputs
        self.ctrl_shm_size = 8 + self.nu * 8  # time + tau + q + qd
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

        # Shared Memory for mocap state
        self.shared_mem_name = "mocap_state_shm"
        self.shared_mem_size = (
            8 + 13 * 8
        )  # 8 bytes for utime (int64), 13 float64s (13*8 bytes)
        try:
            self.mocap_shm = shared_memory.SharedMemory(
                name=self.shared_mem_name, create=True, size=self.shared_mem_size
            )
        except FileNotFoundError:
            self.mocap_shm = shared_memory.SharedMemory(
                name=self.shared_mem_name, create=False, size=self.shared_mem_size
            )
            self.mocap_shm.close()
            self.mocap_shm.unlink()
            self.mocap_shm = shared_memory.SharedMemory(
                name=self.shared_mem_name, create=True, size=self.shared_mem_size
            )
        self.mocap_buffer = self.mocap_shm.buf

        # Initialize Unitree SDK2
        ChannelFactoryInitialize(0, "lo0")
        self.low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_subscriber.Init(self.LowStateHandler, 10)

        self.low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_publisher.Init()

        self.low_cmd_msg = unitree_hg_msg_dds__LowCmd_()
        self.low_cmd_msg.mode_pr = 0
        self.low_cmd_msg.mode_machine = 3
        self.crc = CRC()

    def LowStateHandler(self, msg: LowState_):
        for i in range(self.nq_robot):
            self.q[7 + i] = msg.motor_state[i].q
            self.qd[6 + i] = msg.motor_state[i].dq

    def convert_ctrl_to_robot_ctrl(self, ctrl):
        ctrl_full = np.zeros(self.nu_robot)
        locked_mask = np.zeros(self.nu_robot, dtype=bool)
        locked_mask[self.locked_joint_idx] = True
        ctrl_full[~locked_mask] = ctrl
        return ctrl_full

    def convert_robot_state_to_mujoco_state(self):
        q_jnt = self.q[7 : 7 + self.nq_robot]
        qd_jnt = self.qd[6 : 6 + self.nq_robot]
        locked_mask = np.zeros(self.nq, dtype=bool)
        locked_mask[self.locked_joint_idx] = True
        q_mujoco = np.zeros(self.nq)
        q_mujoco[:7] = self.q[:7]
        q_mujoco[7 : 7 + self.nq_robot] = q_jnt[~locked_mask]
        qd_mujoco = np.zeros(self.nqd)
        qd_mujoco[:6] = self.qd[:6]
        qd_mujoco[6 : 6 + self.nq_robot] = qd_jnt[~locked_mask]
        return q_mujoco, qd_mujoco

    def main_loop(self):
        t0 = time.time()
        rate_limiter = RateLimiter(frequency=1 / self.real_dt)
        try:
            with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, show_left_ui=True, show_right_ui=False
            ) as viewer:
                while viewer.is_running():
                    # Read Mocap state from shared memory
                    q_base, qd_base = unpack_mocap_data(self.mocap_buffer)
                    self.q[:7] = q_base
                    self.qd[:6] = qd_base

                    # Read control inputs from shared memory
                    ctrl_data = self.ctrl_buffer[:]
                    ctrl = np.frombuffer(
                        ctrl_data[8 : 8 + self.nu_robot * 8], dtype=np.float64
                    )
                    ctrl_robot = self.convert_ctrl_to_robot_ctrl(ctrl)

                    # Publish control via Unitree SDK2
                    for idx in range(self.nu_robot):
                        self.low_cmd_msg.motor_cmd[idx].mode = (
                            0x01  # Set appropriate mode
                        )
                        self.low_cmd_msg.motor_cmd[idx].q = ctrl_robot[idx]
                        self.low_cmd_msg.motor_cmd[idx].dq = 0.0
                        self.low_cmd_msg.motor_cmd[idx].tau = 0.0
                        self.low_cmd_msg.motor_cmd[idx].kp = self.kp[idx]
                        self.low_cmd_msg.motor_cmd[idx].kd = self.kd[idx]

                    self.low_cmd_msg.crc = self.crc.Crc(self.low_cmd_msg)
                    self.low_cmd_publisher.Write(self.low_cmd_msg)

                    # Write the state to shared memory
                    t_real = time.time() - t0
                    state_data = bytearray(self.state_shm_size)
                    struct.pack_into("d", state_data, 0, t_real)
                    state_data[8 : 8 + self.nq * 8] = self.q.tobytes()
                    state_data[8 + self.nq * 8 : 8 + self.nq * 8 + self.nqd * 8] = (
                        self.qd.tobytes()
                    )
                    self.state_buffer[:] = state_data

                    q_mujoco, qd_mujoco = self.convert_robot_state_to_mujoco_state()
                    self.mj_data.qpos = q_mujoco
                    self.mj_data.qvel = qd_mujoco
                    mujoco.mj_kinematics(self.mj_model, self.mj_data)

                    # Update the viewer
                    viewer.sync()

                    rate_limiter.sleep()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            # Clean up shared memory
            self.state_shm.close()
            self.state_shm.unlink()
            self.ctrl_shm.close()
            self.ctrl_shm.unlink()
            self.mocap_shm.close()
            self.mocap_shm.unlink()


if __name__ == "__main__":
    viz = G1Real()
    viz.main_loop()
