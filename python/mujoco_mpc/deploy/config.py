import numpy as np

class G1Config:
    # model used in controller
    nq_ctrl: int = 28
    nqd_ctrl: int = 27
    nu_ctrl: int = 21
    dt_ctrl: float = 0.02

    # model used in real robot
    nq_real: int = 36
    nqd_real: int = 35
    nu_real: int = 29
    dt_real: float = 0.005
    locked_joint_idx: np.ndarray = (
        np.array([2, 4, 5, 6, 9, 11, 12, 13]) + 12 + 3
    )  # locked joints in real robot
    kp_real: np.ndarray = np.array(
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
    kd_real: np.ndarray = np.array(
        [
            2.5,
            2.5,
            2.5,
            5,
            0.2,
            0.2,
            2.5,
            2.5,
            2.5,
            5,
            0.2,
            0.2,
            10,
            10,
            10,
            1.5,
            1,
            0.2,
            1,
            0.05,
            0.05,
            0.05,
            1.5,
            1,
            0.2,
            1,
            0.05,
            0.05,
            0.05,
        ]
    )

    # mocap
    mocap_offset: np.ndarray = np.array([0.0, 0.0, 0.0])
    use_mocap_ang_vel: bool = False
    vicon_tracker_ip: str = "128.2.184.3"
    vicon_object_name: str = "lecar_g1"

    # controller
    xml_path_ctrl: str = (
        "/Users/tairanhe/Workspace/mpc_chaoyi/mjpc/tasks/g1/stand/task.xml"
    )
    num_opt_steps: int = 1
    task_id: str = "G1 Stand"

    # sim
    xml_path_sim: str = (
        "./model/g1/g1.xml"
    )
    dt_sim: float = 0.005
    real_time_factor: float = 1.0
    auto_reset: bool = True

class Go2Config:
    # model used in controller
    nq_ctrl: int = 19
    nqd_ctrl: int = 18
    nu_ctrl: int = 12
    dt_ctrl: float = 0.02

    # model used in real robot
    nq_real: int = 19
    nqd_real: int = 18
    nu_real: int = 12
    dt_real: float = 0.005
    locked_joint_idx: np.ndarray = np.zeros(0)
    kp_real: np.ndarray = np.array(
        [80.0] * 12
    )
    kd_real: np.ndarray = np.array(
        [5.0] * 12
    )

    # mocap
    mocap_offset: np.ndarray = np.array([0.0, 0.0, -0.05])
    use_mocap_ang_vel: bool = False
    vicon_tracker_ip: str = "128.2.184.3"
    vicon_object_name: str = "lecar_go2_mpc"

    # controller
    xml_path_ctrl: str = (
        "/Users/tairanhe/Workspace_MPC_Chaoyi/mpc_chaoyi/mjpc/tasks/go2/task_flat.xml"
    )
    num_opt_steps: int = 1
    task_id: str = "Go2 Flat"

    # sim
    xml_path_sim: str = (
        # "/home/pcy/Research/code/mjpc_sim2real_john/mjpc_john/mjpc/tasks/quadruped/task_flat.xml"
        "./model/go2/go2.xml"
    )
    dt_sim: float = 0.002
    ctrl_dt: float = 0.002
    real_time_factor: float = 1.0
    auto_reset: bool = False
    _targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
    _targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                            0.0, 0.67, -1.3, 0.0, 0.67, -1.3]
    _targetPos_3 = [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
                            -0.5, 1.36, -2.65, 0.5, 1.36, -2.65]

    startPos = [0.0] * 12
    duration_1 = 500
    duration_2 = 500
    duration_3 = 1000
    duration_4 = 900
    percent_1 = 0
    percent_2 = 0
    percent_3 = 0
    percent_4 = 0

class QuadrupedConfig(Go2Config):
    dt_ctrl: float = 0.005
    dt_sim: float = 0.005
    xml_path_ctrl: str = "/home/pcy/Research/code/mjpc_sim2real_john/mjpc_john/mjpc/tasks/quadruped/task_flat.xml"
    task_id: str = "Quadruped Flat"
    xml_path_sim: str = "/home/pcy/Research/code/mjpc_sim2real_john/mjpc_john/mjpc/tasks/quadruped/task_flat.xml"
