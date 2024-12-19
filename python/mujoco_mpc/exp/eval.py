#!/usr/bin/env python3
# Copyright 2023
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate a given controller on a specified MuJoCo MPC task multiple times.

This script:
1. Loads a specified MuJoCo MPC task model.
2. Initializes an agent (controller) for that task.
3. Runs the controller rollouts for N episodes.
4. Records and plots the cost (and cost terms) for evaluation.

Example:
  python evaluate_controller.py \
    --task Cartpole \
    --controller DummyController \
    --num-episodes 5
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import shutil
import re

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import tyro

from mujoco_mpc import agent as agent_lib


@dataclass
class EvaluationConfig:
    """Configuration for controller evaluation."""
    
    task: str = "Acrobot"
    """Task ID (e.g., 'Cartpole')."""
    
    controller: str = "Sampling"
    """Controller ID or configuration (string)."""
    
    num_episodes: int = 1
    """Number of episodes (rollouts) to evaluate."""
    
    model_path: Optional[Path] = None
    """Optional path to the task XML model. If not provided, a default path is assumed."""
    
    horizon: int = 1000
    """Total number of simulation steps for each rollout."""
    
    planner_steps: int = 4
    """Number of planner steps per environment step."""

planner_id_map = {
    "Sampling": 0,
    "Gradient": 1,
    "iLQG": 2,
    "iLQS": 3,
    "Robust Sampling": 4,
    "Cross Entropy": 5,
    "Sample Gradient": 6,
    "Feedback Sampling": 7,
}


def run_rollout(
    agent: agent_lib.Agent,
    model: mujoco.MjModel,
    controller: str,
    horizon: int = 1500,
    planner_steps: int = 10
):
    """Roll out the agent/controller for a single episode.

    Args:
        agent: An instance of the agent to control the model.
        model: The MuJoCo model for the given task.
        controller: A placeholder for controller configuration. Currently unused,
          but included for extensibility. Could be a string identifying a particular
          controller configuration, or a custom policy function.
        horizon: The total number of simulation steps (time steps) for the rollout.
        planner_steps: Number of planner steps per environment step.

    Returns:
        time: A (T,) array of time steps.
        qpos: A (model.nq, T) array of joint positions.
        qvel: A (model.nv, T) array of joint velocities.
        ctrl: A (model.nu, T-1) array of control inputs applied at each step.
        cost_total: A (T-1,) array of total costs at each step.
        cost_terms: A (num_cost_terms, T-1) array of individual cost terms.
        cost_term_names: A list of cost term names corresponding to `cost_terms`.
    """
    # Initialize arrays to store data
    T = horizon
    qpos = np.zeros((model.nq, T))
    qvel = np.zeros((model.nv, T))
    ctrl = np.zeros((model.nu, T - 1))
    time = np.zeros(T)
    cost_total = np.zeros(T - 1)
    cost_terms = None
    cost_term_names = None

    # Initialize MuJoCo data
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Cache initial state
    qpos[:, 0] = data.qpos
    qvel[:, 0] = data.qvel
    time[0] = data.time

    # Rollout loop
    for t in range(T - 1):
        # Update agent with current state
        agent.set_state(
            time=data.time,
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.act,
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
            userdata=data.userdata,
        )
        print(data.qpos)

        # Run planner steps
        for _ in range(planner_steps):
            agent.planner_step()

        # Get action from the agent (controller)
        action = agent.get_action()
        data.ctrl = action
        ctrl[:, t] = data.ctrl

        # Record costs
        cost_total[t] = agent.get_total_cost()
        term_vals = agent.get_cost_term_values()
        if cost_terms is None:
            cost_terms = np.zeros((len(term_vals), T - 1))
            cost_term_names = list(term_vals.keys())
        for i, val in enumerate(term_vals.values()):
            cost_terms[i, t] = val

        # Step forward in the simulation
        mujoco.mj_step(model, data)

        # Record state and time
        qpos[:, t + 1] = data.qpos
        qvel[:, t + 1] = data.qvel
        time[t + 1] = data.time

    return time, qpos, qvel, ctrl, cost_total, cost_terms, cost_term_names

def update_planner_config(model_path, planner_id):
    try:
        # Verify source file exists
        if not model_path.exists():
            raise FileNotFoundError(f"Source file not found: {model_path}")

        # Create the benchmark filename
        benchmark_path = model_path.with_name("task_benchmark.xml")

        # Copy the file
        shutil.copy(model_path, benchmark_path)

        # Read and modify the file
        with open(benchmark_path, "r") as file:
            content = file.read()

        # Use regex to handle potential variations in spacing and quotes
        modified_content = re.sub(
            r'<numeric\s+name="agent_planner"\s+data="\d+"\s*/>', 
            f'<numeric name="agent_planner" data="{planner_id}" />', 
            content
        )

        # Write the modified content back to file
        with open(benchmark_path, "w") as file:
            file.write(modified_content)

        return benchmark_path

    except Exception as e:
        print(f"Error updating planner configuration: {str(e)}")
        raise


def main(config: EvaluationConfig) -> None:
    """Main function to run evaluation.
    
    Args:
        config: Configuration dataclass containing all evaluation parameters.
    """

    
    # Derive or find model path if not provided
    if config.model_path is not None:
        model_path = config.model_path
    else:
        # Adjust the model path resolution as per project structure
        model_path = (
            Path(__file__).parent.parent
            / "../../build/mjpc/tasks"
            / config.task.lower()
            / "task.xml"
        )
    planner_id = planner_id_map[config.controller]
    try:
        new_file_path = update_planner_config(model_path, planner_id)
        print(f"Successfully created and updated: {new_file_path}")
    except Exception as e:
        print(f"Failed to update planner configuration: {e}")
    model_path = new_file_path

    # Load the model
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # Initialize the agent
    agent = agent_lib.Agent(task_id=config.task, model=model)

    # Lists to store cost data over episodes
    all_costs = []
    all_cost_terms = []

    # Run multiple episodes
    for episode in range(config.num_episodes):
        (time, qpos, qvel, ctrl, cost_total,
         cost_terms, cost_term_names) = run_rollout(
             agent, 
             model, 
             config.controller,
             horizon=config.horizon,
             planner_steps=config.planner_steps
         )
        all_costs.append(cost_total)
        all_cost_terms.append(cost_terms)
        agent.reset()

    # Convert to arrays for analysis
    all_costs = np.array(all_costs)
    mean_cost = np.mean(all_costs, axis=0)
    std_cost = np.std(all_costs, axis=0)

    # Plot reference rollout (first episode) plus mean and std
    plt.figure(figsize=(10, 6))
    plt.plot(time[:-1], all_costs[0], label="Rollout cost (episode 1)")
    plt.plot(time[:-1], mean_cost, label="Mean cost", color="red")
    plt.fill_between(
        time[:-1],
        mean_cost - std_cost,
        mean_cost + std_cost,
        color="red",
        alpha=0.2,
        label="Mean ± Std"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Cost")
    plt.title(f"Task: {config.task}, Controller: {config.controller}")
    plt.legend()
    plt.tight_layout()
    if not Path("plots").exists():
        Path("plots").mkdir()
    plt.savefig(f"plots/{config.task}_{config.controller}.png")


if __name__ == "__main__":
    tyro.cli(main)