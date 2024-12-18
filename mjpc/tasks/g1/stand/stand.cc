#include "mjpc/tasks/g1/stand/stand.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::g1
{

  std::string Stand::XmlPath() const
  {
    return GetModelPath("g1/stand/task.xml");
  }
  std::string Stand::Name() const { return "G1 Stand"; }

  // ------------------ Residuals for humanoid stand task ------------
  //   Number of residuals: 6
  //     Residual (0): Desired height
  //     Residual (1): Balance: COM_xy - average(feet position)_xy
  //     Residual (2): Com Vel: should be 0 and equal feet average vel
  //     Residual (3): Control: minimise control
  //     Residual (4): Joint vel: minimise joint velocity
  //   Number of parameters: 1
  //     Parameter (0): height_goal
  // ----------------------------------------------------------------
  void Stand::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                   double *residual) const
  {
    int counter = 0;

    int pelvis_body_id_ = mj_name2id(model, mjOBJ_XBODY, "pelvis");
    int torso_body_id = mj_name2id(model, mjOBJ_XBODY, "torso");
    int left_ankle_body_id = mj_name2id(model, mjOBJ_XBODY, "left_ankle_roll_link");
    int right_ankle_body_id = mj_name2id(model, mjOBJ_XBODY, "right_ankle_roll_link");
    int left_foot_body_id = mj_name2id(model, mjOBJ_XBODY, "left_foot");
    int right_foot_body_id = mj_name2id(model, mjOBJ_XBODY, "right_foot");

    // ----- Height: head-feet vertical error ----- //
    double *f1_position = SensorByName(model, data, "sp0");
    double *f2_position = SensorByName(model, data, "sp1");
    double *f3_position = SensorByName(model, data, "sp2");
    double *f4_position = SensorByName(model, data, "sp3");
    double *head_position = SensorByName(model, data, "head_position");
    double head_feet_error =
        head_position[2] - 0.25 * (f1_position[2] + f2_position[2] +
                                   f3_position[2] + f4_position[2]);
    residual[counter++] = head_feet_error - parameters_[0];

    // ----- Balance: CoM-feet xy error ----- //
    double *com_position = SensorByName(model, data, "pelvis_subtreecom");
    double *com_velocity = SensorByName(model, data, "pelvis_subtreelinvel");
    double kFallTime = 0.2;
    double capture_point[3] = {com_position[0], com_position[1], com_position[2]};
    mju_addToScl3(capture_point, com_velocity, kFallTime);

    // average feet xy position
    double fxy_avg[2] = {0.0, 0.0};
    mju_addTo(fxy_avg, f1_position, 2);
    mju_addTo(fxy_avg, f2_position, 2);
    mju_addTo(fxy_avg, f3_position, 2);
    mju_addTo(fxy_avg, f4_position, 2);
    mju_scl(fxy_avg, fxy_avg, 0.25, 2);

    // com-feet xy error
    mju_subFrom(fxy_avg, capture_point, 2);
    double com_feet_distance = mju_norm(fxy_avg, 2);
    residual[counter++] = com_feet_distance;

    // ----- COM xy velocity should be 0 ----- //
    mju_copy(&residual[counter], com_velocity, 2);
    counter += 2;

    // ----- joint velocity ----- //
    mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
    counter += (model->nv - 6);

    // ----- action ----- //
    mju_copy(&residual[counter], data->ctrl, model->nu);
    counter += model->nu;

    // ----- Posture ----- //
    double *home = KeyQPosByName(model, data, "stand");
    mju_sub(residual + counter, data->qpos + 7, home + 7, model->nu);
    // set index at 0, 3, 4, 6, 9, 10 to 0
    residual[counter + 0] = 0;
    residual[counter + 3] = 0;
    residual[counter + 4] = 0;
    residual[counter + 6] = 0;
    residual[counter + 9] = 0;
    residual[counter + 10] = 0;
    counter += model->nu;

    // ----- Upright ----- //
    // Ensure torso, pelvis, left_ankle_roll_link, right_ankle_roll_link stay close to zero rotation
    double identity_quat[4] = {1, 0, 0, 0};

    auto AddOrientationError = [&](int body_id)
    {
      double q_error[3];
      mju_subQuat(q_error, data->xquat + 4 * body_id, identity_quat);
      mju_copy(residual + counter, q_error, 3);
      counter += 3;
    };

    AddOrientationError(torso_body_id);
    AddOrientationError(pelvis_body_id_);
    AddOrientationError(left_ankle_body_id);
    AddOrientationError(right_ankle_body_id);

    // ----- Foot Goal ----- //
    // We want to ensure that left foot is on the left side and right foot on the right side of the torso.
    // Let's define a desired lateral spacing based on parameters_[1].
    double desired_spacing = parameters_[1]; // total desired foot spacing
    double left_foot_desired_y = -0.5 * desired_spacing;
    double right_foot_desired_y = 0.5 * desired_spacing;

    // Get torso frame
    const double *torso_pos = data->xpos + 3 * torso_body_id;
    const double *torso_mat = data->xmat + 9 * torso_body_id;

    auto FootError = [&](int foot_body_id, double desired_y)
    {
      // foot position in world frame
      const double *foot_pos = data->xpos + 3 * foot_body_id;

      // Transform foot_pos to torso frame
      double foot_rel[3];
      mju_sub3(foot_rel, foot_pos, torso_pos);

      // Rotate into torso frame: foot_rel = R_torso' * foot_rel
      double R_torso_T[9];
      mju_transpose(R_torso_T, torso_mat, 3, 3);
      double tmp[3];
      mju_mulMatVec3(tmp, R_torso_T, foot_rel);

      // tmp now holds foot position in torso frame: (x,y,z)
      // We only care about (x,y) for foot placement constraints.
      // For simplicity, let's require foot to have desired_y and x = 0
      // (or you can make another parameter for x if desired)
      double desired_x = 0.0; // assuming we want feet lined up at torso x

      residual[counter++] = tmp[0] - desired_x; // foot x error in torso frame
      residual[counter++] = tmp[1] - desired_y; // foot y error in torso frame
    };

    FootError(left_foot_body_id, left_foot_desired_y);
    FootError(right_foot_body_id, right_foot_desired_y);

    // sensor dim sanity check
    int user_sensor_dim = 0;
    for (int i = 0; i < model->nsensor; i++)
    {
      if (model->sensor_type[i] == mjSENS_USER)
      {
        user_sensor_dim += model->sensor_dim[i];
      }
    }
    if (user_sensor_dim != counter)
    {
      mju_error_i(
          "mismatch between total user-sensor dimension "
          "and actual length of residual %d",
          counter);
    }
  }

} // namespace mjpc::g1
