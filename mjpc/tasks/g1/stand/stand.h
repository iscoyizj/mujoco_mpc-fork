#ifndef MJPC_TASKS_G1_STAND_TASK_H_
#define MJPC_TASKS_G1_STAND_TASK_H_

#include <memory>
#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace g1 {

class Stand : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Stand* task) : mjpc::BaseResidualFn(task) {}

    // ------------------ Residuals for humanoid stand task ------------
    //   Number of residuals: 6
    //     Residual (0): control
    //     Residual (1): COM_xy - average(feet position)_xy
    //     Residual (2): torso_xy - COM_xy
    //     Residual (3): head_z - feet^{(i)}_position_z - height_goal
    //     Residual (4): velocity COM_xy
    //     Residual (5): joint velocity
    //   Number of parameters: 1
    //     Parameter (0): height_goal
    // ----------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  Stand() : residual_(this) {}

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace g1
}  // namespace mjpc

#endif  // MJPC_TASKS_G1_STAND_TASK_H_
