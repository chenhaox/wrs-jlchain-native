#pragma once

#include "wrs_jlchain_native/backend.hpp"

#include <chain.hpp>
#include <chainfksolverpos_recursive.hpp>
#include <chainiksolverpos_nr_jl.hpp>
#include <chainiksolvervel_pinv.hpp>
#include <chainjnttojacsolver.hpp>
#include <jntarray.hpp>

#include <memory>

namespace wrs::jlchain {

class KDLBackend final : public IJLChainBackend {
public:
    explicit KDLBackend(ChainSpec spec);

    const char* name() const override { return "orocos_kdl"; }
    FKResult fk(const std::vector<double>& joint_values, bool with_jacobian) const override;
    FKResult update_state(const std::vector<double>& joint_values);
    std::optional<std::vector<double>> ik(
        const std::array<double, 3>& target_pos,
        const std::array<double, 9>& target_rotmat,
        const std::vector<double>& seed_joint_values) const override;

    const std::vector<double>& motion_values() const { return motion_values_; }
    const std::vector<double>& gl_pos_0() const { return gl_pos_0_; }
    const std::vector<double>& gl_rotmat_0() const { return gl_rotmat_0_; }
    const std::vector<double>& gl_motion_axis() const { return gl_motion_axis_; }
    const std::vector<double>& gl_pos_q() const { return gl_pos_q_; }
    const std::vector<double>& gl_rotmat_q() const { return gl_rotmat_q_; }
    const std::vector<double>& gl_flange_pos() const { return gl_flange_pos_; }
    const std::vector<double>& gl_flange_rotmat() const { return gl_flange_rotmat_; }

private:
    static KDL::Frame to_frame(const std::array<double, 3>& pos, const std::array<double, 9>& rotmat);
    static KDL::Vector to_vector(const std::array<double, 3>& vec);
    KDL::JntArray to_active_jnt_array(const std::vector<double>& joint_values) const;

    ChainSpec spec_;
    int active_dof_{0};
    int total_dof_{0};
    KDL::Chain chain_;
    KDL::JntArray q_min_;
    KDL::JntArray q_max_;
    std::vector<double> motion_values_;
    std::vector<double> gl_pos_0_;
    std::vector<double> gl_rotmat_0_;
    std::vector<double> gl_motion_axis_;
    std::vector<double> gl_pos_q_;
    std::vector<double> gl_rotmat_q_;
    std::vector<double> gl_flange_pos_;
    std::vector<double> gl_flange_rotmat_;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    std::unique_ptr<KDL::ChainJntToJacSolver> jac_solver_;
    std::unique_ptr<KDL::ChainIkSolverVel_pinv> ik_vel_solver_;
    std::unique_ptr<KDL::ChainIkSolverPos_NR_JL> ik_pos_solver_;
};

}  // namespace wrs::jlchain
