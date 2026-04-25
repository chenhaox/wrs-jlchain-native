#include "wrs_jlchain_native/kdl_backend.hpp"

#include <jacobian.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wrs::jlchain {

namespace {

KDL::Rotation to_rotation(const std::array<double, 9>& rotmat)
{
    return KDL::Rotation(
        rotmat[0], rotmat[1], rotmat[2],
        rotmat[3], rotmat[4], rotmat[5],
        rotmat[6], rotmat[7], rotmat[8]);
}

std::array<double, 9> from_rotation(const KDL::Rotation& rot)
{
    return {
        rot(0, 0), rot(0, 1), rot(0, 2),
        rot(1, 0), rot(1, 1), rot(1, 2),
        rot(2, 0), rot(2, 1), rot(2, 2),
    };
}

std::array<double, 3> from_vector(const KDL::Vector& vec)
{
    return {vec.x(), vec.y(), vec.z()};
}

void write_vector(const KDL::Vector& vec, std::vector<double>& out, std::size_t offset)
{
    out[offset] = vec.x();
    out[offset + 1] = vec.y();
    out[offset + 2] = vec.z();
}

void write_rotation(const KDL::Rotation& rot, std::vector<double>& out, std::size_t offset)
{
    out[offset] = rot(0, 0);
    out[offset + 1] = rot(0, 1);
    out[offset + 2] = rot(0, 2);
    out[offset + 3] = rot(1, 0);
    out[offset + 4] = rot(1, 1);
    out[offset + 5] = rot(1, 2);
    out[offset + 6] = rot(2, 0);
    out[offset + 7] = rot(2, 1);
    out[offset + 8] = rot(2, 2);
}

}  // namespace

KDLBackend::KDLBackend(ChainSpec spec)
    : spec_(std::move(spec)),
      active_dof_(std::min<int>(static_cast<int>(spec_.joints.size()), spec_.flange_joint_id + 1)),
      total_dof_(static_cast<int>(spec_.joints.size())),
      q_min_(static_cast<unsigned int>(active_dof_)),
      q_max_(static_cast<unsigned int>(active_dof_)),
      motion_values_(static_cast<std::size_t>(total_dof_), 0.0),
      gl_pos_0_(static_cast<std::size_t>(total_dof_ * 3), 0.0),
      gl_rotmat_0_(static_cast<std::size_t>(total_dof_ * 9), 0.0),
      gl_motion_axis_(static_cast<std::size_t>(total_dof_ * 3), 0.0),
      gl_pos_q_(static_cast<std::size_t>(total_dof_ * 3), 0.0),
      gl_rotmat_q_(static_cast<std::size_t>(total_dof_ * 9), 0.0),
      gl_flange_pos_(3, 0.0),
      gl_flange_rotmat_(9, 0.0)
{
    if (total_dof_ == 0) {
        active_dof_ = 0;
    }
    if (active_dof_ < 0 || active_dof_ > total_dof_) {
        throw std::invalid_argument("Invalid flange_joint_id for JLChain backend.");
    }

    chain_.addSegment(KDL::Segment(
        "wrs_anchor",
        KDL::Joint(KDL::Joint::Fixed),
        to_frame(spec_.anchor_pos, spec_.anchor_rotmat)));

    for (int i = 0; i < active_dof_; ++i) {
        const JointSpec& jnt = spec_.joints[static_cast<std::size_t>(i)];
        const KDL::Frame loc_frame = to_frame(jnt.loc_pos, jnt.loc_rotmat);
        const KDL::Vector axis = loc_frame.M * to_vector(jnt.loc_motion_axis);
        const KDL::Joint::JointType kdl_type =
            jnt.type == JointType::Prismatic ? KDL::Joint::TransAxis : KDL::Joint::RotAxis;
        const KDL::Joint joint("wrs_j" + std::to_string(i), loc_frame.p, axis, kdl_type);
        chain_.addSegment(KDL::Segment("wrs_segment_" + std::to_string(i), joint, loc_frame));
        q_min_(static_cast<unsigned int>(i)) = jnt.lower;
        q_max_(static_cast<unsigned int>(i)) = jnt.upper;
    }

    chain_.addSegment(KDL::Segment(
        "wrs_flange",
        KDL::Joint(KDL::Joint::Fixed),
        to_frame(spec_.loc_flange_pos, spec_.loc_flange_rotmat)));

    fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
    jac_solver_ = std::make_unique<KDL::ChainJntToJacSolver>(chain_);
    ik_vel_solver_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(chain_);
    ik_pos_solver_ = std::make_unique<KDL::ChainIkSolverPos_NR_JL>(
        chain_, q_min_, q_max_, *fk_solver_, *ik_vel_solver_, spec_.max_iterations, spec_.ik_eps);
    update_state(std::vector<double>(static_cast<std::size_t>(total_dof_), 0.0));
}

KDL::Frame KDLBackend::to_frame(const std::array<double, 3>& pos, const std::array<double, 9>& rotmat)
{
    return KDL::Frame(to_rotation(rotmat), to_vector(pos));
}

KDL::Vector KDLBackend::to_vector(const std::array<double, 3>& vec)
{
    return KDL::Vector(vec[0], vec[1], vec[2]);
}

KDL::JntArray KDLBackend::to_active_jnt_array(const std::vector<double>& joint_values) const
{
    if (static_cast<int>(joint_values.size()) < active_dof_) {
        throw std::invalid_argument("Joint value vector is shorter than the active flange chain.");
    }
    KDL::JntArray q(static_cast<unsigned int>(active_dof_));
    for (int i = 0; i < active_dof_; ++i) {
        q(static_cast<unsigned int>(i)) = joint_values[static_cast<std::size_t>(i)];
    }
    return q;
}

FKResult KDLBackend::fk(const std::vector<double>& joint_values, bool with_jacobian) const
{
    KDL::Frame frame;
    KDL::JntArray q = to_active_jnt_array(joint_values);
    const int fk_status = fk_solver_->JntToCart(q, frame);
    if (fk_status < 0) {
        throw std::runtime_error("Orocos KDL FK failed.");
    }

    FKResult result;
    result.pos = from_vector(frame.p);
    result.rotmat = from_rotation(frame.M);
    result.jacobian_cols = total_dof_;

    if (with_jacobian) {
        KDL::Jacobian jac(static_cast<unsigned int>(active_dof_));
        const int jac_status = jac_solver_->JntToJac(q, jac);
        if (jac_status < 0) {
            throw std::runtime_error("Orocos KDL Jacobian failed.");
        }
        result.jacobian.assign(static_cast<std::size_t>(6 * total_dof_), 0.0);
        for (int col = 0; col < active_dof_; ++col) {
            for (int row = 0; row < 6; ++row) {
                result.jacobian[static_cast<std::size_t>(row * total_dof_ + col)] =
                    jac(row, static_cast<unsigned int>(col));
            }
        }
    }
    return result;
}

FKResult KDLBackend::update_state(const std::vector<double>& joint_values)
{
    if (static_cast<int>(joint_values.size()) < total_dof_) {
        throw std::invalid_argument("Joint value vector is shorter than n_dof.");
    }

    KDL::Frame reference = to_frame(spec_.anchor_pos, spec_.anchor_rotmat);
    for (int i = 0; i < total_dof_; ++i) {
        const JointSpec& jnt = spec_.joints[static_cast<std::size_t>(i)];
        const double q = joint_values[static_cast<std::size_t>(i)];
        const KDL::Frame loc_frame = to_frame(jnt.loc_pos, jnt.loc_rotmat);
        const KDL::Vector gl_pos_0 = reference * loc_frame.p;
        const KDL::Rotation gl_rotmat_0 = reference.M * loc_frame.M;
        const KDL::Vector gl_motion_axis = gl_rotmat_0 * to_vector(jnt.loc_motion_axis);

        KDL::Vector gl_pos_q = gl_pos_0;
        KDL::Rotation gl_rotmat_q = gl_rotmat_0;
        if (jnt.type == JointType::Revolute) {
            gl_rotmat_q = KDL::Rotation::Rot2(gl_motion_axis, q) * gl_rotmat_0;
        } else {
            gl_pos_q = gl_pos_0 + gl_motion_axis * q;
        }

        motion_values_[static_cast<std::size_t>(i)] = q;
        write_vector(gl_pos_0, gl_pos_0_, static_cast<std::size_t>(i * 3));
        write_rotation(gl_rotmat_0, gl_rotmat_0_, static_cast<std::size_t>(i * 9));
        write_vector(gl_motion_axis, gl_motion_axis_, static_cast<std::size_t>(i * 3));
        write_vector(gl_pos_q, gl_pos_q_, static_cast<std::size_t>(i * 3));
        write_rotation(gl_rotmat_q, gl_rotmat_q_, static_cast<std::size_t>(i * 9));

        reference = KDL::Frame(gl_rotmat_q, gl_pos_q);
    }

    KDL::Frame flange;
    if (total_dof_ >= 1) {
        const int flange_joint_id = std::clamp(spec_.flange_joint_id, 0, total_dof_ - 1);
        const std::size_t pos_offset = static_cast<std::size_t>(flange_joint_id * 3);
        const std::size_t rot_offset = static_cast<std::size_t>(flange_joint_id * 9);
        KDL::Vector flange_ref_pos(
            gl_pos_q_[pos_offset],
            gl_pos_q_[pos_offset + 1],
            gl_pos_q_[pos_offset + 2]);
        KDL::Rotation flange_ref_rot(
            gl_rotmat_q_[rot_offset], gl_rotmat_q_[rot_offset + 1], gl_rotmat_q_[rot_offset + 2],
            gl_rotmat_q_[rot_offset + 3], gl_rotmat_q_[rot_offset + 4], gl_rotmat_q_[rot_offset + 5],
            gl_rotmat_q_[rot_offset + 6], gl_rotmat_q_[rot_offset + 7], gl_rotmat_q_[rot_offset + 8]);
        flange = KDL::Frame(flange_ref_rot, flange_ref_pos) * to_frame(spec_.loc_flange_pos, spec_.loc_flange_rotmat);
    } else {
        flange = to_frame(spec_.anchor_pos, spec_.anchor_rotmat) * to_frame(spec_.loc_flange_pos, spec_.loc_flange_rotmat);
    }

    write_vector(flange.p, gl_flange_pos_, 0);
    write_rotation(flange.M, gl_flange_rotmat_, 0);

    FKResult result;
    result.pos = from_vector(flange.p);
    result.rotmat = from_rotation(flange.M);
    result.jacobian_cols = total_dof_;
    return result;
}

std::optional<std::vector<double>> KDLBackend::ik(
    const std::array<double, 3>& target_pos,
    const std::array<double, 9>& target_rotmat,
    const std::vector<double>& seed_joint_values) const
{
    KDL::JntArray q_init = to_active_jnt_array(seed_joint_values);
    KDL::JntArray q_out(static_cast<unsigned int>(active_dof_));
    const KDL::Frame target = to_frame(target_pos, target_rotmat);
    const int status = ik_pos_solver_->CartToJnt(q_init, target, q_out);
    if (status < 0) {
        return std::nullopt;
    }

    std::vector<double> result = seed_joint_values;
    result.resize(static_cast<std::size_t>(total_dof_), 0.0);
    for (int i = 0; i < active_dof_; ++i) {
        result[static_cast<std::size_t>(i)] = q_out(static_cast<unsigned int>(i));
    }
    return result;
}

}  // namespace wrs::jlchain
