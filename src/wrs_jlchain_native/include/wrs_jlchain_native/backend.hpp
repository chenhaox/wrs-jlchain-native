#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace wrs::jlchain {

enum class JointType {
    Revolute = 1,
    Prismatic = 2,
};

struct JointSpec {
    std::array<double, 3> loc_pos{};
    std::array<double, 9> loc_rotmat{};
    std::array<double, 3> loc_motion_axis{};
    JointType type{JointType::Revolute};
    double lower{0.0};
    double upper{0.0};
};

struct ChainSpec {
    std::array<double, 3> anchor_pos{};
    std::array<double, 9> anchor_rotmat{};
    std::vector<JointSpec> joints;
    int flange_joint_id{0};
    std::array<double, 3> loc_flange_pos{};
    std::array<double, 9> loc_flange_rotmat{};
    unsigned int max_iterations{100};
    double ik_eps{1e-6};
};

struct FKResult {
    std::array<double, 3> pos{};
    std::array<double, 9> rotmat{};
    std::vector<double> jacobian;
    int jacobian_cols{0};
};

class IJLChainBackend {
public:
    virtual ~IJLChainBackend() = default;
    virtual const char* name() const = 0;
    virtual FKResult fk(const std::vector<double>& joint_values, bool with_jacobian) const = 0;
    virtual std::optional<std::vector<double>> ik(
        const std::array<double, 3>& target_pos,
        const std::array<double, 9>& target_rotmat,
        const std::vector<double>& seed_joint_values) const = 0;
};

}  // namespace wrs::jlchain
