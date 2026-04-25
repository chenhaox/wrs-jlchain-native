#include "wrs_jlchain_native/kdl_backend.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

namespace py = pybind11;
namespace wj = wrs::jlchain;

namespace {

template <std::size_t N>
std::array<double, N> read_array(py::array_t<double, py::array::c_style | py::array::forcecast> array, const char* name)
{
    py::buffer_info info = array.request();
    if (static_cast<std::size_t>(info.size) != N) {
        std::ostringstream oss;
        oss << name << " must contain " << N << " doubles.";
        throw std::invalid_argument(oss.str());
    }
    std::array<double, N> result{};
    const auto* ptr = static_cast<const double*>(info.ptr);
    std::copy(ptr, ptr + N, result.begin());
    return result;
}

std::vector<wj::JointSpec> read_joints(
    py::array_t<double, py::array::c_style | py::array::forcecast> loc_pos,
    py::array_t<double, py::array::c_style | py::array::forcecast> loc_rotmat,
    py::array_t<double, py::array::c_style | py::array::forcecast> loc_motion_axis,
    py::array_t<int, py::array::c_style | py::array::forcecast> joint_types,
    py::array_t<double, py::array::c_style | py::array::forcecast> joint_ranges)
{
    const py::buffer_info pos_info = loc_pos.request();
    const py::buffer_info rot_info = loc_rotmat.request();
    const py::buffer_info axis_info = loc_motion_axis.request();
    const py::buffer_info type_info = joint_types.request();
    const py::buffer_info range_info = joint_ranges.request();

    if (pos_info.ndim != 2 || pos_info.shape[1] != 3) {
        throw std::invalid_argument("loc_pos must have shape (n_dof, 3).");
    }
    const auto n_dof = static_cast<std::size_t>(pos_info.shape[0]);
    if (rot_info.ndim != 3 || static_cast<std::size_t>(rot_info.shape[0]) != n_dof ||
        rot_info.shape[1] != 3 || rot_info.shape[2] != 3) {
        throw std::invalid_argument("loc_rotmat must have shape (n_dof, 3, 3).");
    }
    if (axis_info.ndim != 2 || static_cast<std::size_t>(axis_info.shape[0]) != n_dof || axis_info.shape[1] != 3) {
        throw std::invalid_argument("loc_motion_axis must have shape (n_dof, 3).");
    }
    if (type_info.ndim != 1 || static_cast<std::size_t>(type_info.shape[0]) != n_dof) {
        throw std::invalid_argument("joint_types must have shape (n_dof,).");
    }
    if (range_info.ndim != 2 || static_cast<std::size_t>(range_info.shape[0]) != n_dof || range_info.shape[1] != 2) {
        throw std::invalid_argument("joint_ranges must have shape (n_dof, 2).");
    }

    const auto* pos = static_cast<const double*>(pos_info.ptr);
    const auto* rot = static_cast<const double*>(rot_info.ptr);
    const auto* axis = static_cast<const double*>(axis_info.ptr);
    const auto* types = static_cast<const int*>(type_info.ptr);
    const auto* ranges = static_cast<const double*>(range_info.ptr);

    std::vector<wj::JointSpec> joints;
    joints.reserve(n_dof);
    for (std::size_t i = 0; i < n_dof; ++i) {
        wj::JointSpec spec;
        std::copy(pos + i * 3, pos + i * 3 + 3, spec.loc_pos.begin());
        std::copy(rot + i * 9, rot + i * 9 + 9, spec.loc_rotmat.begin());
        std::copy(axis + i * 3, axis + i * 3 + 3, spec.loc_motion_axis.begin());
        spec.type = types[i] == 2 ? wj::JointType::Prismatic : wj::JointType::Revolute;
        spec.lower = ranges[i * 2];
        spec.upper = ranges[i * 2 + 1];
        joints.push_back(spec);
    }
    return joints;
}

py::array_t<double> make_1d_array(const std::array<double, 3>& values)
{
    py::array_t<double> array({3});
    auto out = array.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < 3; ++i) {
        out(i) = values[static_cast<std::size_t>(i)];
    }
    return array;
}

py::array_t<double> make_rotmat_array(const std::array<double, 9>& values)
{
    py::array_t<double> array({3, 3});
    auto out = array.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < 3; ++row) {
        for (py::ssize_t col = 0; col < 3; ++col) {
            out(row, col) = values[static_cast<std::size_t>(row * 3 + col)];
        }
    }
    return array;
}

py::array_t<double> make_jacobian_array(const std::vector<double>& values, int cols)
{
    py::array_t<double> array({6, cols});
    auto out = array.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < 6; ++row) {
        for (py::ssize_t col = 0; col < cols; ++col) {
            out(row, col) = values[static_cast<std::size_t>(row * cols + col)];
        }
    }
    return array;
}

py::array_t<double> make_vector_view(const std::vector<double>& values, py::object base)
{
    return py::array_t<double>(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(values.size())},
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(sizeof(double))},
        const_cast<double*>(values.data()),
        std::move(base));
}

py::array_t<double> make_matrix_view(const std::vector<double>& values, py::ssize_t rows, py::ssize_t cols, py::object base)
{
    return py::array_t<double>(
        std::vector<py::ssize_t>{rows, cols},
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(cols * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
        const_cast<double*>(values.data()),
        std::move(base));
}

}  // namespace

PYBIND11_MODULE(_wrs_jlchain_native_kdl, m)
{
    py::class_<wj::KDLBackend>(m, "KDLBackend")
        .def(py::init([](
                          py::array_t<double, py::array::c_style | py::array::forcecast> anchor_pos,
                          py::array_t<double, py::array::c_style | py::array::forcecast> anchor_rotmat,
                          py::array_t<double, py::array::c_style | py::array::forcecast> loc_pos,
                          py::array_t<double, py::array::c_style | py::array::forcecast> loc_rotmat,
                          py::array_t<double, py::array::c_style | py::array::forcecast> loc_motion_axis,
                          py::array_t<int, py::array::c_style | py::array::forcecast> joint_types,
                          py::array_t<double, py::array::c_style | py::array::forcecast> joint_ranges,
                          int flange_joint_id,
                          py::array_t<double, py::array::c_style | py::array::forcecast> loc_flange_pos,
                          py::array_t<double, py::array::c_style | py::array::forcecast> loc_flange_rotmat,
                          unsigned int max_iterations,
                          double ik_eps) {
                 wj::ChainSpec spec;
                 spec.anchor_pos = read_array<3>(anchor_pos, "anchor_pos");
                 spec.anchor_rotmat = read_array<9>(anchor_rotmat, "anchor_rotmat");
                 spec.joints = read_joints(loc_pos, loc_rotmat, loc_motion_axis, joint_types, joint_ranges);
                 spec.flange_joint_id = flange_joint_id;
                 spec.loc_flange_pos = read_array<3>(loc_flange_pos, "loc_flange_pos");
                 spec.loc_flange_rotmat = read_array<9>(loc_flange_rotmat, "loc_flange_rotmat");
                 spec.max_iterations = max_iterations;
                 spec.ik_eps = ik_eps;
                 return std::make_unique<wj::KDLBackend>(std::move(spec));
             }),
             py::arg("anchor_pos"),
             py::arg("anchor_rotmat"),
             py::arg("loc_pos"),
             py::arg("loc_rotmat"),
             py::arg("loc_motion_axis"),
             py::arg("joint_types"),
             py::arg("joint_ranges"),
             py::arg("flange_joint_id"),
             py::arg("loc_flange_pos"),
             py::arg("loc_flange_rotmat"),
             py::arg("max_iterations") = 100,
             py::arg("ik_eps") = 1e-6)
        .def_property_readonly("name", [](const wj::KDLBackend& self) { return self.name(); })
        .def("fk", [](const wj::KDLBackend& self,
                      py::array_t<double, py::array::c_style | py::array::forcecast> joint_values,
                      bool toggle_jacobian) -> py::object {
            py::buffer_info info = joint_values.request();
            const auto* ptr = static_cast<const double*>(info.ptr);
            std::vector<double> q(ptr, ptr + info.size);
            wj::FKResult result = self.fk(q, toggle_jacobian);
            if (toggle_jacobian) {
                py::tuple value = py::make_tuple(
                    make_1d_array(result.pos),
                    make_rotmat_array(result.rotmat),
                    make_jacobian_array(result.jacobian, result.jacobian_cols));
                return value;
            }
            py::tuple value = py::make_tuple(make_1d_array(result.pos), make_rotmat_array(result.rotmat));
            return value;
        }, py::arg("joint_values"), py::arg("toggle_jacobian") = false)
        .def("update_state", [](wj::KDLBackend& self,
                                py::array_t<double, py::array::c_style | py::array::forcecast> joint_values) {
            py::buffer_info info = joint_values.request();
            const auto* ptr = static_cast<const double*>(info.ptr);
            std::vector<double> q(ptr, ptr + info.size);
            wj::FKResult result = self.update_state(q);
            return py::make_tuple(make_1d_array(result.pos), make_rotmat_array(result.rotmat));
        }, py::arg("joint_values"))
        .def("motion_values_view", [](wj::KDLBackend& self) {
            return make_vector_view(self.motion_values(), py::cast(&self, py::return_value_policy::reference));
        })
        .def("gl_pos_0_view", [](wj::KDLBackend& self) {
            return make_matrix_view(self.gl_pos_0(), static_cast<py::ssize_t>(self.motion_values().size()), 3,
                                    py::cast(&self, py::return_value_policy::reference));
        })
        .def("gl_rotmat_0_view", [](wj::KDLBackend& self) {
            return py::array_t<double>(
                std::vector<py::ssize_t>{static_cast<py::ssize_t>(self.motion_values().size()), 3, 3},
                std::vector<py::ssize_t>{static_cast<py::ssize_t>(9 * sizeof(double)),
                                         static_cast<py::ssize_t>(3 * sizeof(double)),
                                         static_cast<py::ssize_t>(sizeof(double))},
                const_cast<double*>(self.gl_rotmat_0().data()),
                py::cast(&self, py::return_value_policy::reference));
        })
        .def("gl_motion_axis_view", [](wj::KDLBackend& self) {
            return make_matrix_view(self.gl_motion_axis(), static_cast<py::ssize_t>(self.motion_values().size()), 3,
                                    py::cast(&self, py::return_value_policy::reference));
        })
        .def("gl_pos_q_view", [](wj::KDLBackend& self) {
            return make_matrix_view(self.gl_pos_q(), static_cast<py::ssize_t>(self.motion_values().size()), 3,
                                    py::cast(&self, py::return_value_policy::reference));
        })
        .def("gl_rotmat_q_view", [](wj::KDLBackend& self) {
            return py::array_t<double>(
                std::vector<py::ssize_t>{static_cast<py::ssize_t>(self.motion_values().size()), 3, 3},
                std::vector<py::ssize_t>{static_cast<py::ssize_t>(9 * sizeof(double)),
                                         static_cast<py::ssize_t>(3 * sizeof(double)),
                                         static_cast<py::ssize_t>(sizeof(double))},
                const_cast<double*>(self.gl_rotmat_q().data()),
                py::cast(&self, py::return_value_policy::reference));
        })
        .def("gl_flange_pos_view", [](wj::KDLBackend& self) {
            return make_vector_view(self.gl_flange_pos(), py::cast(&self, py::return_value_policy::reference));
        })
        .def("gl_flange_rotmat_view", [](wj::KDLBackend& self) {
            return py::array_t<double>(
                std::vector<py::ssize_t>{3, 3},
                std::vector<py::ssize_t>{static_cast<py::ssize_t>(3 * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
                const_cast<double*>(self.gl_flange_rotmat().data()),
                py::cast(&self, py::return_value_policy::reference));
        })
        .def("ik", [](const wj::KDLBackend& self,
                      py::array_t<double, py::array::c_style | py::array::forcecast> target_pos,
                      py::array_t<double, py::array::c_style | py::array::forcecast> target_rotmat,
                      py::array_t<double, py::array::c_style | py::array::forcecast> seed_joint_values) -> py::object {
            const auto pos = read_array<3>(target_pos, "target_pos");
            const auto rotmat = read_array<9>(target_rotmat, "target_rotmat");
            py::buffer_info seed_info = seed_joint_values.request();
            const auto* seed_ptr = static_cast<const double*>(seed_info.ptr);
            std::vector<double> seed(seed_ptr, seed_ptr + seed_info.size);
            const auto result = self.ik(pos, rotmat, seed);
            if (!result.has_value()) {
                return py::none();
            }
            py::array_t<double> array({static_cast<py::ssize_t>(result->size())});
            auto out = array.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(result->size()); ++i) {
                out(i) = (*result)[static_cast<std::size_t>(i)];
            }
            return array;
        }, py::arg("target_pos"), py::arg("target_rotmat"), py::arg("seed_joint_values"));
}
