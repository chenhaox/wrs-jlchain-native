cmake_minimum_required(VERSION 3.18)

project(wrs_jlchain_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

if(NOT EIGEN3_INCLUDE_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import includeigen; print(includeigen.get_include())"
        OUTPUT_VARIABLE EIGEN3_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()

if(NOT EIGEN3_INCLUDE_DIR OR NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen")
    message(FATAL_ERROR
        "Eigen headers were not found. Install includeigen or pass "
        "-DEIGEN3_INCLUDE_DIR=/path/to/eigen3.")
endif()

set(WRS_OROCOS_KDL_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../third_party/orocos_kinematics_dynamics/orocos_kdl"
    CACHE PATH "Path to the Orocos KDL source tree")

add_subdirectory("${WRS_OROCOS_KDL_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/orocos_kdl" EXCLUDE_FROM_ALL)

pybind11_add_module(_wrs_jlchain_native_kdl
    wrs_jlchain_native/kdl_backend.cpp
    wrs_jlchain_native/python_bindings.cpp)

target_include_directories(_wrs_jlchain_native_kdl
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/wrs_jlchain_native/include"
        "${WRS_OROCOS_KDL_DIR}/src"
        "${CMAKE_CURRENT_BINARY_DIR}/orocos_kdl/src"
        "${EIGEN3_INCLUDE_DIR}")

target_link_libraries(_wrs_jlchain_native_kdl PRIVATE orocos-kdl)

target_compile_definitions(_wrs_jlchain_native_kdl PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

add_custom_command(TARGET _wrs_jlchain_native_kdl POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${CMAKE_CURRENT_LIST_DIR}/python/wrs_jlchain_native"
        "$<TARGET_FILE_DIR:_wrs_jlchain_native_kdl>/wrs_jlchain_native")

install(TARGETS _wrs_jlchain_native_kdl
    LIBRARY DESTINATION "."
    RUNTIME DESTINATION ".")
