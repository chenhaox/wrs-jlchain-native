# WRS JLChain Native

Native FK, Jacobian, IK, and runtime-state acceleration for WRS
`JLChain`.

The package keeps the WRS Python API intact while moving the hot kinematics
work into a C++ backend. The first backend is implemented with Orocos KDL; the
Python adapter is structured so additional backends such as Pinocchio or RBDL
can be added later without changing robot code.

## Package Names

- GitHub repository: `wrs-jlchain-native`
- PyPI package: `wrs-jlchain-native`
- Python import package: `wrs_jlchain_native`
- Native extension: `_wrs_jlchain_native_kdl`

## Requirements

- Python 3.10 or newer
- NumPy
- A WRS checkout when using the accelerated drop-in `JLChain`
- A C++17 compiler for source builds
- CMake and pybind11 for source builds

Wheel builds use `includeigen` to provide Eigen headers and build the bundled
Orocos KDL source from `third_party/orocos_kinematics_dynamics`.

## Repository Layout

```text
src/
  CMakeLists.txt
  python/wrs_jlchain_native/
    __init__.py
    backends.py
    jlchain.py
  wrs_jlchain_native/
    kdl_backend.cpp
    python_bindings.cpp
    include/wrs_jlchain_native/
      backend.hpp
      kdl_backend.hpp
```

## Install From PyPI

After publishing:

```powershell
python -m pip install wrs-jlchain-native
```

Then use the native-backed chain directly:

```python
from wrs_jlchain_native import JLChain

jlc = JLChain(n_dof=6, backend_name="kdl")
jlc.finalize()
```

This package is intentionally kept outside the WRS source tree. To integrate it
with a WRS robot class, import `wrs_jlchain_native.JLChain` in the WRS-side
module that constructs the robot's kinematic chain, or add a small compatibility
shim in your WRS checkout that aliases this class.

```python
from wrs_jlchain_native import JLChain as NativeJLChain

chain = NativeJLChain(n_dof=6, backend_name="kdl")
```

## Runtime Selection

Per chain:

```python
chain = JLChain(n_dof=6, backend_name="kdl")
```

Globally:

```powershell
$env:WRS_JLCHAIN_BACKEND="kdl"
```

Supported values:

- `auto`: try the native backend and fall back to Python
- `kdl`, `orocos`, `orocos_kdl`: require the Orocos KDL backend
- `python`, `none`, `off`: force the original Python implementation

## Build From Source

Initialize submodules first:

```powershell
git submodule update --init --recursive
```

Build a wheel:

```powershell
python -m pip install build
python -m build --sdist --wheel
```

Manual CMake build:

```powershell
cmake -S src -B build\jlchain_native `
  -DCMAKE_PREFIX_PATH=D:\code\venv312\.venv\Lib\site-packages\pybind11\share\cmake\pybind11 `
  -DPYTHON_EXECUTABLE=D:\code\venv312\.venv\Scripts\python.exe

cmake --build build\jlchain_native --config Release
```

If Eigen is not discovered through `includeigen`, pass it explicitly:

```powershell
-DEIGEN3_INCLUDE_DIR=D:\code\vcpkg\packages\eigen3_x64-windows\include\eigen3
```

For local WRS testing without installing a wheel, put the build output on
`PYTHONPATH`:

```powershell
$env:PYTHONPATH="D:\code\c_implementation_backend\build\jlchain_native\Release;$env:PYTHONPATH"
```

## Benchmark

The benchmark used during development lives in the separate WRS checkout and is
not committed to this standalone package repository. From a WRS workspace that
contains that benchmark, add the native build output or installed wheel to
`PYTHONPATH` and run:

```powershell
D:\code\venv312\.venv\Scripts\python.exe wrs\0000_test_programs\jlchain_speed_benchmark.py
```

It reports FK, FK plus Jacobian, direct Jacobian, FK state update, update plus
Jacobian, and IK timings, along with numerical error checks.

## CI And Publishing

GitHub Actions workflows are in `.github/workflows`:

- `ci.yml`: builds source distribution and wheels on pull requests and pushes
- `publish.yml`: builds distributions and publishes to PyPI on GitHub releases

PyPI publishing uses Trusted Publishing. Configure the PyPI publisher with:

- repository: `<owner>/wrs-jlchain-native`
- workflow: `publish.yml`
- environment: `pypi`

## License

This package links against/builds Orocos KDL, which is LGPL-2.1-or-later. The
package metadata therefore uses `LGPL-2.1-or-later`.
