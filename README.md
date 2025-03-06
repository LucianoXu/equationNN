# ML/AI for Algebraic Equational Reasonings
It is a machine learning project written in Python, with C++ backend for the universal algebra, which is wrapped as a Python library using `pybind11`.

## Installation and Requirements

### Prerequisite and Dependencies
- Compilation of the C++ backend requires a compiler that supports the C++23 standard.
- Java installation to generate the ANLTR4 parser file.
- Python installtion version 3.8 or later.
- CUDA Toolkit installation.
- CMake installation, version 3.10 or later.
- ELab (included as submodule): a machine learning training manager by Yingte.

### Installation
1. Get the vampire solver and add it to the PATH. See https://github.com/vprover/vampire/wiki/Source-Build-for-Users.
For linux system, they also provide compiled binaries as releases.

2. Clone the repository using 
```
git clone --recurse-submodules https://github.com/LucianoXu/equationNN.git
```
navigate to the project root folder, and check out the `main` branch.

3. Install the Python library dependencies using
```
pip install -r requirements.txt
```
Install the `elab` library in the editable way using
```
pip install -e elab
```

4. Compile the C++ backend. You can press `Cmd+Shift+B` and run the *Compile All Versions* task. It automatically creates the *build* directory and compiles the Python library. The compilation requires to link to the Python installation where the later experiments carries out. The CMakeLists will find package of Python directly. To specify the Python installation, use the `cmake` flag `-DPYTHON_EXECUTABLE=...` to specify the `python3` executable.

5. Run the tests to validate installation. For C++ tests, execute
```
ctest --test-dir build/Debug
ctest --test-dir build/Release
```
For python tests, execute
```
pytest
```
And if all tests are passed successfully, it should mean that the installation succeeds.
The tests can also be initiated by the *testing* sidebar in VSCode.

## Experiments

The `main` python script at the project root folder act as the main entry of different kinds of experiments. 
The script accepts extra command line arguments to specify the experiment (task) to execute and corresponding arguments.
See the task scripts to know the command line arguments. In the `main` entry, there is also an example command under each task.

## VSCode configurations
The .vscode directory contains the configurations for compilation/debugging within VSCode.


## Code of Contribute
Work on you on branch and merge it into `main` before pushing. Don't work on `main` directly.
Follow the structure in `main` and `pysrc/tasks` to archive the valuable experiments you composed.


## Others

### extract python requirements.txt
```
pipreqs . --force --ignore vampire
```