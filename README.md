
## Building vampire from source
See https://github.com/vprover/vampire/wiki/Source-Build-for-Users.

## extract python requirements.txt
```
pipreqs . --force --ignore vampire
```

## Compilation Requirements
- Compilation of the C++ backend requires a `clang++` compiler that supports the C++23 standard.
- ANLTR4 is included. You only need to have Java installed.
- The CMakeLists will find package of Python directly. To specify the Python installation, use the `cmake` flag `-DPYTHON_EXECUTABLE=...` to specify the `python3` executable.