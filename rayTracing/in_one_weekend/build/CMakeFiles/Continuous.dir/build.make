# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "E:\Program Files\cmake-3.16.4-win32-x86\bin\cmake.exe"

# The command to remove a file.
RM = "E:\Program Files\cmake-3.16.4-win32-x86\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = E:\Graphics\Practice\rayTracing\in_one_weekend

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = E:\Graphics\Practice\rayTracing\in_one_weekend\build

# Utility rule file for Continuous.

# Include the progress variables for this target.
include CMakeFiles/Continuous.dir/progress.make

CMakeFiles/Continuous:
	"E:\Program Files\cmake-3.16.4-win32-x86\bin\ctest.exe" -D Continuous

Continuous: CMakeFiles/Continuous
Continuous: CMakeFiles/Continuous.dir/build.make

.PHONY : Continuous

# Rule to build all files generated by this target.
CMakeFiles/Continuous.dir/build: Continuous

.PHONY : CMakeFiles/Continuous.dir/build

CMakeFiles/Continuous.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Continuous.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Continuous.dir/clean

CMakeFiles/Continuous.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" E:\Graphics\Practice\rayTracing\in_one_weekend E:\Graphics\Practice\rayTracing\in_one_weekend E:\Graphics\Practice\rayTracing\in_one_weekend\build E:\Graphics\Practice\rayTracing\in_one_weekend\build E:\Graphics\Practice\rayTracing\in_one_weekend\build\CMakeFiles\Continuous.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Continuous.dir/depend
