# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/scratch.isaacw_gpu/anaconda3/envs/conda_0/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/scratch.isaacw_gpu/anaconda3/envs/conda_0/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/scratch.isaacw_gpu/Code/triton

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/scratch.isaacw_gpu/Code/triton/python

# Utility rule file for install-TritonIR.

# Include any custom commands dependencies for this target.
include lib/Dialect/Triton/IR/CMakeFiles/install-TritonIR.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Dialect/Triton/IR/CMakeFiles/install-TritonIR.dir/progress.make

lib/Dialect/Triton/IR/CMakeFiles/install-TritonIR:
	cd /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR && /home/scratch.isaacw_gpu/anaconda3/envs/conda_0/lib/python3.10/site-packages/cmake/data/bin/cmake -DCMAKE_INSTALL_COMPONENT="TritonIR" -P /home/scratch.isaacw_gpu/Code/triton/python/cmake_install.cmake

install-TritonIR: lib/Dialect/Triton/IR/CMakeFiles/install-TritonIR
install-TritonIR: lib/Dialect/Triton/IR/CMakeFiles/install-TritonIR.dir/build.make
.PHONY : install-TritonIR

# Rule to build all files generated by this target.
lib/Dialect/Triton/IR/CMakeFiles/install-TritonIR.dir/build: install-TritonIR
.PHONY : lib/Dialect/Triton/IR/CMakeFiles/install-TritonIR.dir/build

lib/Dialect/Triton/IR/CMakeFiles/install-TritonIR.dir/clean:
	cd /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR && $(CMAKE_COMMAND) -P CMakeFiles/install-TritonIR.dir/cmake_clean.cmake
.PHONY : lib/Dialect/Triton/IR/CMakeFiles/install-TritonIR.dir/clean

lib/Dialect/Triton/IR/CMakeFiles/install-TritonIR.dir/depend:
	cd /home/scratch.isaacw_gpu/Code/triton/python && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/scratch.isaacw_gpu/Code/triton /home/scratch.isaacw_gpu/Code/triton/lib/Dialect/Triton/IR /home/scratch.isaacw_gpu/Code/triton/python /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR/CMakeFiles/install-TritonIR.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/Dialect/Triton/IR/CMakeFiles/install-TritonIR.dir/depend

