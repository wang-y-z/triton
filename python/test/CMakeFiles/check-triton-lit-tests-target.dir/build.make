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

# Utility rule file for check-triton-lit-tests-target.

# Include any custom commands dependencies for this target.
include test/CMakeFiles/check-triton-lit-tests-target.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/check-triton-lit-tests-target.dir/progress.make

test/CMakeFiles/check-triton-lit-tests-target:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/scratch.isaacw_gpu/Code/triton/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running lit suite /home/scratch.isaacw_gpu/Code/triton/test/Target"
	cd /home/scratch.isaacw_gpu/Code/triton/python/test && /home/scratch.isaacw_gpu/anaconda3/envs/conda_0/bin/python3.10 /home/scratch.isaacw_gpu/anaconda3/envs/conda_0/bin/lit -sv /home/scratch.isaacw_gpu/Code/triton/test/Target

check-triton-lit-tests-target: test/CMakeFiles/check-triton-lit-tests-target
check-triton-lit-tests-target: test/CMakeFiles/check-triton-lit-tests-target.dir/build.make
.PHONY : check-triton-lit-tests-target

# Rule to build all files generated by this target.
test/CMakeFiles/check-triton-lit-tests-target.dir/build: check-triton-lit-tests-target
.PHONY : test/CMakeFiles/check-triton-lit-tests-target.dir/build

test/CMakeFiles/check-triton-lit-tests-target.dir/clean:
	cd /home/scratch.isaacw_gpu/Code/triton/python/test && $(CMAKE_COMMAND) -P CMakeFiles/check-triton-lit-tests-target.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/check-triton-lit-tests-target.dir/clean

test/CMakeFiles/check-triton-lit-tests-target.dir/depend:
	cd /home/scratch.isaacw_gpu/Code/triton/python && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/scratch.isaacw_gpu/Code/triton /home/scratch.isaacw_gpu/Code/triton/test /home/scratch.isaacw_gpu/Code/triton/python /home/scratch.isaacw_gpu/Code/triton/python/test /home/scratch.isaacw_gpu/Code/triton/python/test/CMakeFiles/check-triton-lit-tests-target.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/check-triton-lit-tests-target.dir/depend

