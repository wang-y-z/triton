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

# Include any dependencies generated for this target.
include lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/progress.make

# Include the compile flags for this target's objects.
include lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/flags.make

# Object files for target TritonIR
TritonIR_OBJECTS =

# External object files for target TritonIR
TritonIR_EXTERNAL_OBJECTS = \
"/home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR/CMakeFiles/obj.TritonIR.dir/Interfaces.cpp.o" \
"/home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR/CMakeFiles/obj.TritonIR.dir/Dialect.cpp.o" \
"/home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR/CMakeFiles/obj.TritonIR.dir/Ops.cpp.o" \
"/home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR/CMakeFiles/obj.TritonIR.dir/Types.cpp.o" \
"/home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR/CMakeFiles/obj.TritonIR.dir/Traits.cpp.o"

lib/Dialect/Triton/IR/libTritonIR.a: lib/Dialect/Triton/IR/CMakeFiles/obj.TritonIR.dir/Interfaces.cpp.o
lib/Dialect/Triton/IR/libTritonIR.a: lib/Dialect/Triton/IR/CMakeFiles/obj.TritonIR.dir/Dialect.cpp.o
lib/Dialect/Triton/IR/libTritonIR.a: lib/Dialect/Triton/IR/CMakeFiles/obj.TritonIR.dir/Ops.cpp.o
lib/Dialect/Triton/IR/libTritonIR.a: lib/Dialect/Triton/IR/CMakeFiles/obj.TritonIR.dir/Types.cpp.o
lib/Dialect/Triton/IR/libTritonIR.a: lib/Dialect/Triton/IR/CMakeFiles/obj.TritonIR.dir/Traits.cpp.o
lib/Dialect/Triton/IR/libTritonIR.a: lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/build.make
lib/Dialect/Triton/IR/libTritonIR.a: lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/scratch.isaacw_gpu/Code/triton/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library libTritonIR.a"
	cd /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR && $(CMAKE_COMMAND) -P CMakeFiles/TritonIR.dir/cmake_clean_target.cmake
	cd /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TritonIR.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/build: lib/Dialect/Triton/IR/libTritonIR.a
.PHONY : lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/build

lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/clean:
	cd /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR && $(CMAKE_COMMAND) -P CMakeFiles/TritonIR.dir/cmake_clean.cmake
.PHONY : lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/clean

lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/depend:
	cd /home/scratch.isaacw_gpu/Code/triton/python && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/scratch.isaacw_gpu/Code/triton /home/scratch.isaacw_gpu/Code/triton/lib/Dialect/Triton/IR /home/scratch.isaacw_gpu/Code/triton/python /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/depend

