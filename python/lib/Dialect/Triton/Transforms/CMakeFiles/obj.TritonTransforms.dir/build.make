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
include lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/progress.make

# Include the compile flags for this target's objects.
include lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/flags.make

lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.o: lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/flags.make
lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.o: /home/scratch.isaacw_gpu/Code/triton/lib/Dialect/Triton/Transforms/Combine.cpp
lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.o: lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/scratch.isaacw_gpu/Code/triton/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.o"
	cd /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/Transforms && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.o -MF CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.o.d -o CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.o -c /home/scratch.isaacw_gpu/Code/triton/lib/Dialect/Triton/Transforms/Combine.cpp

lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.i"
	cd /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/Transforms && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/scratch.isaacw_gpu/Code/triton/lib/Dialect/Triton/Transforms/Combine.cpp > CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.i

lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.s"
	cd /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/Transforms && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/scratch.isaacw_gpu/Code/triton/lib/Dialect/Triton/Transforms/Combine.cpp -o CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.s

obj.TritonTransforms: lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/Combine.cpp.o
obj.TritonTransforms: lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/build.make
.PHONY : obj.TritonTransforms

# Rule to build all files generated by this target.
lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/build: obj.TritonTransforms
.PHONY : lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/build

lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/clean:
	cd /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/Transforms && $(CMAKE_COMMAND) -P CMakeFiles/obj.TritonTransforms.dir/cmake_clean.cmake
.PHONY : lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/clean

lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/depend:
	cd /home/scratch.isaacw_gpu/Code/triton/python && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/scratch.isaacw_gpu/Code/triton /home/scratch.isaacw_gpu/Code/triton/lib/Dialect/Triton/Transforms /home/scratch.isaacw_gpu/Code/triton/python /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/Transforms /home/scratch.isaacw_gpu/Code/triton/python/lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/Dialect/Triton/Transforms/CMakeFiles/obj.TritonTransforms.dir/depend

