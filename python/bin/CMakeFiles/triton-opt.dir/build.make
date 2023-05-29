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
include bin/CMakeFiles/triton-opt.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include bin/CMakeFiles/triton-opt.dir/compiler_depend.make

# Include the progress variables for this target.
include bin/CMakeFiles/triton-opt.dir/progress.make

# Include the compile flags for this target's objects.
include bin/CMakeFiles/triton-opt.dir/flags.make

bin/CMakeFiles/triton-opt.dir/triton-opt.cpp.o: bin/CMakeFiles/triton-opt.dir/flags.make
bin/CMakeFiles/triton-opt.dir/triton-opt.cpp.o: /home/scratch.isaacw_gpu/Code/triton/bin/triton-opt.cpp
bin/CMakeFiles/triton-opt.dir/triton-opt.cpp.o: bin/CMakeFiles/triton-opt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/scratch.isaacw_gpu/Code/triton/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object bin/CMakeFiles/triton-opt.dir/triton-opt.cpp.o"
	cd /home/scratch.isaacw_gpu/Code/triton/python/bin && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT bin/CMakeFiles/triton-opt.dir/triton-opt.cpp.o -MF CMakeFiles/triton-opt.dir/triton-opt.cpp.o.d -o CMakeFiles/triton-opt.dir/triton-opt.cpp.o -c /home/scratch.isaacw_gpu/Code/triton/bin/triton-opt.cpp

bin/CMakeFiles/triton-opt.dir/triton-opt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/triton-opt.dir/triton-opt.cpp.i"
	cd /home/scratch.isaacw_gpu/Code/triton/python/bin && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/scratch.isaacw_gpu/Code/triton/bin/triton-opt.cpp > CMakeFiles/triton-opt.dir/triton-opt.cpp.i

bin/CMakeFiles/triton-opt.dir/triton-opt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/triton-opt.dir/triton-opt.cpp.s"
	cd /home/scratch.isaacw_gpu/Code/triton/python/bin && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/scratch.isaacw_gpu/Code/triton/bin/triton-opt.cpp -o CMakeFiles/triton-opt.dir/triton-opt.cpp.s

# Object files for target triton-opt
triton__opt_OBJECTS = \
"CMakeFiles/triton-opt.dir/triton-opt.cpp.o"

# External object files for target triton-opt
triton__opt_EXTERNAL_OBJECTS =

bin/triton-opt: bin/CMakeFiles/triton-opt.dir/triton-opt.cpp.o
bin/triton-opt: bin/CMakeFiles/triton-opt.dir/build.make
bin/triton-opt: lib/Analysis/libTritonAnalysis.a
bin/triton-opt: lib/Dialect/Triton/Transforms/libTritonTransforms.a
bin/triton-opt: lib/Dialect/TritonGPU/Transforms/libTritonGPUTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAffineAnalysis.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAffineDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAffineTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAffineUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAMDGPUDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArithmeticDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArithmeticTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArithmeticUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArmNeonDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArmSVEDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArmSVETransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAsyncDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAsyncTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAMXDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAMXTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRBufferizationDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRBufferizationTransformOps.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRBufferizationTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRComplexDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRControlFlowDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRDLTIDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIREmitCDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRFuncDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRFuncTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRGPUOps.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRGPUTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgAnalysis.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgTransformOps.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLLVMIRTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLLVMDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRNVVMDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRROCDLDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMathDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMathTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMemRefDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMemRefTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMemRefUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMLProgramDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRNVGPUDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRNVGPUTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIROpenACCDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIROpenMPDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRPDLDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRPDLInterpDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRQuantDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRQuantTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRQuantUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFTransformOps.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRShapeDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRShapeOpsTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSparseTensorDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSparseTensorTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSparseTensorPipelines.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSparseTensorUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVModuleCombiner.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVConversion.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTensorDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTensorInferTypeOpInterfaceImpl.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTensorTilingInterfaceImpl.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTensorTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTensorUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTosaDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTosaTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTransformDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTransformDialectTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRX86VectorDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRX86VectorTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTosaTestPasses.a
bin/triton-opt: lib/Dialect/Triton/IR/libTritonIR.a
bin/triton-opt: lib/Dialect/Triton/Transforms/libTritonTransforms.a
bin/triton-opt: lib/Dialect/TritonGPU/IR/libTritonGPUIR.a
bin/triton-opt: lib/Dialect/TritonGPU/Transforms/libTritonGPUTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAffineToStandard.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAMDGPUToROCDL.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArithmeticToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArithmeticToSPIRV.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArmNeon2dToIntr.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAsyncToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRBufferizationToMemRef.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRComplexToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRComplexToLibm.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRComplexToStandard.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRControlFlowToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRControlFlowToSPIRV.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRFuncToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRFuncToSPIRV.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRGPUToGPURuntimeTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRGPUToNVVMTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRGPUToROCDLTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRGPUToSPIRV.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRGPUToVulkanTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgToSPIRV.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgToStandard.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLLVMCommonConversion.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMathToLibm.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMathToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMathToSPIRV.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMemRefToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMemRefToSPIRV.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRNVGPUToNVVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIROpenACCToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIROpenACCToSCF.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIROpenMPToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRPDLToPDLInterp.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRReconcileUnrealizedCasts.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFToControlFlow.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFToGPU.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFToOpenMP.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFToSPIRV.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRShapeToStandard.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTensorToLinalg.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTensorToSPIRV.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTosaToArith.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTosaToLinalg.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTosaToSCF.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTosaToTensor.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorToGPU.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorToSCF.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorToSPIRV.a
bin/triton-opt: lib/Conversion/TritonToTritonGPU/libTritonToTritonGPU.a
bin/triton-opt: lib/Conversion/TritonGPUToLLVM/libTritonGPUToLLVM.a
bin/triton-opt: test/lib/Analysis/libTritonTestAnalysis.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIROptLib.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRPass.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAMDGPUToROCDL.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMemRefToSPIRV.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVSerialization.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVBinaryUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArithmeticToSPIRV.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRFuncToSPIRV.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRGPUToNVVMTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRGPUToGPURuntimeTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAsyncToLLVM.a
bin/triton-opt: lib/Analysis/libTritonAnalysis.a
bin/triton-opt: lib/Dialect/Triton/Transforms/libTritonTransforms.a
bin/triton-opt: lib/Dialect/TritonGPU/Transforms/libTritonGPUTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAffineTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAMDGPUDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAsyncTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRBufferizationTransformOps.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIREmitCDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRGPUTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAsyncDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRExecutionEngineUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMPasses.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMCoroutines.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMipo.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMVectorize.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMIRReader.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMLinker.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMInstrumentation.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMObjCARCOpts.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMTarget.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLLVMToLLVMIRTranslation.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgTransformOps.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRROCDLDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMathTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMemRefTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMemRefUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMLProgramDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRNVGPUTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRNVGPUDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIROpenACCDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIROpenMPDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRQuantTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFTransformOps.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRShapeOpsTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRShapeDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSparseTensorPipelines.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSparseTensorTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRFuncTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArithmeticTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgAnalysis.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRFuncToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArithmeticToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRControlFlowToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorToSCF.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAffineToStandard.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRComplexToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRComplexToLibm.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRComplexToStandard.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMathToLibm.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMathToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMemRefToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRReconcileUnrealizedCasts.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFToControlFlow.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorToLLVM.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArmNeonDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArmSVETransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArmSVEDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAMXTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAMXDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTargetLLVMIRExport.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLLVMIRTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRNVVMDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTranslateLib.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMFrontendOpenMP.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMScalarOpts.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMAggressiveInstCombine.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMInstCombine.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMTransformUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSparseTensorUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVModuleCombiner.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVConversion.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSPIRVDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTensorInferTypeOpInterfaceImpl.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTensorTilingInterfaceImpl.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTensorTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTensorUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTosaTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTransformDialectTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTransformDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAffineUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRBufferizationTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRGPUOps.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRDLTIDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLinalgDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMathDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTilingInterface.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAffineAnalysis.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRPresburger.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRX86VectorTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRVectorInterfaces.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRX86VectorDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLLVMCommonConversion.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLLVMDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMAsmParser.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMBitWriter.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMAnalysis.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMProfileData.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMSymbolize.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMDebugInfoPDB.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMDebugInfoMSF.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMDebugInfoDWARF.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMObject.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMBitReader.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMMCParser.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMMC.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMDebugInfoCodeView.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMTextAPI.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTransforms.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRCopyOpInterface.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTosaTestPasses.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTosaDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRQuantUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRQuantDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTransformUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRRewrite.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRPDLToPDLInterp.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRPDLInterpDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRPDLDialect.a
bin/triton-opt: lib/Dialect/TritonGPU/IR/libTritonGPUIR.a
bin/triton-opt: lib/Dialect/Triton/IR/libTritonIR.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSCFDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRBufferizationDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAffineDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRFuncDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRMemRefDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSparseTensorDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRTensorDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArithmeticUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRComplexDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRCastInterfaces.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMCore.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMBinaryFormat.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMRemarks.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMBitstreamReader.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRDialectUtils.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRParallelCombiningOpInterface.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRControlFlowDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRArithmeticDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRDialect.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRPass.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAnalysis.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRCallInterfaces.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRControlFlowInterfaces.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRInferTypeOpInterface.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRLoopLikeInterface.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSideEffectInterfaces.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRInferIntRangeInterface.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRDataLayoutInterfaces.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRViewLikeInterface.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRParser.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRAsmParser.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRIR.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libMLIRSupport.a
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMSupport.a
bin/triton-opt: /usr/lib/x86_64-linux-gnu/libz.so
bin/triton-opt: /home/scratch.isaacw_gpu/download/llvm/install/lib/libLLVMDemangle.a
bin/triton-opt: bin/CMakeFiles/triton-opt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/scratch.isaacw_gpu/Code/triton/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable triton-opt"
	cd /home/scratch.isaacw_gpu/Code/triton/python/bin && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/triton-opt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bin/CMakeFiles/triton-opt.dir/build: bin/triton-opt
.PHONY : bin/CMakeFiles/triton-opt.dir/build

bin/CMakeFiles/triton-opt.dir/clean:
	cd /home/scratch.isaacw_gpu/Code/triton/python/bin && $(CMAKE_COMMAND) -P CMakeFiles/triton-opt.dir/cmake_clean.cmake
.PHONY : bin/CMakeFiles/triton-opt.dir/clean

bin/CMakeFiles/triton-opt.dir/depend:
	cd /home/scratch.isaacw_gpu/Code/triton/python && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/scratch.isaacw_gpu/Code/triton /home/scratch.isaacw_gpu/Code/triton/bin /home/scratch.isaacw_gpu/Code/triton/python /home/scratch.isaacw_gpu/Code/triton/python/bin /home/scratch.isaacw_gpu/Code/triton/python/bin/CMakeFiles/triton-opt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bin/CMakeFiles/triton-opt.dir/depend

