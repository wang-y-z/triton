file(REMOVE_RECURSE
  "libTritonGPUToLLVM.a"
  "libTritonGPUToLLVM.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TritonGPUToLLVM.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
