file(REMOVE_RECURSE
  "libTritonGPUTransforms.a"
  "libTritonGPUTransforms.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TritonGPUTransforms.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
