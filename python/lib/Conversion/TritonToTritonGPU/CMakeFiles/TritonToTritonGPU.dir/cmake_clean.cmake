file(REMOVE_RECURSE
  "libTritonToTritonGPU.a"
  "libTritonToTritonGPU.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TritonToTritonGPU.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
