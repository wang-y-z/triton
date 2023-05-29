file(REMOVE_RECURSE
  "libTritonGPUIR.a"
  "libTritonGPUIR.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TritonGPUIR.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
