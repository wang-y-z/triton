file(REMOVE_RECURSE
  "libTritonPTX.a"
  "libTritonPTX.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TritonPTX.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
