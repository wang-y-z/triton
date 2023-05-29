file(REMOVE_RECURSE
  "TritonCombine.inc"
  "libTritonTransforms.a"
  "libTritonTransforms.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TritonTransforms.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
