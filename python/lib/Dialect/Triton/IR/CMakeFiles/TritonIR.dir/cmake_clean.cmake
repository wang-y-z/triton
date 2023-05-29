file(REMOVE_RECURSE
  "libTritonIR.a"
  "libTritonIR.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TritonIR.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
