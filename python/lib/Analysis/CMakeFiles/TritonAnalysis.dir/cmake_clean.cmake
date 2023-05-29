file(REMOVE_RECURSE
  "libTritonAnalysis.a"
  "libTritonAnalysis.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TritonAnalysis.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
