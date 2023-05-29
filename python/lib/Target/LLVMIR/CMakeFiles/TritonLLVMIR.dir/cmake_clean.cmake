file(REMOVE_RECURSE
  "libTritonLLVMIR.a"
  "libTritonLLVMIR.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TritonLLVMIR.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
