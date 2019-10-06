cmake_minimum_required(VERSION 2.8)
project( HarrisCorner LANGUAGES CUDA CXX)

find_package( OpenCV REQUIRED )

include_directories("${OpenCV_INCLUDE_DIRS}")

add_library(Harris STATIC
  HarrisCorner.h
  HarrisCorner.cpp
  CudaKernels.cu
  CudaKernels.h
 )
 
 # Request that Harris be built with -std=c++11
# As this is a public compile feature anything that links to Harris
# will also build with -std=c++11
target_compile_features(Harris PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the Harris
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( Harris
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
					  
add_executable(HarrisCorner_test main.cu)
set_target_properties(HarrisCorner_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(HarrisCorner_test PRIVATE Harris ${OpenCV_LIBS})