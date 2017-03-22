## cmake .
## make
cmake_minimum_required(VERSION 2.8)
project( KNNFaceRec )
find_package( OpenCV REQUIRED )
add_executable( tst blog_main.cpp )
target_link_libraries( tst ${OpenCV_LIBS})
