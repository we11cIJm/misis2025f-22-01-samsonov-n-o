cmake_minimum_required(VERSION 3.15)
project(walnut_coursework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

include(GNUInstallDirs)

find_package(OpenCV REQUIRED)

set(TARGETS
        walnut_v1
        walnut_v2
        walnut_metrics
        walnut_error_map
)

foreach(target IN LISTS TARGETS)
  add_executable(${target} src/${target}.cpp)
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
endforeach()

if(CMAKE_CONFIGURATION_TYPES)
  foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${cfg} cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR}/${cfg})
  endforeach()
else()
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

install(TARGETS ${TARGETS}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES README.md DESTINATION . OPTIONAL)

install(CODE [[
    file(MAKE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/results")
]])