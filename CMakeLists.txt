cmake_minimum_required(VERSION 3.15)
project(walnut_coursework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV REQUIRED)

set(WALNUT_SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(WALNUT_RESULTS_DIR ${CMAKE_SOURCE_DIR}/results)
set(WALNUT_GT_DIR ${CMAKE_SOURCE_DIR}/gt_final)
set(WALNUT_DATASET_DIR ${CMAKE_SOURCE_DIR}/dataset/Walnut2/Reconstructions)
set(WALNUT_REPORT_DIR ${CMAKE_SOURCE_DIR}/report)

foreach(target walnut_v1 walnut_v2 walnut_metrics walnut_error_map)
  add_executable(${target} src/${target}.cpp)
  target_include_directories(${target} PRIVATE ${WALNUT_SRC_DIR})
  target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
endforeach()

add_custom_target(coursework_pipeline
  COMMAND ${CMAKE_COMMAND} -E make_directory ${WALNUT_RESULTS_DIR}/v1
  COMMAND ${CMAKE_COMMAND} -E make_directory ${WALNUT_RESULTS_DIR}/v2
  COMMAND ${CMAKE_COMMAND} -E make_directory ${WALNUT_REPORT_DIR}/tables
  COMMAND $<TARGET_FILE:walnut_v1>
          --dataset ${WALNUT_DATASET_DIR}
          --gt-dir ${WALNUT_GT_DIR}
          --results ${WALNUT_RESULTS_DIR}/v1
  COMMAND $<TARGET_FILE:walnut_metrics>
          --pred-root ${WALNUT_RESULTS_DIR}/v1
          --gt-root ${WALNUT_GT_DIR}
          --csv ${WALNUT_RESULTS_DIR}/v1/metrics.csv
          --tex-out ${WALNUT_REPORT_DIR}/tables/v1_metrics.tex
  COMMAND $<TARGET_FILE:walnut_error_map>
          --pred-root ${WALNUT_RESULTS_DIR}/v1
          --gt-root ${WALNUT_GT_DIR}
          --out-root ${WALNUT_RESULTS_DIR}/v1/error_maps
  COMMAND $<TARGET_FILE:walnut_v2>
          --dataset ${WALNUT_DATASET_DIR}
          --gt-dir ${WALNUT_GT_DIR}
          --results ${WALNUT_RESULTS_DIR}/v2
  COMMAND $<TARGET_FILE:walnut_metrics>
          --pred-root ${WALNUT_RESULTS_DIR}/v2
          --gt-root ${WALNUT_GT_DIR}
          --csv ${WALNUT_RESULTS_DIR}/v2/metrics.csv
          --tex-out ${WALNUT_REPORT_DIR}/tables/v2_metrics.tex
  COMMAND $<TARGET_FILE:walnut_error_map>
          --pred-root ${WALNUT_RESULTS_DIR}/v2
          --gt-root ${WALNUT_GT_DIR}
          --out-root ${WALNUT_RESULTS_DIR}/v2/error_maps
  DEPENDS walnut_v1 walnut_v2 walnut_metrics walnut_error_map
  VERBATIM
)

find_program(PDFLATEX_EXECUTABLE pdflatex)
if(PDFLATEX_EXECUTABLE)
  add_custom_target(coursework_report_pdf
    COMMAND ${PDFLATEX_EXECUTABLE} -interaction=nonstopmode -halt-on-error report.tex
    COMMAND ${PDFLATEX_EXECUTABLE} -interaction=nonstopmode -halt-on-error report.tex
    WORKING_DIRECTORY ${WALNUT_REPORT_DIR}
    DEPENDS coursework_pipeline
    VERBATIM
  )
endif()
