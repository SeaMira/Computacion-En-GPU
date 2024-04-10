if(BUILD_TESTING)
  set(MATRIXLIB_TESTS 
    "${PROJECT_SOURCE_DIR}/test/Matrix.cc"
  )
  add_executable(${PROJECT_NAME}Tests ${MATRIXLIB_TESTS})
  target_link_libraries(${PROJECT_NAME}Tests PRIVATE GTest::gtest_main ${PROJECT_NAME})

  include(GoogleTest)
  gtest_discover_tests(${PROJECT_NAME}Tests)
endif()