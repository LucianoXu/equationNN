#################################
# Tests list
set(tests
    test_ualg
)

foreach(test ${tests})
    add_executable(${test} ${test}.cpp)

    target_link_libraries(
        ${test}
        gtest_main ENVBACKEND
    )

    add_test(
        NAME ${test}
        COMMAND ${test}
    )
endforeach()