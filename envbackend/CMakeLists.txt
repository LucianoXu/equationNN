
add_subdirectory(third-party)

add_subdirectory(src)

add_subdirectory(test)


add_executable(
    MAIN 

    main.cpp
)

target_link_libraries(
    MAIN
    PUBLIC
        ENVBACKEND
)