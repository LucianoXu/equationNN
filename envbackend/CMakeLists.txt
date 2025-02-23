
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


pybind11_add_module(
    ENVBACKENDPY 
    
    pybind_interface.cpp
)

set_target_properties(ENVBACKENDPY PROPERTIES OUTPUT_NAME "envbackend")

install(TARGETS ENVBACKENDPY DESTINATION .)

target_link_libraries(
    ENVBACKENDPY
    PUBLIC
        ENVBACKEND
)