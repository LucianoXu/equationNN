
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

set(ENVBACKENDPY_NAME "envbackend")

set_target_properties(ENVBACKENDPY PROPERTIES OUTPUT_NAME ${ENVBACKENDPY_NAME})

install(TARGETS ENVBACKENDPY DESTINATION .)

target_link_libraries(
    ENVBACKENDPY
    PUBLIC
        ENVBACKEND
)

# Use stubgen to create .pyi files to sit alongside the just-built python module
add_custom_command(
    TARGET ENVBACKENDPY POST_BUILD
    COMMAND stubgen -m ${ENVBACKENDPY_NAME} -o .
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Use stubgen to create .pyi for statement completion..."
)