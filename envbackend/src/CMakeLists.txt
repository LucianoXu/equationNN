
set(ANTLR_PARSER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/antlr_generated")

set(GRAMMAR_NAME "ENVBACKEND")

set(ANTLR_PARSER_GEN_FILES 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}.interp 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}.tokens 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}BaseListener.cpp 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}BaseListener.h 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}BaseVisitor.cpp 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}BaseVisitor.h 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}Lexer.cpp 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}Lexer.h 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}Lexer.interp 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}Lexer.tokens 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}Listener.cpp 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}Listener.h 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}Parser.cpp 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}Parser.h 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}Visitor.cpp 
    ${ANTLR_PARSER_DIR}/${GRAMMAR_NAME}Visitor.h
)

# Custom command for ANTLR
add_custom_command(
    OUTPUT ${ANTLR_PARSER_GEN_FILES}
    COMMAND java -jar /Users/xuyingte/Java3P/antlr-4.13.2-complete.jar -Dlanguage=Cpp -visitor ${CMAKE_CURRENT_SOURCE_DIR}/${GRAMMAR_NAME}.g4 -o ${ANTLR_PARSER_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${GRAMMAR_NAME}.g4
    COMMENT "Running ANTLR to generate C++ code from ${GRAMMAR_NAME}.g4"
)

add_custom_target(antlr_parser_gen
    DEPENDS ${ANTLR_PARSER_GEN_FILES}
)


add_library(
    ENVBACKEND 

    algebra.cpp
    parser.cpp
    env.cpp

    ${ANTLR_PARSER_GEN_FILES}
)

target_link_libraries(
    ENVBACKEND
    PUBLIC
        DEPENDENCIES
)


target_include_directories(
    ENVBACKEND
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ANTLR_PARSER_DIR}
)

add_dependencies(ENVBACKEND antlr_parser_gen)
