/*
*/

#pragma once

#include "antlr4-runtime.h"

#include "ENVBACKENDLexer.h"
#include "ENVBACKENDParser.h"
#include "ENVBACKENDBaseListener.h"

#include <stack>

#include "algebra.hpp"

namespace ualg {

    /**
     * @brief Parse the given code.
     * 
     * @param code 
     * @return TermPtr
     */
    std::optional<TermPtr> parse_term(const std::string& code);

    /**
     * @brief Parse the given code.
     * 
     * @param code 
     * @return std::optional<Algebra> 
     */
    std::optional<Algebra> parse_alg(const std::string& code);

} // namespace ualg