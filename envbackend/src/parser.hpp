/*
*/

#pragma once

#include "antlr4-runtime.h"

#include "ENVBACKENDLexer.h"
#include "ENVBACKENDParser.h"
#include "ENVBACKENDBaseListener.h"

#include <stack>

#include "algebra.hpp"
#include "env.hpp"

namespace ualg {
    /**
     * @brief Parse the given code.
     * 
     * @param code 
     * @return std::optional<equation>
     */
    std::optional<equation> parse_equation(const std::string& code);

    /**
     * @brief Parse the given code.
     * 
     * @param code 
     * @return std::optional<TermPtr>
     */
    std::optional<TermPtr> parse_term(const std::string& code);

    /**
     * @brief Parse the given code.
     * 
     * @param code 
     * @return std::optional<TermPos> 
     */
    std::optional<TermPos> parse_pos(const std::string& code);

    /**
     * @brief Parse the given code.
     * 
     * @param code 
     * @return std::optional<subst> 
     */
    std::optional<subst> parse_subst(const std::string& code);

    /**
     * @brief Parse the given code.
     * 
     * @param code 
     * @return std::optional<Algebra> 
     */
    std::optional<Signature> parse_signature(const std::string& code);


    /**
     * @brief Parse the given code.
     * 
     * @param code 
     * @return std::optional<Algebra> 
     */
    std::optional<Algebra> parse_alg(const std::string& code);

    /**
     * @brief Parse the given code.
     * 
     * @param code 
     * @return std::optional<proof_action> 
     */
    std::optional<proof_action> parse_proof_action(const std::string& code);

} // namespace ualg