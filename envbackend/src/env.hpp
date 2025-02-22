/*
    The file that defines the environment for the algebraic manipulation of equations.
 */

#pragma once

namespace ualg {

    struct proof_action {
        std::string rule_name;
        TermPos pos;
        subst spec_subst;
    };

    struct proof_step {
        equation eq;
        proof_action act;
    };

    enum ACT_RESULT {
        FAILURE,
        SUCCESS
    };

    class SymbolKernel {
    private:
        // The algebra
        Algebra algebra;

        const Signature& sig;

        // The compiled rewriting rules
        std::vector<std::string> rule_names;
        std::map<std::string, RewriteRule> rules;
    public:
        SymbolKernel(const Algebra& _algebra);

        const std::vector<std::string>& get_rule_names() const {
            return rule_names;
        }


        /**
         * @brief Perform the action on the equation (state). It will modify the equation according to the rule, and return the result.
         * 
         * @param eq 
         * @param rule_name 
         * @param pos 
         * @param spec_subst 
         * @return ACT_RESULT 
         */
        ACT_RESULT action(equation& eq, const proof_action& act);
    };

    // /**
    //  * @brief The environment that directly deals with string representation of the algebraic objects.
    //  * 
    //  */
    // class Env {
    // private:
    //     SymbolKernel kernel;

    // };

    // ACT_RESULT action(const SymbolKernel& kernel, string code);

} // namespace ualg