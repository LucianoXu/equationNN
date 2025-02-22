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


        const Algebra& get_algebra() const {
            return algebra;
        }

        const std::vector<std::string>& get_rule_names() const {
            return rule_names;
        }

        const std::map<std::string, RewriteRule>& get_rules() const {
            return rules;
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

        /**
         * @brief Get the valid rule pos objects
         * 
         * @param eq 
         * @return std::vector<std::pair<std::string, TermPos>> 
         */
        std::vector<std::pair<std::string, TermPos>> get_valid_rule_pos(const equation& eq) const;
    };

    // /**
    //  * @brief The environment that directly deals with string representation of the algebraic objects.
    //  * 
    //  */
    // class Env {
    // private:
    //     SymbolKernel kernel;

    // };

    /**
     * @brief return the vocabulary for the algebra.
     * 
     * @param code 
     * @return std::vector<std::string>, the int binding for the variables are the index in the vector.
     */
    std::vector<std::string> get_vocab(const Algebra& algebra);

    /**
     * @brief The machine that predicts the next possible token in the algebraic manipulation.
     * Note that the machine always starts with the "<SOS>" token.
     * 
     */
    class NextTokenMachine {
    public:
        enum State {
            LHS,
            EQ,
            RHS,
            COLON,
            RULE_NAME,
            POS,
            SUBST,
            SUBST_TERM,
            SUBST_COLON,
            COMMA,
            EOS,
            HALT
        };
        enum Parenthesis {
            NONE,
            OPENING,
            CLOSING
        };

        struct CandidateTree {
            int choice;
            std::vector<CandidateTree> children;
        };

    private:
        const Algebra& algebra;
        const Signature& sig;
        SymbolKernel kernel;
        std::set<int> func_symbols;
        std::set<int> var_symbols;
        std::map<int, std::set<int>> required_vars_map;

        std::vector<std::string> vocab;
        std::map<std::string, int> vocab_map;
        std::vector<int> pos_int_map;

        std::vector<int> token_seq;
        std::set<int> valid_next_tokens;

        // States to determine the grammar stage
        State state;
        // The stack for the term generation. The first int is the total arity, the second int is the term constructed.
        std::stack<std::pair<int, int>> term_gen_stack;

        // The stack for the parenthesis
        Parenthesis parenthesis;

        // The list of the valid rewrite rule names & positions
        std::vector<CandidateTree> valid_rule_pos_tree;
        std::vector<CandidateTree>* p_current_rule_pos_tree;

        // The set of remaining variables to be specified in the substitution
        std::set<int> remaining_vars;

    private:

        void calculate_valid_next_tokens();

        void parse_valid_rule_pos_tree(std::vector<std::pair<std::string, TermPos>>&& valid_rule_pos);

    public:
        /**
         * @brief Construct a new Next Token Machine object and push the "<SOS>" token.
         * 
         * @param algebra 
         */
        NextTokenMachine(const Algebra& algebra);

        const std::set<int>& get_valid_next_tokens() const {
            return valid_next_tokens;
        }

        /**
         * @brief Push the token into the machine and update the state.
         * 
         * @param token 
         * @return bool, whether the token is valid and successfully pushed.
         */
        bool push_token(int token);

        bool push_token(std::string token) {
            if (vocab_map.find(token) == vocab_map.end()) {
                return false;
            }
            return push_token(vocab_map[token]);
        }

        std::string to_string() const;

        State get_state() const {
            return state;
        }
    };

} // namespace ualg