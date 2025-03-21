/*
    The file that defines the environment for the algebraic manipulation of equations.
 */

#pragma once

#include <stack>
#include "algebra.hpp"

namespace ualg {

    struct proof_state {
        equation eq;

        proof_state() {}

        proof_state(const equation& _eq) : eq(_eq) {}

        proof_state(const proof_state& other) : eq(other.eq) {}
        
        std::string to_string() const {
            return "<STT> " + eq.to_string() + " </STT>";
        }

        std::string to_repr() const {
            return "<STT " + to_string() + ">";
        }

        bool operator==(const proof_state& other) const {
            return eq == other.eq;
        }
    };

    struct proof_action {
        std::string rule_name;
        TermPos pos;
        subst spec_subst;

        std::string to_string() const {
            std::string content;
            if (rule_name == "SUBST") {
                content = "SUBST " + spec_subst.begin()->first + " " + spec_subst.begin()->second->to_string();
            }
            else {
                content = rule_name + " (";
                for (const auto& p : pos) {
                    content += std::to_string(p) + " ";
                }
                content += ") ";
                content += ualg::to_string(spec_subst);
            }
            return "<ACT> " + content + " </ACT>";
        }

        std::string to_repr() const {
            return "<ACT " + to_string() + ">";
        }
    };

    struct proof_step {
        proof_state stt;
        proof_action act;

        std::string to_string() const {
            return stt.to_string() + act.to_string();
        }

        std::string to_repr() const {
            return "<STEP " + to_string() + ">";
        }
    };

    enum ACT_RESULT {
        INVALID,    // incorrect format
        FAILURE,
        SUCCESS
    };

    /**
     * @brief Check whether the equation is symmetric according to the signature.
     */
    bool check_symm_axiom(const Signature& sig, const equation& eq);


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
         * @param stt 
         * @param rule_name 
         * @param pos 
         * @param spec_subst 
         * @return ACT_RESULT 
         */
        ACT_RESULT action(proof_state& stt, const proof_action& act) const;

        /**
         * @brief Perform the action on the equation (state) by the code. It will modify the equation according to the rule, and return the result.
         * 
         * @throw noexcept
         */
        ACT_RESULT action_by_code(proof_state& stt, const std::string& action_code) const;

        /**
         * @brief Get the valid rule pos objects
         * 
         * @param eq 
         * @return std::vector<std::pair<std::string, TermPos>> 
         */
        std::vector<std::pair<std::string, TermPos>> get_valid_rule_pos(const proof_state& stt) const;
    };




    /**
     * @brief The tokenizer class. It is generated according to the algebra, and can transform between the string and the int.
     */
    class Tokenizer {
    private:
        const Algebra& algebra;
        const Signature& sig;
        std::vector<std::string> vocab;
        std::map<std::string, int> vocab_map;
        std::vector<int> pos_int_map;

    public:
        Tokenizer(const Algebra& _algebra);

        const Algebra& get_algebra() const {
            return algebra;
        }

        const std::vector<std::string>& get_vocab() const {
            return vocab;
        }

        int get_vocab_size() const {
            return vocab.size();
        }

        int get_encoding(const std::string& str) const {
            auto find_res = vocab_map.find(str);
            if (find_res == vocab_map.end()) {
                throw std::runtime_error("Invalid token representation: " + str);
            }
            return vocab_map.at(str);
        }

        std::string get_token(int token) const {
            if (token < 0 || token >= vocab.size()) {
                throw std::runtime_error("Invalid token number: " + std::to_string(token));
            }
            return vocab[token];
        }

        int get_pos_int_encoding(int pos) const {
            if (pos < 0 || pos >= pos_int_map.size()) {
                throw std::runtime_error("Invalid position number: " + std::to_string(pos));
            }
            return pos_int_map[pos];
        }

        bool is_valid_token(std::string token) const {
            return vocab_map.find(token) != vocab_map.end();
        }

        std::vector<int> encode(const std::string& code) const;

        std::string decode(const std::vector<int>& encoding) const;
    };

    /**
     * @brief The function that checks whether the step is valid.
     * 
     * @param kernel
     * @param code
     */
    bool check_step(const SymbolKernel& kernel, std::string code);

    /**
     * @brief The machine that predicts the next possible token in the algebraic manipulation.
     * Note that the machine always starts with the "<SOS>" token.
     * 
     */
    class NextTokenMachine {
    public:
        enum State {
            START_STT,
            LHS,
            EQ,
            RHS,
            END_STT,

            START_ACT,
            ACT_NAME,
            // The rule branch
            POS,
            SUBST,
            SUBST_TERM,
            SUBST_COLON,
            COMMA,
            // The subst branch
            SUBST_ACT_NAME,
            SUBST_ACT_TERM,

            END_ACT,
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
        Tokenizer tokenizer;
        SymbolKernel kernel;

        // whether the substitution is allowed
        bool allow_subst;

        std::set<int> func_symbols;
        std::set<int> var_symbols;

        // This mapping preserves the required variable tokens for each rule.
        std::map<int, std::set<int>> required_vars_map;

        // This set preserves the variables that can be substituted.
        std::set<int> subst_variables;

        // The encodings for input history
        std::vector<int> encodings;

        // The valid next tokens
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
        NextTokenMachine(const Algebra& algebra, bool allow_subst = true);

        /**
         * @brief Construct a new Next Token Machine object by copying the other machine.
         * 
         * @param other
         */
        NextTokenMachine(const NextTokenMachine& other);

        NextTokenMachine copy() const {
            return NextTokenMachine(*this);
        }

        const std::vector<int>& get_encodings() const {
            return encodings;
        }

        std::string get_input() const {
            return tokenizer.decode(encodings);
        }

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
            if (!tokenizer.is_valid_token(token)) {
                return false;
            }
            return push_token(tokenizer.get_encoding(token));
        }

        /**
         * @brief Push the input encodings into the machine and update the state.
         * 
         * @param code
         * @return bool, whether the code is valid and successfully pushed.
         */
        bool push_encodings(const std::vector<int>& encodings);

        /**
         * @brief Push the string input into the machine and update the state.
         * 
         * @param code
         * @return bool, whether the code is valid and successfully pushed.
         */
        bool push_string(std::string code);

        std::string to_string() const;

        State get_state() const {
            return state;
        }
    };

} // namespace ualg