
#include "envbackend.hpp"

using namespace std;

namespace ualg {

    SymbolKernel::SymbolKernel(const Algebra& _algebra) : algebra(_algebra), sig(algebra.get_signature()) {
        for (const auto& [name, eq] : algebra.get_axioms()) {
            // Forward Direction (L2R)
            auto L2R_name = name + "_L2R";
            rule_names.push_back(L2R_name);
            rules[L2R_name] = RewriteRule(eq.lhs, eq.rhs, sig);

            // Backward Direction (R2L)
            auto R2L_name = name + "_R2L";
            rule_names.push_back(R2L_name);
            rules[R2L_name] = RewriteRule(eq.rhs, eq.lhs, sig);
        }
    }

    ACT_RESULT SymbolKernel::action(equation& eq, const proof_action& act) {
        auto [rule_name, pos, spec_subst] = act;
        if (pos.size() == 0) {
            return FAILURE;
        }

        try {
            auto side = pos[0] == 0 ? eq.lhs : eq.rhs;

            // check whether the rule name is valid
            auto find_res = rules.find(rule_name);
            if (find_res == rules.end()) {
                return FAILURE;
            }
            auto& rule = find_res->second;
            auto res = rule.apply_at(side, TermPos(pos.begin()+1, pos.end()), spec_subst);

            if (res.has_value()) {
                if (pos[0] == 0) {
                    eq.lhs = res.value();
                }
                else {
                    eq.rhs = res.value();
                }
                return SUCCESS;
            }
            else {
                return FAILURE;
            }
        }
        catch (const std::exception& e) {
            return FAILURE;
        }
    }

    ACT_RESULT SymbolKernel::action(equation& eq, const string& action_code) {
        auto act = parse_proof_action(action_code);
        if (!act.has_value()) {
            return FAILURE;
        }
        return action(eq, act.value());
    }

    std::vector<std::pair<std::string, TermPos>> SymbolKernel::get_valid_rule_pos(const equation& eq) const {
        vector<pair<string, TermPos>> res;
        // Get all the subterms
        auto lhs_subterms = eq.lhs->get_all_subterms();
        auto rhs_subterms = eq.rhs->get_all_subterms();
        
        // iterate throw all the rules in the kernel
        for (const auto& [rule_name, rule] : rules) {
            for (const auto& [pos, subterm] : lhs_subterms) {
                if (rule.match_term(subterm)) {
                    auto new_pos = pos;
                    new_pos.insert(new_pos.begin(), 0);
                    res.push_back({rule_name, new_pos});
                }
            }

            for (const auto& [pos, subterm] : rhs_subterms) {
                if (rule.match_term(subterm)) {
                    auto new_pos = pos;
                    new_pos.insert(new_pos.begin(), 1);
                    res.push_back({rule_name, new_pos});
                }
            }
        }

        return res;
    }

    Tokenizer::Tokenizer(const Algebra& _algebra) : algebra(_algebra), sig(_algebra.get_signature()) {
        // The basic symbols
        vocab = {"<PAD>", "<SOS>", "<EOS>", "(", ")", ":", "{", "}", ",", "="};

        // The function symbols
        int max_arity = 2;  // starting with 2, for the choice in equation A = B
        for (const auto& [name, arity] : sig.get_func_symbols()) {
            vocab.push_back(name);
            if (arity > max_arity) {
                max_arity = arity;
            }
        }

        // The variables
        for (const auto& var : sig.get_variables()) {
            vocab.push_back(var);
        }

        // The positions
        for (int i = 0; i < max_arity; ++i) {
            vocab.push_back(::to_string(i));
        }

        // The rewrite rule names
        auto kernel = SymbolKernel(algebra);
        for (const auto& name : kernel.get_rule_names()) {
            vocab.push_back(name);
        }

        ////////////////////////////////////////////////
        // Build the mapping
        for (size_t i = 0; i < vocab.size(); ++i) {
            vocab_map[vocab[i]] = i;
        }

        for (int i = 0; i < max_arity; ++i) {
            pos_int_map.push_back(vocab_map[::to_string(i)]);
        }
    }

    vector<int> Tokenizer::encode(const string& code) const {
        vector<int> res;
        auto tokens = parse_tokens(code);
        for (const auto& str : tokens) {
            res.push_back(get_encoding(str));
        }
        return res;
    }

    string Tokenizer::decode(const vector<int>& encoding) const {
        string res;
        for (const auto& token : encoding) {
            res += get_token(token) + " ";
        }
        return res;
    }

    void NextTokenMachine::calculate_valid_next_tokens() {
        valid_next_tokens.clear();
        switch (state) {
        case LHS:
        case RHS:
        case SUBST_TERM:
            if (parenthesis == NONE) {
                // Add function symbols and variables
                valid_next_tokens = func_symbols;
                valid_next_tokens.insert(var_symbols.begin(), var_symbols.end());
            }
            else if (parenthesis == OPENING) {
                valid_next_tokens = {tokenizer.get_encoding("(")};
            }
            else if (parenthesis == CLOSING) {
                valid_next_tokens = {tokenizer.get_encoding(")")};
            }
            break;

        case EQ:
            valid_next_tokens = {tokenizer.get_encoding("=")};
            break;

        case COLON:
        case SUBST_COLON:
            valid_next_tokens = {tokenizer.get_encoding(":")};
            break;

        case RULE_NAME:
            for (const auto& rule : *p_current_rule_pos_tree) {
                valid_next_tokens.insert(rule.choice);
            }
            break;
            
        case POS:
            // For POS part, the closing parenthesis is not fixed. So we don't loop up the parenthesis == closing condition.
            if (parenthesis == OPENING) {
                valid_next_tokens = {tokenizer.get_encoding("(")};
            }
            else {
                for (const auto& token : *p_current_rule_pos_tree) {
                    if (token.choice == -1) {
                        valid_next_tokens.insert(tokenizer.get_encoding(")"));
                    }
                    else {
                        valid_next_tokens.insert(token.choice);
                    }
                }
            }
            break;

        case SUBST:
            if (parenthesis == OPENING) {
                valid_next_tokens = {tokenizer.get_encoding("{")};
            }
            else if (parenthesis == CLOSING) {
                valid_next_tokens = {tokenizer.get_encoding("}")};
            }
            else {
                // Check the remaining variables
                for (const auto& var : remaining_vars) {
                    valid_next_tokens.insert(var);
                }
            }
            break;
        
        case COMMA:
            valid_next_tokens = {tokenizer.get_encoding(",")};
            break;

        case EOS:
            valid_next_tokens = {tokenizer.get_encoding("<EOS>")};
            break;

        case HALT:
            break;

        default:
            throw std::runtime_error("Invalid state.");
        }   
    }

    // DEBUG PURPOSE
    void print_tree(const std::vector<NextTokenMachine::CandidateTree>& tree, const vector<string>& mapping, int depth = 0) {
        for (const auto& node : tree) {
            if (node.choice == -1) {
                std::cout << std::string(depth * 2, ' ') << "- END\n";
                continue;
            }
            std::cout << std::string(depth * 2, ' ') << "- " << mapping[node.choice] << "\n";
            print_tree(node.children, mapping, depth + 1);
        }
    }

    void NextTokenMachine::parse_valid_rule_pos_tree(std::vector<std::pair<std::string, TermPos>>&& valid_rule_pos) {

        valid_rule_pos_tree.clear();
        for (const auto& [rule_name, pos] : valid_rule_pos) {
            auto p_trees = &valid_rule_pos_tree;

            auto branch = find_if(p_trees->begin(), p_trees->end(), [&](const CandidateTree& t) { return t.choice == tokenizer.get_encoding(rule_name); });
            if (branch == p_trees->end()) {
                p_trees->push_back({tokenizer.get_encoding(rule_name), {}});
                branch = p_trees->end() - 1;
            }
            p_trees = &(branch->children);

            for (int i = 0; i < pos.size(); ++i) {
                auto it = find_if(p_trees->begin(), p_trees->end(), [&](const CandidateTree& t) { return t.choice == tokenizer.get_pos_int_encoding(pos[i]); });
                if (it == p_trees->end()) {
                    p_trees->push_back({tokenizer.get_pos_int_encoding(pos[i]), {}});
                    it = p_trees->end() - 1;
                }
                p_trees = &(it->children);
            }

            // add the leaf node to represent the end of the path
            p_trees->push_back({-1, {}});
        }

        // output the tree for debug purpose
        // cout << endl;
        // print_tree(valid_rule_pos_tree, vocab);
        // cout << endl;
    }

    NextTokenMachine::NextTokenMachine(const Algebra& _algebra) : algebra(_algebra), sig(_algebra.get_signature()), tokenizer(Tokenizer(_algebra)), kernel(SymbolKernel(_algebra)) {

        for (const auto& f : algebra.get_signature().get_func_symbols()) {
            func_symbols.insert(tokenizer.get_encoding(f.name));
        }

        for (const auto& v : algebra.get_signature().get_variables()) {
            var_symbols.insert(tokenizer.get_encoding(v));
        }

        // get the maximum arity
        int max_arity = 0;
        for (const auto& [name, arity] : algebra.get_signature().get_func_symbols()) {
            if (arity > max_arity) {
                max_arity = arity;
            }
        }

        // calculate the required variables
        for (const auto& [name, rule] : kernel.get_rules()) {
            auto name_set = rule.get_required_subst_vars();
            set<int> token_set = {};
            for (const auto& var : name_set) {
                token_set.insert(tokenizer.get_encoding(var));
            }
            required_vars_map[tokenizer.get_encoding(name)] = token_set;
        }


        encodings.push_back(tokenizer.get_encoding("<SOS>"));
        state = LHS;
        parenthesis = NONE;

        calculate_valid_next_tokens();
    }

    NextTokenMachine::NextTokenMachine(const NextTokenMachine& other) : 
            algebra(other.algebra),
            sig(other.sig),
            tokenizer(other.tokenizer),
            kernel(other.kernel) {
        func_symbols = other.func_symbols;
        var_symbols = other.var_symbols;
        required_vars_map = other.required_vars_map;
        encodings = other.encodings;
        valid_next_tokens = other.valid_next_tokens;
        state = other.state;
        term_gen_stack = other.term_gen_stack;
        parenthesis = other.parenthesis;

        valid_rule_pos_tree = other.valid_rule_pos_tree;
        p_current_rule_pos_tree = &valid_rule_pos_tree;

        remaining_vars = other.remaining_vars;
    }

    bool NextTokenMachine::push_token(int token) {
        if (valid_next_tokens.find(token) == valid_next_tokens.end()) {
            return false;
        }

        encodings.push_back(token);

        std::vector<CandidateTree> temp;
        string eq_code = "";

        switch (state) {
        case LHS:
        case RHS:
        case SUBST_TERM:
            if (parenthesis == OPENING || parenthesis == CLOSING) {
                parenthesis = NONE;
            }
            else {
                // a function symbol or a variable
                if (var_symbols.find(token) != var_symbols.end()) {
                    if (!term_gen_stack.empty()) term_gen_stack.top().second += 1;
                }
                else {
                    // get the arity of the function symbol
                    int arity = sig.get_arity(tokenizer.get_token(token));
                    if (!term_gen_stack.empty()) term_gen_stack.top().second += 1;
                    if (arity > 0) {   
                        term_gen_stack.push({arity, 0});
                        parenthesis = OPENING;
                    }
                }
            }

            // check whether the term is completed
            if (!term_gen_stack.empty() && term_gen_stack.top().first == term_gen_stack.top().second) {
                term_gen_stack.pop();
                parenthesis = CLOSING;
            }

            if (parenthesis == NONE && term_gen_stack.empty()) {
                if (state == LHS) {
                    state = EQ;
                }
                else if (state == RHS) {
                    state = COLON;
                }
                else if (state == SUBST_TERM) {
                    if (remaining_vars.empty()) {
                        state = SUBST;
                        parenthesis = CLOSING;
                    }
                    else {
                        state = COMMA;
                    }
                }
            }

            break;
        
        case EQ:
            state = RHS;
            break;

        case COLON:
            state = RULE_NAME;

            // get the valid rule names
            for (int i = 1; i < encodings.size()-1; ++i) {
                eq_code += tokenizer.get_token(encodings[i]) + " ";
            }
            parse_valid_rule_pos_tree(
                kernel.get_valid_rule_pos(parse_equation(eq_code).value())
            );

            p_current_rule_pos_tree = &valid_rule_pos_tree;

            break;

        case RULE_NAME:
            // get the choice and swap
            for (auto& tree : *p_current_rule_pos_tree) {
                if (tree.choice == token) {
                    p_current_rule_pos_tree = &tree.children;
                    break;
                }
            }

            // calculate the remaining variables
            remaining_vars = required_vars_map[token];

            state = POS;
            parenthesis = OPENING;
            break;   

        case POS:
            if (parenthesis == OPENING) {
                parenthesis = NONE;
            }

            else if (token == tokenizer.get_encoding(")")) {
                parenthesis = OPENING;
                state = SUBST;
            }

            else {
                // a position
                for (auto& tree : *p_current_rule_pos_tree) {
                    if (tree.choice == token) {
                        p_current_rule_pos_tree = &tree.children;
                    break;
                    }
                }
            }
            break;

        case SUBST:
            if (parenthesis == OPENING) {
                parenthesis = remaining_vars.empty()? CLOSING : NONE;
            }

            else if (parenthesis == CLOSING) {
                state = EOS;
            }

            else {
                // a variable
                remaining_vars.erase(token);
                state = SUBST_COLON;
            }
            break;

        case SUBST_COLON:
            state = SUBST_TERM;
            break;

        case COMMA:
            state = SUBST;
            break;

        case EOS:
            state = HALT;
            break;

        default:
            throw std::runtime_error("Invalid state.");
        }

        calculate_valid_next_tokens();
        return true;
    }

    std::string NextTokenMachine::to_string() const {
        string res = "SEQ:\t\t";
        // input
        for (int encoding : encodings) {
            res += tokenizer.get_token(encoding) + " ";
        }
        // possible next tokens
        res += "\nVALID NEXT:\t";
        for (int encoding : valid_next_tokens) {
            res += tokenizer.get_token(encoding) + " ";
        }
        return res;
    }

} // namespace ualg