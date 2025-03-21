
#include "envbackend.hpp"

using namespace std;

namespace ualg {

    bool check_symm_axiom(const Signature& sig, const equation& eq) {
        return match(eq.lhs, eq.rhs, sig, {}).has_value() && match(eq.rhs, eq.lhs, sig, {}).has_value();
    }

    SymbolKernel::SymbolKernel(const Algebra& _algebra) : algebra(_algebra), sig(algebra.get_signature()) {
        for (const auto& [name, eq] : algebra.get_axioms()) {
            if (check_symm_axiom(sig, eq)) {
                // Symmetric Axiom
                rule_names.push_back(name);
                rules[name] = RewriteRule(eq.lhs, eq.rhs, sig);
            }
            else {
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
    }

    ACT_RESULT SymbolKernel::action(proof_state& stt, const proof_action& act) const {
        auto [rule_name, pos, spec_subst] = act;

        // if it is a substitution
        if (rule_name == "SUBST") {
            if (pos.size() != 0) {
                return FAILURE;
            }
            stt.eq.lhs = apply_subst(stt.eq.lhs, spec_subst);
            stt.eq.rhs = apply_subst(stt.eq.rhs, spec_subst);
            return SUCCESS;
        }

        // if it is a normal rule
        if (pos.size() == 0) {
            return FAILURE;
        }

        try {
            auto side = pos[0] == 0 ? stt.eq.lhs : stt.eq.rhs;

            // check whether the rule name is valid
            auto find_res = rules.find(rule_name);
            if (find_res == rules.end()) {
                return FAILURE;
            }
            auto& rule = find_res->second;
            auto res = rule.apply_at(side, TermPos(pos.begin()+1, pos.end()), spec_subst);

            if (res.has_value()) {
                if (pos[0] == 0) {
                    stt.eq.lhs = res.value();
                }
                else {
                    stt.eq.rhs = res.value();
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

    ACT_RESULT SymbolKernel::action_by_code(proof_state& stt, const string& action_code) const {
        auto act = parse_proof_action(action_code);
        if (!act.has_value()) {
            // check whether the action code is finished
            auto tokens = parse_tokens(action_code);

            // if the action code is empty, generation is not finished
            // if the last token is "</ACT>", the generation is finished. In this case, the behavior is unexpected.
            if (tokens.size() > 0 && tokens.back() == "</ACT>") {
                // If the invalid action is a finished action, raise an exception
                throw std::runtime_error("Invalid Finished Action: " + action_code);
            }
            return INVALID;
        }
        return action(stt, act.value());
    }

    std::vector<std::pair<std::string, TermPos>> SymbolKernel::get_valid_rule_pos(const proof_state& stt) const {
        vector<pair<string, TermPos>> res;
        // Get all the subterms
        auto lhs_subterms = stt.eq.lhs->get_all_subterms();
        auto rhs_subterms = stt.eq.rhs->get_all_subterms();
        
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
        vocab = {"<PAD>", "<STT>", "</STT>", "<ACT>", "</ACT>", "(", ")", ":", "{", "}", ",", "="};

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

        // The SUBST action and rewrite rule names
        vocab.push_back("SUBST");
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

    /**
     * @brief Check wether the code is valid.
     */
    bool check_step(const SymbolKernel& kernel, std::string code) {
        optional<proof_step> step = parse_proof_step(code);
        if (!step.has_value()) {
            return false;
        }
        auto res = kernel.action(step->stt, step->act);
        return res == SUCCESS;
    }

    void NextTokenMachine::calculate_valid_next_tokens() {
        valid_next_tokens.clear();
        switch (state) {
        case START_STT:
            valid_next_tokens = {tokenizer.get_encoding("<STT>")};
            break;
        case LHS:
        case RHS:
        case SUBST_TERM:
        case SUBST_ACT_TERM:
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

        case END_STT:
            valid_next_tokens = {tokenizer.get_encoding("</STT>")};
            break;

        case SUBST_COLON:
            valid_next_tokens = {tokenizer.get_encoding(":")};
            break;

        case START_ACT:
            valid_next_tokens = {tokenizer.get_encoding("<ACT>")};
            break;

        case ACT_NAME:
            for (const auto& rule : *p_current_rule_pos_tree) {
                valid_next_tokens.insert(rule.choice);
            }
            if (allow_subst && subst_variables.size() > 0) {
                valid_next_tokens.insert(tokenizer.get_encoding("SUBST"));
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

        case SUBST_ACT_NAME:
            valid_next_tokens = subst_variables;
            break;

        case END_ACT:
            valid_next_tokens = {tokenizer.get_encoding("</ACT>")};
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

    NextTokenMachine::NextTokenMachine(const Algebra& _algebra, bool _allow_subst) : algebra(_algebra), sig(_algebra.get_signature()), tokenizer(Tokenizer(_algebra)), kernel(SymbolKernel(_algebra)), allow_subst(_allow_subst) {

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


        encodings = {};
        state = START_STT;
        parenthesis = NONE;

        calculate_valid_next_tokens();
    }

    NextTokenMachine::NextTokenMachine(const NextTokenMachine& other) : 
            algebra(other.algebra),
            sig(other.sig),
            tokenizer(other.tokenizer),
            kernel(other.kernel),
            allow_subst(other.allow_subst) {
        func_symbols = other.func_symbols;
        var_symbols = other.var_symbols;
        required_vars_map = other.required_vars_map;
        subst_variables = other.subst_variables;
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
        // first check whether the token is valid
        if (valid_next_tokens.find(token) == valid_next_tokens.end()) {
            return false;
        }

        encodings.push_back(token);

        std::vector<CandidateTree> temp;
        proof_state stt;
        set<string> lhs_vars, rhs_vars;

        switch (state) {
        case START_STT:
            state = LHS;
            break;
        case LHS:
        case RHS:
        case SUBST_TERM:
        case SUBST_ACT_TERM:
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

            // transform to the next state
            if (parenthesis == NONE && term_gen_stack.empty()) {
                if (state == LHS) {
                    state = EQ;
                }
                else if (state == RHS) {
                    state = END_STT;
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
                else if (state == SUBST_ACT_TERM) {
                    state = END_ACT;
                }
                else {
                    throw std::runtime_error("Invalid state.");
                }
            }

            break;
        
        case EQ:
            state = RHS;
            break;

        case END_STT:
            state = START_ACT;

            // get the equation
            stt = parse_proof_state(tokenizer.decode(encodings)).value();

            // get the valid rule names
            parse_valid_rule_pos_tree(
                kernel.get_valid_rule_pos(stt)
            );

            p_current_rule_pos_tree = &valid_rule_pos_tree;

            // calculate the possible variables to be substituted
            subst_variables.clear();
            lhs_vars = stt.eq.lhs->get_variables(sig);
            rhs_vars = stt.eq.rhs->get_variables(sig);
            for (const auto& var : lhs_vars) {
                subst_variables.insert(tokenizer.get_encoding(var));
            }
            for (const auto& var : rhs_vars) {
                subst_variables.insert(tokenizer.get_encoding(var));
            }
            break;

        case START_ACT:
            state = ACT_NAME;
            break;

        case ACT_NAME:
            if (token == tokenizer.get_encoding("SUBST")) {
                state = SUBST_ACT_NAME;
            }
            else {
                // get the choice
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
            }
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
                state = END_ACT;
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

        case SUBST_ACT_NAME:
            state = SUBST_ACT_TERM;
            break;

        case END_ACT:
            state = HALT;
            break;

        default:
            throw std::runtime_error("Invalid state.");
        }

        calculate_valid_next_tokens();
        return true;
    }

    bool NextTokenMachine::push_encodings(const vector<int>& encodings) {
        for (int encoding : encodings) {
            if (!push_token(encoding)) {
                return false;
            }
        }
        return true;
    }

    bool NextTokenMachine::push_string(std::string code) {
        auto tokens = parse_tokens(code);
        for (const auto& token : tokens) {
            if (!push_token(token)) {
                return false;
            }
        }
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