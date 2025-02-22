#include "envbackend.hpp"
using namespace std;

namespace ualg {


    //////////////////////////////////////////////////////////////////
    // Signature

    string Signature::to_string() const {
        string res = "[function]\n";
        for (const auto& func : init_func_symbols) {
            res += func.name + " : " + ::to_string(func.arity) + "\n";
        }
        res += "\n";

        res += "[variable]\n";
        for (const auto& var : init_variables) {
            res += var + " ";
        }
        res += "\n";
        return res;
    }

    bool Signature::term_valid(TermPtr term) const {
        auto& head = term->get_head();
        auto& args = term->get_args();

        if (term->is_atomic()) {
            // If the term is a variable or a constant function symbol.
            if (variables.find(head) != variables.end()
                || (func_symbols.find(head) != func_symbols.end() && func_symbols.at(head).arity == 0)) {
                return true;
            }
            return false;
        }

        // If the term is a function construction.
        if (func_symbols.find(head) == func_symbols.end()) {
            return false;
        }

        if (func_symbols.at(head).arity != args.size()) {
            return false;
        }
        
        for (const auto& arg : args) {
            if (!term_valid(arg)) {
                return false;
            }
        }

        return true;
    }

    Term::Term(const string& head) {
        this->head = head;
        this->args = {};
    }

    Term::Term(const string& head, const vector<TermPtr>& args) {
        this->head = head;
        this->args = args;
    }

    Term::Term(const string& head, vector<TermPtr>&& args) {
        this->head = head;
        this->args = std::move(args);
    }

    const string& Term::get_head() const {
        return this->head;
    }

    const vector<TermPtr>& Term::get_args() const {
        return this->args;
    }

    bool Term::operator == (const Term& other) const {
        if (this->head != other.head) {
            return false;
        }
        if (this->args.size() != other.args.size()) {
            return false;
        }
        for (int i = 0; i < this->args.size(); i++) {
            if (!(*this->args[i] == *other.args[i])) {
                return false;
            }
        }
        return true;
    }

    bool Term::operator != (const Term& other) const {
        return !(*this == other);
    }

    size_t Term::get_term_size() const {
        size_t size = 1;
        for (const auto& arg : args) {
            size += arg->get_term_size();
        }
        return size;
    }

    bool Term::is_atomic() const {
        return args.size() == 0;
    }

    string Term::to_string() const {
        string str = this->head;
        if (args.size() == 0) {
            return str;
        }
        
        str += "(";
        for (int i = 0; i < args.size(); i++) {
            str += args[i]->to_string() + " ";
        }
        if (args.size() > 0) str.pop_back();
        str += ")";
        return str;
    }

    set<string> Term::get_variables(const Signature& sig) const {
        auto& sig_variables = sig.get_variables();
        if (args.size() == 0) {
            if (sig_variables.find(head) != sig_variables.end()) {
                return {head};
            }
            return {};
        }

        set<string> res;
        for (const auto& arg : args) {
            auto arg_vars = arg->get_variables(sig);
            res.insert(arg_vars.begin(), arg_vars.end());
        }

        return res;
    }

    TermPtr Term::get_subterm(const TermPos& pos) const {
        if (pos.size() == 0) {
            return this->shared_from_this();
        }

        // if (pos[0] >= args.size()) {
        //     throw runtime_error("Position out of range.");
        // }

        return args[pos[0]];
    }

    TermPtr Term::replace_term(TermPtr pattern, TermPtr replacement) const {
        if (*this == *pattern) {
            return replacement;
        }

        vector<TermPtr> new_args;
        for (const auto& arg : args) {
            new_args.push_back(arg->replace_term(pattern, replacement));
        }
        return make_shared<const Term>(this->head, std::move(new_args));
    }

    TermPtr Term::replace_at(const TermPos& pos, TermPtr new_subterm) const {
        if (pos.size() == 0) {
            return new_subterm;
        }

        vector<TermPtr> new_args;
        for (int i = 0; i < args.size(); i++) {
            if (i == pos[0]) {
                new_args.push_back(args[i]->replace_at(
                    TermPos(pos.begin() + 1, pos.end()),
                    new_subterm));
            }
            else {
                new_args.push_back(args[i]);
            }
        }
        return make_shared<const Term>(this->head, std::move(new_args));
    }

    void _get_all_subterms(TermPtr term, TermPos pos_prefix, vector<pair<TermPos, TermPtr>>& results) {
        results.push_back({pos_prefix, term});

        for (int i = 0; i < term->get_args().size(); i++) {
            pos_prefix.push_back(i);
            _get_all_subterms(term->get_args()[i], pos_prefix, results);
            pos_prefix.pop_back();
        }
    }

    /**
     * @brief Get the all subterms object
     * 
     * @return std::vector<std::pair<TermPos, TermPtr>> 
     */
    vector<pair<TermPos, TermPtr>> Term::get_all_subterms() const {
        vector<pair<TermPos, TermPtr>> results;
        TermPos pos_prefix;
        _get_all_subterms(this->shared_from_this(), pos_prefix, results);
        return results;
    }


    //////////////////////////////////////////////////////////////////
    // Algebra

    string Algebra::to_string() const {
        string res = "";
        res += sig.to_string();
        res += "\n[axiom]\n";
        for (const auto& axiom : init_axioms) {
            res += "(" + axiom.first + ") " + axiom.second.to_string() + "\n";
        }
        return res;
    }



    //////////////////////////////////////////////////////////////////
    // About substitution and matching

    string to_string(const subst& s) {
        string res = "{";
        for (const auto& [var, term] : s) {
            res += var + ": " + term->to_string() + ", ";
        }
        if (s.size() > 0) {
            res.pop_back();
            res.pop_back();
        }
        res += "}";
        return res;
    }

    bool subst_eq(const subst& s1, const subst& s2) {
        if (s1.size() != s2.size()) {
            return false;
        }

        for (const auto& [var, term] : s1) {
            if (s2.find(var) == s2.end() || *s2.at(var) != *term) {
                return false;
            }
        }
        return true;
    }

    TermPtr apply_subst(TermPtr term, const subst& s) {
        if (term->is_atomic()) {
            if (s.find(term->get_head()) != s.end()) {
                return s.at(term->get_head());
            }
            return term;
        }

        vector<TermPtr> new_args;
        for (const auto& arg : term->get_args()) {
            new_args.push_back(apply_subst(arg, s));
        }
        return make_shared<const Term>(term->get_head(), std::move(new_args));
    }

    optional<subst> _match(stack<pair<TermPtr, TermPtr>>& ineqs, const set<std::string>& vars, subst& spec_subst) {

        while (!ineqs.empty()) {

            auto [lhs, rhs] = ineqs.top();
            ineqs.pop();

            // if the lhs is variable
            if (lhs->is_atomic() && vars.find(lhs->get_head()) != vars.end()) {
                // if the variable is already specified in the substitution
                if (spec_subst.find(lhs->get_head()) != spec_subst.end()) {
                    if (*spec_subst.at(lhs->get_head()) == *rhs) {
                        continue;
                    }
                    else {
                        return nullopt;
                    }
                }
                else {
                    spec_subst[lhs->get_head()] = rhs;
                    continue;
                }
            }
            // if the lhs is function construction
            else {
                // if rhs is variable, then not matchable
                if (rhs->is_atomic() &&  vars.find(rhs->get_head()) != vars.end()) {
                    return nullopt;
                }

                else {
                    if (lhs->get_head() == rhs->get_head()) {
                        auto& lhs_args = lhs->get_args();
                        auto& rhs_args = rhs->get_args();
                        for (int i = 0; i < lhs->get_args().size(); i++) {
                            ineqs.push({lhs_args[i], rhs_args[i]});
                        }
                        continue;
                    }
                    else {
                        return nullopt;
                    }
                }   
            }
        }

        return spec_subst;
    }


    optional<subst> match(TermPtr term, TermPtr pattern, const set<std::string>& vars, const subst& spec_subst) {
        stack<pair<TermPtr, TermPtr>> ineqs;
        ineqs.push({pattern, term});

        auto temp_subst = spec_subst;
        return _match(ineqs, vars, temp_subst);
    }

    optional<subst> match(TermPtr term, TermPtr pattern, const Signature& sig, const subst& spec_subst) {
        // NOTICE: no validity check for the term and pattern
        return match(term, pattern, sig.get_variables(), spec_subst);
    }

    string RewriteRule::to_string() const {
        return lhs->to_string() + " -> " + rhs->to_string();
    }

    bool RewriteRule::match_term(TermPtr term) const {
        return ualg::match(term, lhs, *p_sig, {}).has_value();
    }

    optional<TermPtr> RewriteRule::apply(TermPtr term, const subst& spec_subst) const {
        auto temp_subst = spec_subst;
        
        // check whether all required variables are in the substitution
        for (const auto& var : required_subst_vars) {
            if (temp_subst.find(var) == temp_subst.end()) {
                throw runtime_error("Variable " + var + " is not specified the substitution " + ualg::to_string(temp_subst) + ".");
            }
        }

        auto res = match(term, lhs, *p_sig, temp_subst);

        if (res.has_value()) {
            return apply_subst(rhs, res.value());
        }
        else {
            return nullopt;
        }
    }

    
    std::optional<TermPtr> RewriteRule::apply_at(TermPtr term, const TermPos& pos, const subst& spec_subst) const {
        if (pos.size() == 0) {
            return apply(term, spec_subst);
        }

        auto new_subterm = term->get_subterm(pos);
        auto new_subterm_res = apply_at(new_subterm, TermPos(pos.begin() + 1, pos.end()), spec_subst);

        if (new_subterm_res.has_value()) {
            return term->replace_at(pos, new_subterm_res.value());
        }
        else {
            return nullopt;
        }
    }

} // namespace ualg