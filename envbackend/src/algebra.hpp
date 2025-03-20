/*
Defining the description class of an algebra.
Note that we strictly follow the universal algebra setting. The description consists of the following information:

- function symbols and their arities
- variables
- equational axioms

The description of algebra follows a grammar. An example is demonstrated below.

```
[function]
& : 2
| : 2
~ : 1

[variable]
x y z u v w

[axiom]
(AX1) &(x, y) = &(y, x)
(AX2) &(x, |(y, z)) = |(&(x, y), &(x, z))
```

*/

#pragma once

#include <string>
#include <map>
#include <set>
#include <memory>
#include <vector>
#include <optional>

namespace ualg{

    class Term;
    using TermPos = std::vector<unsigned int>;
    using TermPtr = std::shared_ptr<const Term>;

    class Signature {
    public:
        struct func_symbol {
            std::string name;
            unsigned arity;
        };

    private:
        std::vector<func_symbol> init_func_symbols;
        std::map<std::string, func_symbol> func_symbols;

        std::vector<std::string> init_variables;
        std::set<std::string> variables;

    public:
        Signature(
            const std::vector<func_symbol>& _func_symbols,
            const std::vector<std::string>& _variables) : 
            init_func_symbols(_func_symbols), init_variables(_variables) {

            for (const auto& func : _func_symbols) {
                func_symbols[func.name] = func;
            }

            for (const auto& var : _variables) {
                this->variables.insert(var);
            }
        }

        Signature() = default;

        const std::vector<func_symbol>& get_func_symbols() const {
            return init_func_symbols;
        }

        const std::vector<std::string>& get_init_variables() const {
            return init_variables;
        }

        const std::set<std::string>& get_variables() const {
            return variables;
        }

        unsigned get_arity(const std::string& name) const {
            return func_symbols.at(name).arity;
        }

        std::string to_string() const;

        bool term_valid(TermPtr term) const;
    };


    class Term : public std::enable_shared_from_this<Term> {
    protected:
        std::string head;
        std::vector<TermPtr> args;

    public: 
        Term(const std::string& head);
        Term(const std::string& head, const std::vector<TermPtr>& args);

        const std::string& get_head() const;
        const std::vector<TermPtr>& get_args() const;

        bool operator == (const Term& other) const;
        bool operator != (const Term& other) const;

        std::size_t get_term_size() const;

        bool is_atomic() const;

        std::string to_string() const;

        /**
         * @brief The representation of the term (used in python).
         * 
         * @return std::string 
         */
        std::string to_repr() const;

        std::set<std::string> get_variables(const Signature& sig) const;

        TermPtr get_subterm(const TermPos& pos) const;

        /**
         * @brief Replace all ocurrences of the pattern by the replacement term.
         * 
         * @param pattern 
         * @param replacement 
         * @return TermPtr 
         */
        TermPtr replace_term(TermPtr pattern, TermPtr replacement) const;

        TermPtr replace_at(const TermPos& pos, TermPtr new_subterm) const;

        /**
         * @brief Get the all subterms object
         * 
         * @return std::vector<std::pair<TermPos, TermPtr>> 
         */
        std::vector<std::pair<TermPos, TermPtr>> get_all_subterms() const;
    };

    struct equation {
        TermPtr lhs;
        TermPtr rhs;

        equation() = default;

        equation(TermPtr _lhs, TermPtr _rhs) : lhs(_lhs), rhs(_rhs) {}

        equation(const equation& other) : lhs(other.lhs), rhs(other.rhs) {}

        int get_size() const {
            return lhs->get_term_size() + rhs->get_term_size() + 1;
        }

        std::set<std::string> get_variables(const Signature& sig) const {
            auto lhs_vars = lhs->get_variables(sig);
            auto rhs_vars = rhs->get_variables(sig);
            lhs_vars.insert(rhs_vars.begin(), rhs_vars.end());
            return lhs_vars;
        }

        std::string to_string() const {
            return lhs->to_string() + " = " + rhs->to_string();
        }

        std::string to_repr() const {
            return "<Equation " + to_string() + ">";
        }

        bool operator == (const equation& other) const {
            return *lhs == *other.lhs && *rhs == *other.rhs;
        }
    };

    class Algebra {

    private:
        Signature sig;
        std::vector<std::pair<std::string, equation>> init_axioms;

    public:
        Algebra(
            const Signature& sig,
            const std::vector<std::pair<std::string, equation>>& axioms) :
            sig(sig), init_axioms(axioms) {

            // check whether the axioms are valid in the signature
            for (const auto& axiom : axioms) {
                auto [name, eq] = axiom;
                if (!sig.term_valid(eq.lhs) || !sig.term_valid(eq.rhs)) {
                    throw std::runtime_error("Invalid axiom: " + name + ": " + eq.to_string());
                }
            }
        }

        Algebra() = default;

        std::string to_string() const;

        const Signature& get_signature() const {
            return sig;
        }

        const std::vector<std::pair<std::string, equation>>& get_axioms() const {
            return init_axioms;
        }
    };


    /**
     * @brief A substitution is a map from variable names to terms.
     * 
     */
    using subst = std::map<std::string, TermPtr>;

    std::string to_string(const subst& s);

    bool subst_eq(const subst& s1, const subst& s2);

    TermPtr apply_subst(TermPtr term, const subst& s);

    /**
     * @brief The function tries to match the given term with the specified pattern, variables and substitution.
     * 
     * @param term 
     * @param pattern 
     * @param vars
     * @param given_subst 
     * @return std::optional<subst> 
     */
    std::optional<subst> match(TermPtr term, TermPtr pattern, const std::set<std::string>& vars, const subst& spec_subst);

    /**
     * @brief Match a term with the pattern, using the given signature and substitutions. The validity of the term is *NOT* checked.
     * 
     * @param term 
     * @param pattern 
     * @param sig 
     * @param spec_subst 
     * @return std::optional<subst> 
     */
    std::optional<subst> match(TermPtr term, TermPtr pattern, const Signature& sig, const subst& spec_subst);

    class RewriteRule {
    private:
        TermPtr lhs;
        TermPtr rhs;
        std::shared_ptr<Signature> p_sig;

        // the set of variables that must be include in the spec_subst
        std::set<std::string> required_subst_vars;

    public:
        RewriteRule(TermPtr _lhs, TermPtr _rhs, const Signature& sig) : lhs(_lhs), rhs(_rhs) {
            p_sig = std::make_shared<Signature>(sig);

            // check whether the lhs and rhs are valid in the signature
            if (!sig.term_valid(lhs) || !sig.term_valid(rhs)) {
                throw std::runtime_error("Invalid rewrite rule: " + lhs->to_string() + " -> " + rhs->to_string());
            }

            // Calculate the required variables for the substitution
            auto lhs_vars = lhs->get_variables(*p_sig);
            auto rhs_vars = rhs->get_variables(*p_sig);
            for (const auto& var : rhs_vars) {
                if (lhs_vars.find(var) == lhs_vars.end()) {
                    required_subst_vars.insert(var);
                }
            }
        }
        RewriteRule() = default;

        const std::set<std::string>& get_required_subst_vars() const {
            return required_subst_vars;
        }

        std::string to_string() const;

        /**
         * @brief Check whether the rule can be applied to the term.
         * 
         * @param term 
         * @return true 
         * @return false 
         */
        bool match_term(TermPtr term) const;

        /**
         * @brief Apply the rewrite rule to the term with the specified substitution.
         * 
         * @param term 
         * @param spec_subst 
         * @return std::optional<TermPtr> 
         */
        std::optional<TermPtr> apply(TermPtr term, const subst& spec_subst) const;

        std::optional<TermPtr> apply_at(TermPtr term, const TermPos& pos, const subst& spec_subst) const;
    };

}   // namespace ualg