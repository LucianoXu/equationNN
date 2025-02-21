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

namespace ualg{

    class Term;
    using TermPos = std::vector<unsigned int>;
    using TermPtr = std::shared_ptr<const Term>;

    class Term : public std::enable_shared_from_this<Term> {
    protected:
        std::string head;
        std::vector<TermPtr> args;

    public: 
        Term(const std::string& head);
        Term(const std::string& head, const std::vector<TermPtr>& args);
        Term(const std::string& head, std::vector<TermPtr>&& args);

        const std::string& get_head() const;
        const std::vector<TermPtr>& get_args() const;

        bool operator == (const Term& other) const;
        bool operator != (const Term& other) const;

        std::size_t get_term_size() const;

        bool is_atomic() const;

        std::string to_string() const;

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
    };

    class Algebra {
    public:
        struct func_symbol {
            std::string name;
            unsigned arity;
        };

    private:
        std::vector<func_symbol> func_symbols;
        std::vector<std::string> variables;
        std::vector<std::tuple<std::string, TermPtr, TermPtr>> axioms;

    public:
        Algebra(        
            std::vector<func_symbol> _func_symbols,
            std::vector<std::string> variables,
            std::vector<std::tuple<std::string, TermPtr, TermPtr>> axioms) :
            func_symbols(_func_symbols), variables(variables), axioms(axioms) {}

        Algebra() = default;

        std::string to_string() const;
    };

}   // namespace ualg