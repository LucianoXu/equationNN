
#include "utils.hpp"
#include <stdexcept>

using namespace std;

namespace ualg {

    string vampire_expression(map<string, string> func_map, const map<string, string> var_map, const TermPtr& term) {

        // check whether the term is a variable
        auto var_find_res = var_map.find(term->get_head());
        if (var_find_res != var_map.end()) {
            return var_find_res->second;
        }
        
        // it is a function 
        auto func_find_res = func_map.find(term->get_head());
        if (func_find_res != func_map.end()) {
            string res = func_find_res->second + "(";
            for (const auto& arg : term->get_args()) {
                res += vampire_expression(func_map, var_map, arg) + ", ";
            }
            res.pop_back();
            res.pop_back();
            res += ")";
            return res;
        }

        throw runtime_error("Invalid term: " + term->to_string());
    }

    string vampire_problem_encode(const Algebra& algebra, equation eq, bool comment) {
        auto& sig = algebra.get_signature();

        // Generate the mapping from the function symbols, variables to vampire allowed symbols
        map<string, string> func_map;
        map<string, string> var_map;
        for (const auto& [name, arity] : sig.get_func_symbols()) {
            func_map[name] = "f" + std::to_string(func_map.size());
        }

        for (const auto& var : sig.get_variables()) {
            var_map[var] = "X" + std::to_string(var_map.size());
        }

        string res = "";

        // Add the comment
        if (comment) {
            string comment = "";
            comment += algebra.to_string() + "\n";
            comment += "PROBLEM: " + eq.to_string() + "\n";

            // add '%' to the beginning of each line
            res += "% ";
            for (auto& c : comment) {
                res += c;
                if (c == '\n') {
                    res += "% ";
                }
            }
            res += "\n\n";
        }

        // Add the axioms
        int i = 0;
        for (auto& [name, eq] : algebra.get_axioms()) {
            // Get all the variables in the equation
            set<string> vars = eq.get_variables(sig);
            res += "fof(ax" + std::to_string(i) + ", axiom, ![";
            for (auto& var : vars) {
                res += var_map[var] + ", ";
            }
            res.pop_back();
            res.pop_back();
            res += "] : ";
            res += vampire_expression(func_map, var_map, eq.lhs) + " = " + vampire_expression(func_map, var_map, eq.rhs) + ").\n";

            i++;
        }

        // Add the goal
        set<string> vars = eq.get_variables(sig);
        res += "fof(goal, conjecture, ![";
        for (auto& var : vars) {
            res += var_map[var] + ", ";
        }
        res.pop_back();
        res.pop_back();
        res += "] : ";
        res += vampire_expression(func_map, var_map, eq.lhs) + " = " + vampire_expression(func_map, var_map, eq.rhs) + ").\n";

        return res;
    }
} // namespace ualg