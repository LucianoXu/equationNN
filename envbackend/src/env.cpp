
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

            auto& rule = rules[rule_name];
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
    
} // namespace ualg