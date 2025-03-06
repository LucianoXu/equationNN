#pragma once

#include "algebra.hpp"
#include "env.hpp"

namespace ualg {
    /**
     * @brief Generate the Vampire encode for the given equation.
     */
    std::string vampire_problem_encode(const Algebra& algebra, equation eq, bool comment = true);

} // namespace ualg