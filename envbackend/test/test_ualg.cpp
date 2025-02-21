#include <gtest/gtest.h>

#include "envbackend.hpp"

using namespace std;
using namespace ualg;


TEST(TestAlg, parsing_term) {
    auto actual_res = ualg::parse_term("f(x y)").value();
    auto expected_res = make_shared<Term>("f", vector<TermPtr>{make_shared<Term>("x"), make_shared<Term>("y")});
    EXPECT_EQ(*actual_res, *expected_res);
}

TEST(TestAlg, parsing_alg) {
    string actual_res = parse_alg(R"(
        [function]
        & : 2
        | : 2
        ~ : 1

        [variable]
        x y z u v w

        [axiom]
        (AX1) &(x y) = &(y x)
        (AX2) &(x |(y z)) = |(&(x y) &(x z))
    )").value().to_string();

    string expected_res = 
R"([function]
& : 2
| : 2
~ : 1

[variable]
x y z u v w 

[axiom]
(AX1) &(x y) = &(y x)
(AX2) &(x |(y z)) = |(&(x y) &(x z))
)";
    
    EXPECT_EQ(actual_res, expected_res);
}