#include <gtest/gtest.h>

#include "envbackend.hpp"

using namespace std;
using namespace ualg;


TEST(TestAlg, parsing_term) {
    auto actual_res = ualg::parse_term("f(x y)").value();
    auto expected_res = make_shared<Term>("f", vector<TermPtr>{make_shared<Term>("x"), make_shared<Term>("y")});
    EXPECT_EQ(*actual_res, *expected_res);
}

TEST(TestAlg, parsing_subst) {
    auto actual_res = ualg::parse_subst("{x: y, z: f(x y)}").value();
    subst expected_res = {{"x", make_shared<Term>("y")}, {"z", make_shared<Term>("f", vector<TermPtr>{make_shared<Term>("x"), make_shared<Term>("y")})}};
    EXPECT_TRUE(subst_eq(actual_res, expected_res));
}

TEST(TestAlg, parsing_sig) {
    string actual_res = parse_signature(R"(
        [function]
        & : 2
        | : 2
        ~ : 1

        [variable]
        x y z u v w
    )").value().to_string();

    string expected_res = 
R"([function]
& : 2
| : 2
~ : 1

[variable]
x y z u v w 
)";
    
    EXPECT_EQ(actual_res, expected_res);
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

TEST(TestAlg, signature_check) {
    Signature sig = parse_signature(R"(
        [function]
        & : 2
        | : 2
        ~ : 1
        zero : 0

        [variable]
        x y z u v w
    )").value();

    auto term1 = parse_term("f(x y)").value();
    EXPECT_TRUE(!sig.term_valid(term1));

    auto term2 = parse_term("zero").value();
    EXPECT_TRUE(sig.term_valid(term2));

    auto term3 = parse_term("&(x ~(zero))").value();
    EXPECT_TRUE(sig.term_valid(term3));

    auto term4 = parse_term("&(x ~(zero) zero)").value();
    EXPECT_TRUE(!sig.term_valid(term4));
}

TEST(TestAlg, matching1) {
    auto term = parse_term("f(x y)").value();
    auto pattern = parse_term("f(X Y)").value();
    
    auto res = match(term, pattern, {"X", "Y"}, {});
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(to_string(res.value()), "{X: x, Y: y}");
}

TEST(TestAlg, matching2) {
    auto term = parse_term("f(g(x y) x)").value();
    auto pattern = parse_term("f(g(X Y) Z)").value();
    
    auto res = match(term, pattern, {"X", "Y", "Z"}, {});
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(to_string(res.value()), "{X: x, Y: y, Z: x}");
}

TEST(TestAlg, matching3) {
    auto term = parse_term("f(g(x y) x)").value();
    auto pattern = parse_term("f(g(X Y) Z)").value();
    
    auto res = match(term, pattern, {"X", "Y", "Z"}, parse_subst("{Z: x}").value());
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(to_string(res.value()), "{X: x, Y: y, Z: x}");
}

TEST(TestAlg, matching4) {
    auto term = parse_term("f(g(x y) x)").value();
    auto pattern = parse_term("f(g(X Y) Y)").value();
    
    auto res = match(term, pattern, {"X", "Y"}, {});
    EXPECT_TRUE(!res.has_value());
}

TEST(TestAlg, rewrite1) {
    Signature sig = parse_signature(R"(
        [function]
        & : 2
        | : 2
        ~ : 1
        zero : 0

        [variable]
        x y z u v w
    )").value();
    
    auto lhs = parse_term("&(x y)").value();
    auto rhs = parse_term("&(y x)").value();
    auto rule = RewriteRule(lhs, rhs, sig);

    auto term = parse_term("&(u v)").value();
    auto res = rule.apply(term, {});
    EXPECT_EQ(*res.value(), *parse_term("&(v u)").value());
}


TEST(TestAlg, rewrite2) {
    Signature sig = parse_signature(R"(
        [function]
        & : 2
        | : 2
        ~ : 1
        zero : 0

        [variable]
        x y z u v w
    )").value();
    
    auto lhs = parse_term("&(x y)").value();
    auto rhs = parse_term("|(&(y x) z)").value();
    auto rule = RewriteRule(lhs, rhs, sig);

    auto term = parse_term("&(y x)").value();
    auto res = rule.apply(term, parse_subst("{z: u}").value());
    EXPECT_EQ(*res.value(), *parse_term("|(&(x y) u)").value());
}


TEST(TestAlg, rewrite3) {
    Signature sig = parse_signature(R"(
        [function]
        & : 2
        | : 2
        ~ : 1
        zero : 0

        [variable]
        x y z u v w
    )").value();
    
    auto lhs = parse_term("&(x y)").value();
    auto rhs = parse_term("|(&(y x) z)").value();
    auto rule = RewriteRule(lhs, rhs, sig);

    auto term = parse_term("|(y x)").value();
    auto res = rule.apply(term, parse_subst("{z: u}").value());
    EXPECT_TRUE(!res.has_value());
}