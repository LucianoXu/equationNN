#include <gtest/gtest.h>

#include "envbackend.hpp"

using namespace std;
using namespace ualg;


TEST(TestAlg, parsing_term) {
    auto actual_res = ualg::parse_term("f(x y)").value();
    auto expected_res = make_shared<Term>("f", vector<TermPtr>{make_shared<Term>("x"), make_shared<Term>("y")});
    EXPECT_EQ(*actual_res, *expected_res);
}

TEST(TestAlg, parsing_pos) {
    auto actual_res = ualg::parse_pos("(1 2 3)").value();
    TermPos expected_res = {1, 2, 3};
    EXPECT_EQ(actual_res, expected_res);
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

TEST(TestAlg, term_get_subterm) {
    auto term = parse_term("f(g(x y) x)").value();
    auto res = term->get_subterm({0, 1});
    auto expected_res = make_shared<Term>("y");
    EXPECT_EQ(*res, *expected_res);
}

TEST(TestAlg, term_replace_term) {
    auto term1 = parse_term("f(a g(b c))").value();
    auto term2 = parse_term("f(a g(b c))").value();
    auto pattern = parse_term("a").value();
    auto res = term2->replace_term(pattern, term1);
    EXPECT_EQ(*res, *parse_term("f(f(a g(b c)) g(b c))").value());
}

TEST(TestAlg, term_get_all_subterms) {
    auto term = parse_term("f(g(x y) x)").value();
    auto res = term->get_all_subterms();
    EXPECT_EQ(res.size(), 5);
    EXPECT_EQ(res[0].first, TermPos());
    EXPECT_EQ(res[1].first, TermPos({0}));
    EXPECT_EQ(res[2].first, TermPos({0, 0}));
    EXPECT_EQ(res[3].first, TermPos({0, 1}));
    EXPECT_EQ(res[4].first, TermPos({1}));
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


TEST(TestAlg, rewrite_at1) {
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

    auto term = parse_term("|(y &(u v))").value();
    auto pos = parse_pos("(1)").value();
    auto res = rule.apply_at(term, pos, parse_subst("{z: u}").value());

    EXPECT_EQ(*res.value(), *parse_term("|(y |(&(v u) u))").value());
}


TEST(TestAlg, proof_action1) {
    Algebra alg = parse_alg(R"(
        [function]
        & : 2
        | : 2
        ~ : 1
        zero : 0

        [variable]
        x y z u v w

        [axiom]
        (AX1) &(x y) = |(&(y x) z)
    )").value();
    
    SymbolKernel kernel(alg);

    auto lhs = parse_term("|(y &(u v))").value();
    auto rhs = parse_term("|(&(v u) u)").value();

    equation eq = {lhs, rhs};
    proof_action act = parse_proof_action("AX1_L2R (0 1) {z: u}").value();

    auto res = kernel.action(eq, act);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(*eq.lhs, *parse_term("|(y |(&(v u) u))").value());
}

TEST(TestAlg, next_token_machine1) {

    Algebra alg = parse_alg(R"(
        [function]
        & : 2
        | : 2
        ~ : 1
        zero : 0

        [variable]
        x y z u v w

        [axiom]
        (AX1) &(x y) = |(&(y x) z)
        (AX2) &(x y) = |(z w)
    )").value();

    auto machine = NextTokenMachine(alg);

    vector<string> seq = {"&", "(", "x", "y", ")", "=", "&", "(", "&", "(", "u", "u", ")", "y", ")", ":", "AX2_L2R", "(", "1", ")", "{", "w", ":", "|", "(", "zero", "zero", ")", ",", "z", ":", "&", "(", "u", "v", ")", "}", "<EOS>"};
    
    while (machine.get_state() != NextTokenMachine::HALT) {
        cout << machine.to_string() << endl;

        string token;

        // use predefined sequence
        if (seq.size() == 0) {
            // input a string
            break;
        }
        else {
            token = seq[0];
            seq.erase(seq.begin());
        }
        
        if (!machine.push_token(token)) {
            cout << "Invalid token." << endl;
        }
    }

    EXPECT_EQ(machine.get_state(), NextTokenMachine::HALT);
}

TEST(TestAlg, apply_action) {
    Algebra alg = parse_alg(R"(
        [function]
        & : 2
        | : 2
        ~ : 1
        zero : 0

        [variable]
        x y z u v w

        [axiom]
        (AX1) &(x y) = |(&(y x) z)
        (AX2) &(x y) = |(z w)
    )").value();

    SymbolKernel kernel(alg);

    auto eq = parse_equation("&(x y) = &(&(u u) y)").value();
    apply_action(kernel, eq, "AX2_L2R (1) {w: |(zero zero), z: &(u v)}");
    EXPECT_EQ(eq, parse_equation("&(x y) = |(&(u v) |(zero zero))").value());
}