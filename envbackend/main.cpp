

#include <iostream>
#include "envbackend.hpp"

using namespace std;
using namespace ualg;

int main() {
    auto x = parse_term("f(x y)");
    cout<<x.value()->to_string()<<endl;

    auto alg = parse_alg(R"(
        [function]
        & : 2
        | : 2
        ~ : 1

        [variable]
        x y z u v w

        [axiom]
        (AX1) &(x y) = &(y x)
        (AX2) &(x |(y z)) = |(&(x y) &(x z))

    )").value();
    cout << alg.to_string() << endl;

    cout<<"Hello World!"<<endl;
}