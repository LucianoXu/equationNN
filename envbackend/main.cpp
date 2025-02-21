

#include <iostream>
#include "envbackend.hpp"

using namespace std;
using namespace ualg;

int main() {

    auto term = parse_term("f(x Y)").value();
    auto pattern = parse_term("f(x y)").value();
    
    auto res = match(term, pattern, {"X", "Y"}, {});
    if (res.has_value()) {
        cout << to_string(res.value()) << endl;
    } else {
        cout<<"Not matched!"<<endl;
    }

    cout<<"Hello World!"<<endl;
}