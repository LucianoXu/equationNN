

#include <iostream>
#include "envbackend.hpp"

using namespace std;
using namespace ualg;

int main() {

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

    // vector<string> seq = {"&", "(", "x", "y", ")", "=", "&", "(", "&", "(", "u", "u", ")", "y", ")", ":", "AX1_L2R", "(", "0", ")", "{", "z", ":", "zero", "}"};

    vector<string> seq = {"&", "(", "x", "y", ")", "=", "&", "(", "&", "(", "u", "u", ")", "y", ")", ":", "AX2_L2R", "(", "1", ")", "{", "w", ":", "|", "(", "zero", "zero", ")", ",", "z", ":", "&", "(", "u", "v", ")", "}", "<EOS>"};

    
    while (machine.get_state() != NextTokenMachine::HALT) {
        cout << machine.to_string() << endl;

        string token;

        // use predefined sequence
        if (seq.size() == 0) {
            // input a string
            cin >> token;
        }
        else {
            token = seq[0];
            seq.erase(seq.begin());
        }
        
        if (!machine.push_token(token)) {
            cout << "Invalid token." << endl;
        }
    }

    cout << "HALT" << endl;

}