#include "envbackend.hpp"
using namespace std;

namespace ualg {
    Term::Term(const string& head) {
        this->head = head;
        this->args = {};
    }

    Term::Term(const string& head, const vector<TermPtr>& args) {
        this->head = head;
        this->args = args;
    }

    Term::Term(const string& head, vector<TermPtr>&& args) {
        this->head = head;
        this->args = std::move(args);
    }

    const string& Term::get_head() const {
        return this->head;
    }

    const vector<TermPtr>& Term::get_args() const {
        return this->args;
    }

    bool Term::operator == (const Term& other) const {
        if (this->head != other.head) {
            return false;
        }
        if (this->args.size() != other.args.size()) {
            return false;
        }
        for (int i = 0; i < this->args.size(); i++) {
            if (!(*this->args[i] == *other.args[i])) {
                return false;
            }
        }
        return true;
    }

    bool Term::operator != (const Term& other) const {
        return !(*this == other);
    }

    size_t Term::get_term_size() const {
        size_t size = 1;
        for (const auto& arg : args) {
            size += arg->get_term_size();
        }
        return size;
    }

    bool Term::is_atomic() const {
        return args.size() == 0;
    }

    string Term::to_string() const {
        string str = this->head;
        if (args.size() == 0) {
            return str;
        }
        
        str += "(";
        for (int i = 0; i < args.size(); i++) {
            str += args[i]->to_string() + " ";
        }
        if (args.size() > 0) str.pop_back();
        str += ")";
        return str;
    }


    TermPtr Term::get_subterm(const TermPos& pos) const {
        if (pos.size() == 0) {
            return this->shared_from_this();
        }

        // if (pos[0] >= args.size()) {
        //     throw runtime_error("Position out of range.");
        // }

        return args[pos[0]];
    }

    TermPtr Term::replace_term(TermPtr pattern, TermPtr replacement) const {
        if (*this == *pattern) {
            return replacement;
        }

        vector<TermPtr> new_args;
        for (const auto& arg : args) {
            new_args.push_back(arg->replace_term(pattern, replacement));
        }
        return make_shared<const Term>(this->head, std::move(new_args));
    }

    TermPtr Term::replace_at(const TermPos& pos, TermPtr new_subterm) const {
        if (pos.size() == 0) {
            return new_subterm;
        }

        vector<TermPtr> new_args;
        for (int i = 0; i < args.size(); i++) {
            if (i == pos[0]) {
                new_args.push_back(args[i]->replace_at(
                    TermPos(pos.begin() + 1, pos.end()),
                    new_subterm));
            }
            else {
                new_args.push_back(args[i]);
            }
        }
        return make_shared<const Term>(this->head, std::move(new_args));
    }

    string Algebra::to_string() const {
        string res = "[function]\n";
        for (const auto& func : func_symbols) {
            res += func.name + " : " + ::to_string(func.arity) + "\n";
        }
        res += "\n[variable]\n";
        for (const auto& var : variables) {
            res += var + " ";
        }
        res += "\n";
        res += "\n[axiom]\n";
        for (const auto& axiom : axioms) {
            res += "(" + get<0>(axiom) + ") " + get<1>(axiom)->to_string() + " = " + get<2>(axiom)->to_string() + "\n";
        }
        return res;
    }
}
