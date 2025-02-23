#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "envbackend.hpp"

using namespace std;
using namespace ualg;

namespace py = pybind11;


PYBIND11_MODULE(envbackend, m) {
    m.doc() = "The C++ backend for universal algebra and term rewriting.";

    py::class_<Term, shared_ptr<Term>>(m, "Term")
        .def(py::init<const string&>())
        .def(py::init<const string&, const vector<TermPtr>&>())
        .def(py::init<const string&, vector<TermPtr>&&>())
        .def_property_readonly("head", &Term::get_head)
        .def_property_readonly("args", &Term::get_args)
        .def("__eq__", &Term::operator==)
        .def("__ne__", &Term::operator!=)
        .def("get_term_size", &Term::get_term_size)
        .def("is_atomic", &Term::is_atomic)
        .def("get_variables", &Term::get_variables)
        .def("get_subterm", &Term::get_subterm)
        .def("replace_term", &Term::replace_term)
        .def("replace_at", &Term::replace_at)
        .def("__str__", &Term::to_string)
        .def("__repr__", &Term::to_repr);

    py::class_<equation>(m, "Equation")
        .def(py::init<TermPtr, TermPtr>())
        .def("__str__", &equation::to_string)
        .def("__eq__", &equation::operator==)
        .def("__repr__", &equation::to_repr);

    py::class_<Signature, shared_ptr<Signature>>(m, "Signature");
    py::class_<Algebra, shared_ptr<Algebra>>(m, "Algebra");

    py::enum_<ACT_RESULT>(m, "ACT_RESULT")
        .value("FAILURE", ACT_RESULT::FAILURE)
        .value("SUCCESS", ACT_RESULT::SUCCESS)
        .export_values();

    py::class_<SymbolKernel>(m, "SymbolKernel")
        .def(py::init<const Algebra&>())
        .def("action", &SymbolKernel::action);

    py::class_<NextTokenMachine> next_tok_machine(m, "NextTokenMachine");
    next_tok_machine.def(py::init<const Algebra&>())
        .def("get_valid_next_tokens", &NextTokenMachine::get_valid_next_tokens)
        .def("push_token", py::overload_cast<int>(&NextTokenMachine::push_token))
        .def("push_token", py::overload_cast<string>(&NextTokenMachine::push_token))
        .def_property_readonly("state", &NextTokenMachine::get_state)
        .def("__str__", &NextTokenMachine::to_string);

    py::enum_<NextTokenMachine::State>(next_tok_machine, "State")
        .value("LHS", NextTokenMachine::State::LHS)
        .value("EQ", NextTokenMachine::State::EQ)
        .value("RHS", NextTokenMachine::State::RHS)
        .value("COLON", NextTokenMachine::State::COLON)
        .value("RULE_NAME", NextTokenMachine::State::RULE_NAME)
        .value("POS", NextTokenMachine::State::POS)
        .value("SUBST", NextTokenMachine::State::SUBST)
        .value("SUBST_TERM", NextTokenMachine::State::SUBST_TERM)
        .value("SUBST_COLON", NextTokenMachine::State::SUBST_COLON)
        .value("COMMA", NextTokenMachine::State::COMMA)
        .value("EOS", NextTokenMachine::State::EOS)
        .value("HALT", NextTokenMachine::State::HALT)
        .export_values();

    m.def("parse_term", &parse_term, "A function that parses the term code.");
    m.def("parse_equation", &parse_equation, "A function that parses the equation code.");
    m.def("parse_alg", &parse_alg, "A function that parses the algebra code.");
    m.def("get_vocab", &get_vocab, "A function that returns the vocabulary for the algebra.");
    m.def("apply_action", &apply_action, "A function that applies the action to the equation.");
}