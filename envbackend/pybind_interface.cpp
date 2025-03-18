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
        .def(py::init<const equation&>())
        .def_readwrite("lhs", &equation::lhs)
        .def_readwrite("rhs", &equation::rhs)
        .def_property_readonly("size", &equation::get_size)
        .def("__str__", &equation::to_string)
        .def("__eq__", &equation::operator==)
        .def("__repr__", &equation::to_repr);

    py::class_<Signature, shared_ptr<Signature>>(m, "Signature")
        .def_property_readonly("func_symbols", &Signature::get_func_symbols)
        .def_property_readonly("variables", &Signature::get_variables)
        .def("term_valid", &Signature::term_valid)
        .def("__str__", &Signature::to_string);

    py::class_<Algebra, shared_ptr<Algebra>>(m, "Algebra")
        .def_property_readonly("signature", &Algebra::get_signature)
        .def("__str__", &Algebra::to_string);

    py::enum_<ACT_RESULT>(m, "ACT_RESULT")
        .value("FAILURE", ACT_RESULT::FAILURE)
        .value("SUCCESS", ACT_RESULT::SUCCESS)
        .export_values();

    py::class_<SymbolKernel>(m, "SymbolKernel")
        .def(py::init<const Algebra&>())
        .def("action", &SymbolKernel::action)
        .def("action_by_code", &SymbolKernel::action_by_code);

    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<const Algebra&>())
        .def_property_readonly("vocab", &Tokenizer::get_vocab)
        .def("get_vocab_size", &Tokenizer::get_vocab_size)
        .def("get_token", &Tokenizer::get_token)
        .def("get_encoding", &Tokenizer::get_encoding)
        .def("get_pos_int_encoding", &Tokenizer::get_pos_int_encoding)
        .def("is_valid_token", &Tokenizer::is_valid_token)
        .def("encode", &Tokenizer::encode)
        .def("decode", &Tokenizer::decode);

    py::class_<NextTokenMachine> next_tok_machine(m, "NextTokenMachine");
    next_tok_machine
        .def(py::init<const Algebra&, bool>())
        .def(py::init<const NextTokenMachine&>())
        .def("copy", &NextTokenMachine::copy)
        .def_property_readonly("encodings", &NextTokenMachine::get_encodings)
        .def_property_readonly("input", &NextTokenMachine::get_input)
        .def_property_readonly("valid_next_tokens", &NextTokenMachine::get_valid_next_tokens)
        .def("push_token", py::overload_cast<int>(&NextTokenMachine::push_token))
        .def("push_token", py::overload_cast<string>(&NextTokenMachine::push_token))
        .def("push_encodings", &NextTokenMachine::push_encodings)
        .def("push_string", &NextTokenMachine::push_string)
        .def_property_readonly("state", &NextTokenMachine::get_state)
        .def("__str__", &NextTokenMachine::to_string);

    py::enum_<NextTokenMachine::State>(next_tok_machine, "State")
        .value("START_STT", NextTokenMachine::State::START_STT)
        .value("LHS", NextTokenMachine::State::LHS)
        .value("EQ", NextTokenMachine::State::EQ)
        .value("RHS", NextTokenMachine::State::RHS)
        .value("END_STT", NextTokenMachine::State::END_STT)
        .value("START_ACT", NextTokenMachine::State::START_ACT)
        .value("ACT_NAME", NextTokenMachine::State::ACT_NAME)
        .value("POS", NextTokenMachine::State::POS)
        .value("SUBST", NextTokenMachine::State::SUBST)
        .value("SUBST_TERM", NextTokenMachine::State::SUBST_TERM)
        .value("SUBST_COLON", NextTokenMachine::State::SUBST_COLON)
        .value("COMMA", NextTokenMachine::State::COMMA)
        .value("SUBST_ACT_NAME", NextTokenMachine::State::SUBST_ACT_NAME)
        .value("SUBST_ACT_TERM", NextTokenMachine::State::SUBST_ACT_TERM)
        .value("END_ACT", NextTokenMachine::State::END_ACT)
        .value("HALT", NextTokenMachine::State::HALT)
        .export_values();

    py::class_<proof_state>(m, "proof_state")
        .def(py::init<equation>())
        .def(py::init<const proof_state&>())
        .def_readwrite("eq", &proof_state::eq)
        .def("__eq__", &proof_state::operator==)
        .def("__str__", &proof_state::to_string)
        .def("__repr__", &proof_state::to_repr);

    py::class_<proof_action>(m, "proof_action")
        .def(py::init<const string&, const TermPos&, const subst&>())
        .def_readwrite("rule_name", &proof_action::rule_name)
        .def_readwrite("pos", &proof_action::pos)
        .def_readwrite("subst", &proof_action::spec_subst)
        .def("__str__", &proof_action::to_string)
        .def("__repr__", &proof_action::to_repr);

    py::class_<proof_step>(m, "proof_step")
        .def(py::init<proof_state, proof_action>())
        .def_readwrite("stt", &proof_step::stt)
        .def_readwrite("act", &proof_step::act)
        .def("__str__", &proof_step::to_string)
        .def("__repr__", &proof_step::to_repr);

    m.def("parse_term", &parse_term, "A function that parses the term code.");
    m.def("parse_equation", &parse_equation, "A function that parses the equation code.");
    m.def("parse_alg", &parse_alg, "A function that parses the algebra code.");
    m.def("parse_proof_state", &parse_proof_state, "A function that parses the proof state code.");
    m.def("parse_proof_action", &parse_proof_action, "A function that parses the proof action code.");
    m.def("parse_proof_step", &parse_proof_step, "A function that parses the proof step code.");
    m.def("check_step", &check_step, "A function that checks whether the action is valid.");

    m.def("vampire_problem_encode", &vampire_problem_encode, "A function that generates the Vampire encode for the given problem.");
}