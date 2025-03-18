from ..env import env
from ..model import *
from ..ext_solver import vampire_solve

def test_parse_term():
    term = env.parse_term("f(a g(b c))")
    assert str(term) == "f(a g(b c))"

def test_term_eq():
    term1 = env.parse_term("x")
    assert term1 is not None
    term2 = env.parse_term("x")
    assert term2 is not None
    eq = env.Equation(term1, term2)
    # notice that eq.lhs and eq.rhs are not pointers. So the following assertion is correct.
    assert eq.lhs == eq.rhs


def test_term_construct():
    term1 = env.Term("f", [env.Term("a"), env.Term("g", [env.Term("b"), env.Term("c")])])
    term2 = env.parse_term("f(a g(b c))")
    assert term1 == term2

def test_term_get_head():
    term = env.parse_term("f(a g(b c))")
    assert term is not None
    assert term.head == "f"

def test_term_replace_term():
    term = env.parse_term("f(a g(b c))")
    assert term is not None
    pattern = env.parse_term("g(b c)")
    assert pattern is not None
    new_term = term.replace_term(pattern, term)
    assert new_term == env.parse_term("f(a f(a g(b c)))")

def test_get_subterm():
    term = env.parse_term("f(a g(b c))")
    assert term is not None
    subterm = term.get_subterm([1, 1])
    assert subterm == env.parse_term("c")

# an example of an algebra
algebra_code = '''
    [function]
    & : 2
    | : 2
    ~ : 1
    zero : 0

    [variable]
    x y z u v w

    [axiom]
    (AX1) &(x y) = |(&(y x) z)
'''

def test_parse_alg():
    alg = env.parse_alg(algebra_code)
    assert alg is not None

def test_tokenizer1():
    alg = env.parse_alg(algebra_code)
    assert alg is not None
    tokenizer = env.Tokenizer(alg)
    assert tokenizer.vocab == ['<PAD>', '<STT>', '</STT>', '<ACT>', '</ACT>', '(', ')', ':', '{', '}', ',', '=', '&', '|', '~', 'zero', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', 'SUBST', 'AX1_L2R', 'AX1_R2L']

    encoding = tokenizer.encode("zero = x")
    assert tokenizer.decode(encoding) == "zero = x "

def test_tokenizer2():
    algebra_code = '''
    [function]
    & : 2
    | : 2
    ~ : 1
    simp : 2
    sconj : 2

    [variable]
    x y z u v w

    [axiom]
    (commM) &(x y) = &(y x)
    (commJ) |(x y) = |(y x)
    (assocM) &(x &(y z)) = &(&(x y) z)
    (assocJ) |(x |(y z)) = |(|(x y) z)
    (absorbM) &(x |(x y)) = x
    (absorbJ) |(x &(x y)) = x
    (negneg) ~(~(x)) = x
    (deMorgenM) ~(&(x y)) = |(~(x) ~(y))
    (deMorgenJ) ~(|(x y)) = &(~(x) ~(y))
    (OML) |(x y) = |(&(|(x y) x) &(|(x y) ~(x)))
    (SIMP) simp(x y) = |(~(x) &(x y))
    (SCONJ) sconj(x y) = &(x |(~(x) y))'''

    alg = env.parse_alg(algebra_code)
    assert alg is not None
    tokenizer = env.Tokenizer(alg)
    assert tokenizer.vocab == ['<PAD>', '<STT>', '</STT>', '<ACT>', '</ACT>', '(', ')', ':', '{', '}', ',', '=', '&', '|', '~', 'simp', 'sconj', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', 'SUBST', 'commM', 'commJ', 'assocM_L2R', 'assocM_R2L', 'assocJ_L2R', 'assocJ_R2L', 'absorbM_L2R', 'absorbM_R2L', 'absorbJ_L2R', 'absorbJ_R2L', 'negneg_L2R', 'negneg_R2L', 'deMorgenM_L2R', 'deMorgenM_R2L', 'deMorgenJ_L2R', 'deMorgenJ_R2L', 'OML_L2R', 'OML_R2L', 'SIMP_L2R', 'SIMP_R2L', 'SCONJ_L2R', 'SCONJ_R2L']

def test_next_token_machine():
    alg = env.parse_alg(algebra_code)
    assert alg is not None
    machine = env.NextTokenMachine(alg, True)
    assert machine.push_token("<STT>")
    assert machine.push_token("zero")
    assert machine.push_token("=")
    assert machine.push_token("x")
    assert machine.state == env.NextTokenMachine.State.END_STT

def test_next_token_machine_copy():
    alg = env.parse_alg(algebra_code)
    assert alg is not None
    machine = env.NextTokenMachine(alg, True)
    machine_copy = machine.copy()

    assert machine.push_token("<STT>")
    assert machine.push_token("zero")
    assert machine.push_token("=")
    assert machine.push_token("x")
    assert machine.state == env.NextTokenMachine.State.END_STT

    assert machine_copy.push_token("<STT>")
    assert machine_copy.push_token("zero")
    assert machine_copy.push_token("=")
    assert machine_copy.push_token("x")
    assert machine_copy.state == env.NextTokenMachine.State.END_STT



def test_apply_action():
    algebra_code = '''
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
    '''
    alg = env.parse_alg(algebra_code)
    assert alg is not None
    kernel = env.SymbolKernel(alg)
    stt = env.parse_proof_state("<STT> &(x y) = &(&(u u) y) </STT>")
    assert stt is not None
    kernel.action_by_code(stt, "<ACT> AX2_L2R (1) {w: |(zero zero), z: &(u v)} </ACT>")
    assert stt == env.parse_proof_state("<STT> &(x y) = |(&(u v) |(zero zero)) </STT>")
    
def test_gen_valid_check():
    alg_code = '''
    [function]
    & : 2
    | : 2
    ~ : 1
    zero : 0

    [variable]
    x y z u v w

    [axiom]
    (AX1) x = &(x z)
    (AX2) &(x y) = |(&(y x) z)
    '''
    scenario = Scenario(alg_code)
    
    # a random model
    model_args = SmallArgs(vocab_size=scenario.tokenizer.get_vocab_size(), context_length=160)
    model = Llama3(model_args, device='cuda')

    for i in range(100):
        res = generate(model, scenario, "", True)
        print(res)
        if not env.check_step(scenario.kernel, res):
            raise Exception("Invalid action")

def test_push_string():
    alg_code = '''
    [function]
    * : 2

    [variable]
    x y z u v w

    [axiom]
    (AX1) *(x y) = *(*(y y) x)
    '''
    scenario = Scenario(alg_code)

    machine = env.NextTokenMachine(scenario.alg, True)

    assert machine.push_string("<STT> *(*(*(x x) x) x)")

def test_vampire_solver():

    alg_code = '''
    [function]
    & : 2
    | : 2
    ~ : 1
    zero : 0

    [variable]
    x y z u v w

    [axiom]
    (AX1) x = &(x z)
    (AX2) &(x y) = |(&(y x) z)
    '''

    scenario = Scenario(alg_code)

    problem = env.parse_equation("&(x y) = |(&(y x) z)")
    assert problem is not None

    res = vampire_solve("vampire", scenario, problem)
    assert res.is_true == True