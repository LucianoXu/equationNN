#include "parser.hpp"

using namespace std;
using namespace antlr4;

namespace ualg {

    class ENVBACKENDTermBuilder : public ENVBACKENDBaseListener {
    public:
        ENVBACKENDTermBuilder();
    
        equation get_equation();

        TermPtr get_term();

        TermPos get_pos();

        subst get_subst();

        Signature get_signature();

        Algebra get_algebra();

        proof_step get_proof_step();

        proof_action get_proof_action();

        void exitProofstep(ENVBACKENDParser::ProofstepContext *ctx) override;

        void exitRuleAction(ENVBACKENDParser::RuleActionContext *ctx) override;

        void exitSubstAction(ENVBACKENDParser::SubstActionContext *ctx) override;

        void exitAlg(ENVBACKENDParser::AlgContext *ctx) override;

        void exitSig(ENVBACKENDParser::SigContext *ctx) override;

        void exitEmptySubst(ENVBACKENDParser::EmptySubstContext *ctx) override;

        void exitNonEmptySubst(ENVBACKENDParser::NonEmptySubstContext *ctx) override;
    
        // Called when entering a 'Identifier' node
        void exitIdentifier(ENVBACKENDParser::IdentifierContext *ctx) override;
    
        // Called when entering a 'Application' node
        void exitApplication(ENVBACKENDParser::ApplicationContext *ctx) override;
    
        // Called when entering a 'Func' node
        void exitFunc(ENVBACKENDParser::FuncContext *ctx) override;
    
        // Called when entering a 'Axiom' node
        void exitAxiom(ENVBACKENDParser::AxiomContext *ctx) override;

        void exitEquation(ENVBACKENDParser::EquationContext *ctx) override;

        void exitNameFunc(ENVBACKENDParser::NameFuncContext *ctx) override;

        void exitSymbolFunc(ENVBACKENDParser::SymbolFuncContext *ctx) override;

        void exitPos(ENVBACKENDParser::PosContext *ctx) override;
    
    private:
        stack<string> funcnames;
        stack<TermPtr> term_stack;
        stack<equation> eq_stack;
        stack<subst> subst_stack;
        stack<TermPos> pos_stack;

        // the stack for the algebra specification
        vector<Signature::func_symbol> functions;
        vector<string> variables;
        vector<pair<string, equation>> axioms;
        Signature sig;
        Algebra algebra;
        proof_action act;
        proof_step step;
    };
    
    ENVBACKENDTermBuilder::ENVBACKENDTermBuilder() {}
    
    equation ENVBACKENDTermBuilder::get_equation() {
        return std::move(eq_stack.top());
    }

    TermPtr ENVBACKENDTermBuilder::get_term() {
        if (!term_stack.empty()) {
            return std::move(term_stack.top());
        }
        throw runtime_error("No root node found.");
    }

    TermPos ENVBACKENDTermBuilder::get_pos() {
        return std::move(pos_stack.top());
    }

    subst ENVBACKENDTermBuilder::get_subst() {
        return std::move(subst_stack.top());
    }

    Signature ENVBACKENDTermBuilder::get_signature() {
        return sig;
    }

    Algebra ENVBACKENDTermBuilder::get_algebra() {
        return algebra;
    }

    proof_action ENVBACKENDTermBuilder::get_proof_action() {
        return act;
    }

    proof_step ENVBACKENDTermBuilder::get_proof_step() {
        return step;
    }

    void ENVBACKENDTermBuilder::exitRuleAction(ENVBACKENDParser::RuleActionContext *ctx) {
        act = {ctx->NAME()->getText(), pos_stack.top(), subst_stack.top()};
        pos_stack.pop();
        subst_stack.pop();
    }

    void ENVBACKENDTermBuilder::exitSubstAction(ENVBACKENDParser::SubstActionContext *ctx) {
        act = {"SUBST", TermPos(), subst{{ctx->NAME()->getText(), term_stack.top()}}};
        term_stack.pop();
    }

    void ENVBACKENDTermBuilder::exitProofstep(ENVBACKENDParser::ProofstepContext *ctx) {
        step = {eq_stack.top(), act};
        eq_stack.pop();
    }

    void ENVBACKENDTermBuilder::exitAlg(ENVBACKENDParser::AlgContext *ctx) {

        algebra = Algebra(sig, axioms);
    }

    void ENVBACKENDTermBuilder::exitSig(ENVBACKENDParser::SigContext *ctx) {
        for (auto name : ctx->NAME()) {
            variables.push_back(name->getText());
        }

        sig = Signature(functions, variables);
    }


    void ENVBACKENDTermBuilder::exitEmptySubst(ENVBACKENDParser::EmptySubstContext *ctx) {
        subst_stack.push({});
    }

    void ENVBACKENDTermBuilder::exitNonEmptySubst(ENVBACKENDParser::NonEmptySubstContext *ctx) {
        subst subst_data = {};
        for (int i = ctx->NAME().size() - 1; i >= 0; --i) {
            string var_name = ctx->NAME(i)->getText();
            subst_data[var_name] = term_stack.top();
            term_stack.pop();
        }
        subst_stack.push(subst_data);
    }
    
    void ENVBACKENDTermBuilder::exitIdentifier(ENVBACKENDParser::IdentifierContext *ctx) {
        string var_name = funcnames.top();
        funcnames.pop();
        term_stack.push(make_shared<Term>(var_name));
    }
    
    void ENVBACKENDTermBuilder::exitApplication(ENVBACKENDParser::ApplicationContext *ctx) {
        string function_name = funcnames.top();
        funcnames.pop();
        vector<TermPtr> arguments;
    
        // Iterate through the arguments
        for (int i = 0; i < ctx->expr().size(); ++i) {
            arguments.insert(arguments.begin(), term_stack.top());
            term_stack.pop();
        }
        
        // I don't understand why using std::move(arguments) leads to unexpected behavior
        auto application_term = make_shared<Term>(function_name, arguments);
        term_stack.push(application_term);
    }
    
    void ENVBACKENDTermBuilder::exitFunc(ENVBACKENDParser::FuncContext *ctx) {
        string func_name = funcnames.top();
        funcnames.pop();
        unsigned arity = stoi(ctx->INT()->getText());
        functions.push_back({func_name, arity});
    }
    
    void ENVBACKENDTermBuilder::exitAxiom(ENVBACKENDParser::AxiomContext *ctx) {
        string axiom_name = ctx->NAME()->getText();
        equation eq = eq_stack.top();
        axioms.push_back(make_pair(axiom_name, eq));
    }

    void ENVBACKENDTermBuilder::exitEquation(ENVBACKENDParser::EquationContext *ctx) {
        TermPtr right_term = term_stack.top();
        term_stack.pop();
        TermPtr left_term = term_stack.top();
        term_stack.pop();
        eq_stack.push({left_term, right_term});
    }

    void ENVBACKENDTermBuilder::exitNameFunc(ENVBACKENDParser::NameFuncContext *ctx) {
        funcnames.push(ctx->NAME()->getText());
    }

    void ENVBACKENDTermBuilder::exitSymbolFunc(ENVBACKENDParser::SymbolFuncContext *ctx) {
        funcnames.push(ctx->SYMBOL()->getText());
    }

    void ENVBACKENDTermBuilder::exitPos(ENVBACKENDParser::PosContext *ctx) {
        TermPos pos;
        for (int i = 0; i < ctx->INT().size(); ++i) {
            pos.push_back(stoi(ctx->INT(i)->getText()));
        }
        pos_stack.push(pos);
    }

    vector<string> parse_tokens(const string& code) {
        
        ANTLRInputStream input(code);
        ENVBACKENDLexer lexer(&input);
        CommonTokenStream tokens(&lexer);

        tokens.fill();
        
        vector<string> token_list;
        for (auto token : tokens.getTokens()) {
            token_list.push_back(token->getText());
        }
        token_list.pop_back(); // remove the last token, which is <EOF>
        return token_list;
    }


    template<typename T, typename ParserMethod, typename BuilderMethod>
    optional<T> parse_generic(const string &code, ParserMethod parseFn, BuilderMethod getterFn) {
    
        ANTLRInputStream input(code);
        ENVBACKENDLexer lexer(&input);
        CommonTokenStream tokens(&lexer);
        tokens.fill();
    
        ENVBACKENDParser parser(&tokens);
        tree::ParseTree* tree = (parser.*parseFn)();
        
        // Check for errors
        if (parser.getNumberOfSyntaxErrors() > 0 || tokens.LA(1) != Token::EOF) {
            return nullopt;
        }

        ENVBACKENDTermBuilder treeBuilder;
        antlr4::tree::ParseTreeWalker::DEFAULT.walk(&treeBuilder, tree);

        // Retrieve the result
        return optional<T>((treeBuilder.*getterFn)());
    }


    optional<equation> parse_equation(const string& code) {
        return parse_generic<equation>(code, &ENVBACKENDParser::equation, &ENVBACKENDTermBuilder::get_equation);
    }

    optional<TermPtr> parse_term(const string& code) {
        return parse_generic<TermPtr>(code, &ENVBACKENDParser::expr, &ENVBACKENDTermBuilder::get_term);
    }


    optional<TermPos> parse_pos(const string& code) {
        return parse_generic<TermPos>(code, &ENVBACKENDParser::pos, &ENVBACKENDTermBuilder::get_pos);
    }

    optional<subst> parse_subst(const string& code) {
        return parse_generic<subst>(code, &ENVBACKENDParser::subst, &ENVBACKENDTermBuilder::get_subst);
    }

    optional<Signature> parse_signature(const string& code) {
        return parse_generic<Signature>(code, &ENVBACKENDParser::sig, &ENVBACKENDTermBuilder::get_signature);
    }

    optional<Algebra> parse_alg(const string& code) {
        return parse_generic<Algebra>(code, &ENVBACKENDParser::alg, &ENVBACKENDTermBuilder::get_algebra);
    }

    optional<proof_action> parse_proof_action(const string& code) {
        return parse_generic<proof_action>(code, &ENVBACKENDParser::proofaction, &ENVBACKENDTermBuilder::get_proof_action);
    }


    optional<proof_step> parse_proof_step(const string& code) {
        return parse_generic<proof_step>(code, &ENVBACKENDParser::proofstep, &ENVBACKENDTermBuilder::get_proof_step);
    }
} // namespace ualg