#include "parser.hpp"

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

        void exitProofaction(ENVBACKENDParser::ProofactionContext *ctx) override;

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
        std::stack<std::string> funcnames;
        std::stack<TermPtr> term_stack;
        std::stack<equation> eq_stack;
        std::stack<subst> subst_stack;
        std::stack<TermPos> pos_stack;

        // the stack for the algebra specification
        std::vector<Signature::func_symbol> functions;
        std::vector<std::string> variables;
        std::vector<std::pair<std::string, equation>> axioms;
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
        throw std::runtime_error("No root node found.");
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

    void ENVBACKENDTermBuilder::exitProofaction(ENVBACKENDParser::ProofactionContext *ctx) {
        act = {ctx->NAME()->getText(), pos_stack.top(), subst_stack.top()};
        pos_stack.pop();
        subst_stack.pop();
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
            std::string var_name = ctx->NAME(i)->getText();
            subst_data[var_name] = term_stack.top();
            term_stack.pop();
        }
        subst_stack.push(subst_data);
    }
    
    void ENVBACKENDTermBuilder::exitIdentifier(ENVBACKENDParser::IdentifierContext *ctx) {
        std::string var_name = funcnames.top();
        funcnames.pop();
        term_stack.push(std::make_shared<Term>(var_name));
    }
    
    void ENVBACKENDTermBuilder::exitApplication(ENVBACKENDParser::ApplicationContext *ctx) {
        std::string function_name = funcnames.top();
        funcnames.pop();
        std::vector<TermPtr> arguments;
    
        // Iterate through the arguments
        for (int i = 0; i < ctx->expr().size(); ++i) {
            arguments.insert(arguments.begin(), term_stack.top());
            term_stack.pop();
        }
    
        auto application_term = std::make_shared<Term>(function_name, std::move(arguments));
        term_stack.push(application_term);
    }
    
    void ENVBACKENDTermBuilder::exitFunc(ENVBACKENDParser::FuncContext *ctx) {
        std::string func_name = funcnames.top();
        funcnames.pop();
        unsigned arity = std::stoi(ctx->INT()->getText());
        functions.push_back({func_name, arity});
    }
    
    void ENVBACKENDTermBuilder::exitAxiom(ENVBACKENDParser::AxiomContext *ctx) {
        std::string axiom_name = ctx->NAME()->getText();
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
            pos.push_back(std::stoi(ctx->INT(i)->getText()));
        }
        pos_stack.push(pos);
    }

    std::optional<equation> parse_equation(const std::string& code) {
        using namespace antlr4;
        
        ANTLRInputStream input(code);
        ENVBACKENDLexer lexer(&input);
        CommonTokenStream tokens(&lexer);

        tokens.fill();
        
        ENVBACKENDParser parser(&tokens);
        tree::ParseTree *tree = parser.equation();

        // Create the tree builder
        ENVBACKENDTermBuilder treeBuilder;
        // Check for errors
        if (parser.getNumberOfSyntaxErrors() == 0) {
            antlr4::tree::ParseTreeWalker::DEFAULT.walk(&treeBuilder, tree);

            // Retrieve the root of the custom tree
            return treeBuilder.get_equation();            
        } else {
            return std::nullopt;
        }
    }

    std::optional<TermPtr> parse_term(const std::string& code) {
        using namespace antlr4;
        
        ANTLRInputStream input(code);
        ENVBACKENDLexer lexer(&input);
        CommonTokenStream tokens(&lexer);

        tokens.fill();
        
        ENVBACKENDParser parser(&tokens);
        tree::ParseTree *tree = parser.expr();

        // Create the tree builder
        ENVBACKENDTermBuilder treeBuilder;
        // Check for errors
        if (parser.getNumberOfSyntaxErrors() == 0) {
            antlr4::tree::ParseTreeWalker::DEFAULT.walk(&treeBuilder, tree);

            // Retrieve the root of the custom tree
            return treeBuilder.get_term();            
        } else {
            return std::nullopt;
        }
    }


    std::optional<TermPos> parse_pos(const std::string& code) {
        using namespace antlr4;
        
        ANTLRInputStream input(code);
        ENVBACKENDLexer lexer(&input);
        CommonTokenStream tokens(&lexer);

        tokens.fill();
        
        ENVBACKENDParser parser(&tokens);
        tree::ParseTree *tree = parser.pos();

        // Create the tree builder
        ENVBACKENDTermBuilder treeBuilder;
        // Check for errors
        if (parser.getNumberOfSyntaxErrors() == 0) {
            antlr4::tree::ParseTreeWalker::DEFAULT.walk(&treeBuilder, tree);

            // Retrieve the root of the custom tree
            return treeBuilder.get_pos();            
        } else {
            return std::nullopt;
        }
    }

    std::optional<subst> parse_subst(const std::string& code) {
        using namespace antlr4;
        
        ANTLRInputStream input(code);
        ENVBACKENDLexer lexer(&input);
        CommonTokenStream tokens(&lexer);

        tokens.fill();
        
        ENVBACKENDParser parser(&tokens);
        tree::ParseTree *tree = parser.subst();

        // Create the tree builder
        ENVBACKENDTermBuilder treeBuilder;
        // Check for errors
        if (parser.getNumberOfSyntaxErrors() == 0) {
            antlr4::tree::ParseTreeWalker::DEFAULT.walk(&treeBuilder, tree);

            // Retrieve the root of the custom tree
            return treeBuilder.get_subst();            
        } else {
            return std::nullopt;
        }
    }

    std::optional<Signature> parse_signature(const std::string& code) {
        using namespace antlr4;
        
        ANTLRInputStream input(code);
        ENVBACKENDLexer lexer(&input);
        CommonTokenStream tokens(&lexer);

        tokens.fill();
        
        ENVBACKENDParser parser(&tokens);
        tree::ParseTree *tree = parser.sig();

        // Create the tree builder
        ENVBACKENDTermBuilder treeBuilder;
        // Check for errors
        if (parser.getNumberOfSyntaxErrors() == 0) {
            antlr4::tree::ParseTreeWalker::DEFAULT.walk(&treeBuilder, tree);

            // Retrieve the root of the custom tree
            return treeBuilder.get_signature();            
        } else {
            return std::nullopt;
        }
    }

    std::optional<Algebra> parse_alg(const std::string& code) {
        using namespace antlr4;
        
        ANTLRInputStream input(code);
        ENVBACKENDLexer lexer(&input);
        CommonTokenStream tokens(&lexer);

        tokens.fill();
        
        ENVBACKENDParser parser(&tokens);
        tree::ParseTree *tree = parser.alg();

        // Create the tree builder
        ENVBACKENDTermBuilder treeBuilder;
        // Check for errors
        if (parser.getNumberOfSyntaxErrors() == 0) {
            antlr4::tree::ParseTreeWalker::DEFAULT.walk(&treeBuilder, tree);

            // Retrieve the root of the custom tree
            return treeBuilder.get_algebra();            
        } else {
            return std::nullopt;
        }
    }

    std::optional<proof_action> parse_proof_action(const std::string& code) {
        using namespace antlr4;
        
        ANTLRInputStream input(code);
        ENVBACKENDLexer lexer(&input);
        CommonTokenStream tokens(&lexer);

        tokens.fill();
        
        ENVBACKENDParser parser(&tokens);
        tree::ParseTree *tree = parser.proofaction();

        // Create the tree builder
        ENVBACKENDTermBuilder treeBuilder;
        // Check for errors
        if (parser.getNumberOfSyntaxErrors() == 0) {
            antlr4::tree::ParseTreeWalker::DEFAULT.walk(&treeBuilder, tree);

            // Retrieve the root of the custom tree
            return treeBuilder.get_proof_action();            
        } else {
            return std::nullopt;
        }
    }

} // namespace ualg