#include "parser.hpp"

namespace ualg {


    class ENVBACKENDTermBuilder : public ENVBACKENDBaseListener {
    public:
        ENVBACKENDTermBuilder();
    
        TermPtr get_root();

        Algebra get_algebra();

        void exitAlg(ENVBACKENDParser::AlgContext *ctx) override;
    
        // Called when entering a 'Identifier' node
        void exitIdentifier(ENVBACKENDParser::IdentifierContext *ctx) override;
    
        // Called when entering a 'Application' node
        void exitApplication(ENVBACKENDParser::ApplicationContext *ctx) override;
    
        // Called when entering a 'Func' node
        void exitFunc(ENVBACKENDParser::FuncContext *ctx) override;
    
        // Called when entering a 'Axiom' node
        void exitAxiom(ENVBACKENDParser::AxiomContext *ctx) override;

        void exitNameFunc(ENVBACKENDParser::NameFuncContext *ctx) override;

        void exitSymbolFunc(ENVBACKENDParser::SymbolFuncContext *ctx) override;
    
    private:
        std::stack<std::string> funcnames;
        std::stack<TermPtr> term_stack;

        // the stack for the algebra specification
        std::vector<Algebra::func_symbol> functions;
        std::vector<std::string> variables;
        std::vector<std::tuple<std::string, TermPtr, TermPtr>> axioms;
        Algebra algebra;
    };
    
    ENVBACKENDTermBuilder::ENVBACKENDTermBuilder() {}
    
    TermPtr ENVBACKENDTermBuilder::get_root() {
        if (!term_stack.empty()) {
            return std::move(term_stack.top());
        }
        throw std::runtime_error("No root node found.");
    }

    Algebra ENVBACKENDTermBuilder::get_algebra() {
        return algebra;
    }

    void ENVBACKENDTermBuilder::exitAlg(ENVBACKENDParser::AlgContext *ctx) {
        for (auto name : ctx->NAME()) {
            variables.push_back(name->getText());
        }

        algebra = Algebra(functions, variables, axioms);
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
        TermPtr right_term = term_stack.top();
        term_stack.pop();
        TermPtr left_term = term_stack.top();
        term_stack.pop();
        axioms.push_back(std::make_tuple(axiom_name, left_term, right_term));
    }


    void ENVBACKENDTermBuilder::exitNameFunc(ENVBACKENDParser::NameFuncContext *ctx) {
        funcnames.push(ctx->NAME()->getText());
    }

    void ENVBACKENDTermBuilder::exitSymbolFunc(ENVBACKENDParser::SymbolFuncContext *ctx) {
        funcnames.push(ctx->SYMBOL()->getText());
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
            return treeBuilder.get_root();            
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

} // namespace ualg