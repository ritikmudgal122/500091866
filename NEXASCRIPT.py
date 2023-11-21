import string
import sys
# Index class represents the position in the input text
class Index:
    def __init__(self,index,line,column):
        self.index = index
        self.line = line
        self.column = column
# Method to advance the index based on the current character
    def advance(self,curChar=None):
        self.index +=1
        self.column +=1
        if curChar == '\n':
            self.line+=1
            self.column+=0
        return self
 # Method to create a copy of the current index
    def copy(self):
        return Index(self.index,self.line,self.column)

#Tokens Implementation--
# Token types
INT, FLOAT, PLUS, MINUS, MUL, DIV, LPAREN, RPAREN, EOF, IDENTIFIER, KEYWORD, EQUALS, EE, NE, LT, GT, LTE, GTE = 'INT', 'FLOAT', 'PLUS', 'MINUS', 'MUL', 'DIV', 'LPAREN', 'RPAREN', 'EOF', 'IDENTIFIER', 'KEYWORD', 'EQUALS', 'EE', 'NE', 'LT', 'GT', 'LTE', 'GTE'
KEYWORD=['var','if','then','elif','else', 'and', 'or', 'not']
DIGIT = '0123456789'
LETTERS = string.ascii_letters
lettersDigits = LETTERS + DIGIT
# Token class represents a token with a type, value, and position
class Tokens:
    def __init__(self,type,value=None, posStart=None, posEnd=None ):
        self.type = type
        self.value = value
        if posStart:
            self.posStart = posStart.copy()
            self.posEnd = posStart.copy()
            self.posEnd.advance()
        if posEnd:
            self.posEnd = posEnd.copy()
 # Method to check if a token matches a given type and value
    def matches(self,type, value ):
        return self.type == type and self.value == value

    def __repr__(self):
        if self.value:
            return f'{self.type}:{self.value}'
        return f'{self.type}'

#Lexer Implementation--
# Lexer class for tokenizing input text
class lexer:
    def __init__(self,text):
        self.text = text
        self.pos = Index(-1,0,-1)
        self.currentToken = None
        self.advance()
# Method to advance the lexer position
    def advance(self):
        self.pos.advance(self.currentToken)
        self.currentToken = self.text[self.pos.index] if self.pos.index < len(self.text) else None

    # Method to tokenize the input text
    def makeTokens(self): #Detects the type of token the data type is, and makes that specific token type
        tokensList = []
        while self.currentToken != None:
            if self .currentToken in ' \t':
                self.advance()
            elif self.currentToken in DIGIT:
                tokensList.append(self.makeNum())
            elif self.currentToken in LETTERS:
                tokensList.append(self.makeIdentifier())
            elif self.currentToken == '+':
                tokensList.append(Tokens(PLUS, posStart=self.pos))
                self.advance()
            elif self.currentToken == '-':
                tokensList.append(Tokens(MINUS,posStart=self.pos))
                self.advance()
            elif self.currentToken == '*':
                tokensList.append(Tokens(MUL,posStart=self.pos))
                self.advance()
            elif self.currentToken == '/':
                tokensList.append(Tokens(DIV,posStart=self.pos))
                self.advance()
            elif self.currentToken == '(':
                tokensList.append(Tokens(LPAREN,posStart=self.pos))
                self.advance()
            elif self.currentToken == ')':
                tokensList.append(Tokens(RPAREN,posStart=self.pos))
                self.advance()
            elif self.currentToken == '!':
                tok, error = self.makeNotEqual()
                if error:return [], error
                tokensList.append(tok)
            elif self.currentToken == '=':
                tokensList.append(self.makeEquals())
            elif self.currentToken == '<':
                tokensList.append(self.makeLessThan())
            elif self.currentToken == '>':
                tokensList.append(self.makeGreaterThan())
                # ... (Other tokenization rules)
            else:
                # posStart = self.pos.copy()
                char = self.currentToken
                self.advance()
                return [], sys.exit("Illegal Character: " + "'" + char + "'")
        tokensList.append(Tokens(EOF, posStart=self.pos))
        return tokensList, None
# Method to tokenize numerical values
    def makeNum(self): #Helps with the detection if the value is a int or float based on if there is a decimal on the token
        numStr = ''
        decimalCount = 0
        posStart = self.pos.copy()
        while self.currentToken != None and self.currentToken in DIGIT + '.':
            if self.currentToken == '.':
                if decimalCount == 1: break
                decimalCount += 1
                numStr += '.'
            else:
                numStr += self.currentToken
            self.advance()
        if decimalCount == 0:
            return Tokens(INT, int(numStr), posStart, self.pos)
        else:
            return Tokens(FLOAT, float(numStr), posStart, self.pos)
  # Method for tokenizing identifiers
    def makeIdentifier(self):
        identifierStr = ''
        posStart = self.pos.copy()
        while self.currentToken != None and self.currentToken in lettersDigits + '_':
            identifierStr += self.currentToken
            self.advance()
     # Determine the token type based on whether the identifier is a keyword
        tokenType = KEYWORD if identifierStr in KEYWORD else IDENTIFIER
        return Tokens(tokenType, identifierStr,posStart,self.pos)

    def makeNotEqual(self):
        posStart = self.pos.copy()
           # Method for tokenizing '!='
        self.advance()
        if self.currentToken == '=':
            self.advance()
            return Tokens(NE,posStart=posStart, posEnd = self.pos),None
        self.advance()
         # If '=' does not follow '!', raise an error
        sys.exit("'=' goes after '!' ")
 # Method for tokenizing '=' or '=='
    def makeEquals(self):
        tokType = EQUALS
        posStart = self.pos.copy()
        self.advance()
        if self.currentToken == '=':
            self.advance()
            tokType = EE
        return Tokens(tokType,posStart=posStart,posEnd=self.pos)
 # Method for tokenizing '<' or '<='
    def makeLessThan(self):
        tokType = LT
        posStart = self.pos.copy()
        self.advance()
        if self.currentToken == '=':
            self.advance()
            tokType = LTE
        return Tokens(tokType,posStart=posStart,posEnd=self.pos)
  # Method for tokenizing '>' or '>='
    def makeGreaterThan(self):
        tokType = GT
        posStart = self.pos.copy()
        self.advance()
        if self.currentToken == '=':
            self.advance()
            tokType = GTE
        return Tokens(tokType, posStart=posStart, posEnd=self.pos)

#Parser Implementation-
class numNode:  # Node class for numerical values
    def __init__(self,tok):
        self.tok = tok
        self.posStart = self.tok.posStart
        self.posEnd = self.tok.posEnd

    def __repr__(self):
        return f'{self.tok}'

class varAccessNode: # Node class for accessing variables
    def __init__(self,varNameTok):
        self.varNameTok = varNameTok
        self.posStart = self.varNameTok.posStart
        self.posEnd = self.varNameTok.posEnd

class variableAssignNode: # Node class for variable assignment
    def __init__(self,varNameTok,valueNode):
        self.varNameTok = varNameTok
        self.valueNode = valueNode
        self.posStart = self.varNameTok.posStart
        self.posEnd = self.valueNode.posEnd

class operationNode:  # Node class for representing binary operations
    def __init__(self,left,opToken,right):
        self.left = left
        self.opToken = opToken
        self.right = right
        self.posStart = self.left.posStart
        self.posEnd = self.right.posEnd

    def __repr__(self):
        return f'({self.left},{self.opToken},{self.right})'

class ifNode: # Node class for representing if statements
    def __init__(self,case, elseCase):
        self.case = case
        self.elseCase = elseCase
        self.posStart = self.case[0][0].posStart   # Set the start and end positions for the ifNode
        self.posEnd = (self.elseCase or self.case[len(self.case)-1][0]).posEnd

class parserResult:
    def __init__(self): # Result class for parser methods
        self.error = None
        self.node = None
        self.advanceCount = 0

    def resgisterAdvance(self):# Method to register advances in the parser
        self.advanceCount += 1

    def register(self,result):
        self.advanceCount += result.advanceCount   # Method to register the result of a sub-parser
        if result.error: self.error = result.error
        return result.node

    def success(self,node):# Method to indicate successful parsing
        self.node = node
        return self

    def failure(self,error):  # Method to indicate parsing failure
        if not self.error or self.advanceCount==0:
            self.error = error
        return self

class parser: #This parser class main job is to specify the grammar of a factor,term,expression,if expression,variable expression
    def __init__(self,tokens):
        self.tokens = tokens
        self.tokensIndex = -1
        self.advance()

    def advance(self): # Method to advance the parser's token index
        self.tokensIndex += 1
        if self.tokensIndex < len(self.tokens):
            self.currentToken = self.tokens[self.tokensIndex]
        return self.currentToken

    def parse(self):  # Method to initiate parsing
        result = self.expr()
        if not result.error and self.currentToken.type != EOF:
            sys.exit("Missing or Expected +,-,*,/,==,!=,<,>,<=,>=,and,or operators ")
        return result

#1. Control flow elements (like if statements)
    def ifExpr(self): # Method to parse if expressions
        res = parserResult()
        case = []
        elseCase = None
        if not self.currentToken.matches(KEYWORD,'if'):  # Check if the current token is the 'if' keyword
            sys.exit("Missing or Expected KEYWORD 'if' ")
        res.resgisterAdvance()
        self.advance()
        condition = res.register(self.expr())# Parse the condition for the 'if' statement
        if res.error:return res
        if not self.currentToken.matches(KEYWORD,'then'):  # Check if the current token is the 'then' keyword
            sys.exit("Missing or Expected KEYWORD 'then' ")
        res.resgisterAdvance()
        self.advance()
        expr = res.register(self.expr()) # Parse the expression for the 'if' statement
        if res.error:return res
        case.append((condition,expr)) # Add the 'if' case to the list
        while self.currentToken.matches(KEYWORD,'elif'):# Parse 'elif' cases if present
            res.resgisterAdvance()
            self.advance()
            condition = res.register(self.expr())  # Parse the condition for the 'elif' statement
            if res.error:return res
            if not self.currentToken.matches(KEYWORD,'then'): # Check if the current token is the 'then' keyword
                sys.exit("Missing or Expected KEYWORD 'then' ")
            res.resgisterAdvance()
            self.advance()
            expr=res.register(self.expr())
            if res.error:return res# Parse the expression for the 'elif' statement
            case.append((condition,expr))
        if self.currentToken.matches(KEYWORD,'else'): # Check for an 'else' case
            res.resgisterAdvance()
            self.advance()
            elseCase = res.register(self.expr()) # Parse the expression for the 'else' case
            if res.error:return res
        return res.success(ifNode(case,elseCase)) # Set the start and end positions for the ifNode

    def variableExpr(self):# Method to parse variable expressions
        res = parserResult()
        tok = self.currentToken
        if tok.type in (INT,FLOAT): # Check if the token is an integer or float
            res.resgisterAdvance()
            self.advance()
            return res.success(numNode(tok))
        elif tok.type == IDENTIFIER: # Check if the token is an identifier
            res.resgisterAdvance()
            self.advance()
            return res.success(varAccessNode(tok))
        elif tok.type == LPAREN: # Check if the token is an opening parenthesis
            res.resgisterAdvance()
            self.advance()
            expr = res.register(self.expr())
            if res.error:
                return res
            if self.currentToken.type == RPAREN: # Check if the closing parenthesis follows
                res.resgisterAdvance()
                self.advance()
                return res.success(expr)
            else:
                sys.exit("Missing or Expected ')' ")
        elif tok.matches(KEYWORD,'if'):# Check if the token is the 'if' keyword
            ifExpr = res.register(self.ifExpr())
            if res.error:return res
            return res.success(ifExpr)
        sys.exit("Missing or Expected identifier, int, float")# Raise an error for other cases

    def factor(self):# Method to parse factors
        result = parserResult()
        tok = self.currentToken
        if tok.type in (PLUS, MINUS): # Check if the token is a unary operator (+ or -)
            result.resgisterAdvance()
            self.advance()
            # factor = result.register(self.factor())
            if result.error:
                return result
            sys.exit("Unary operators are not supported") # Unary operators are not supported, raise an error
        return self.variableExpr()# Return the result of parsing variable expressions

    def term(self):# Method to parse terms
        return self.operation(self.factor,(MUL,DIV))

    def arithmethicExpr(self): # Method to parse arithmetic expressions
        return self.operation(self.term,(PLUS,MINUS))

    def comparisonExpr(self): # Method to parse comparison expressions
        res = parserResult()
        if self.currentToken.matches(KEYWORD,'not'):# Check for the 'not' keyword
            # opToken = self.currentToken
            res.resgisterAdvance()
            self.advance()
            # node = res.register(self.comparisonExpr())
            if res.error: return res
            sys.exit("Unary operators are not supported")# Unary operators are not supported, raise an error

        node = res.register(self.operation(self.arithmethicExpr, (EE,NE,LT,GT,LTE,GTE)))
        if res.error:
            sys.exit("Missing or Expected identifier, int, float,identifier, +, -, or '(' , 'not'")
        return res.success(node)

    def expr(self): # Method to parse expressions
        res = parserResult()
        if self.currentToken.matches(KEYWORD,'var'):# Check for the 'var' keyword for variable assignment
            res.resgisterAdvance()
            self.advance()
            if self.currentToken.type != IDENTIFIER:   # Check for the presence of an identifier
                sys.exit("Missing or Expected Identifier") 
            variableName = self.currentToken
            res.resgisterAdvance()
            self.advance()
            if self.currentToken.type != EQUALS: # Check for the presence of '='
                sys.exit("Missing or Expected '=' ")
            res.resgisterAdvance()
            self.advance()
            expr = res.register(self.expr())# Parse the expression for assignment
            if res.error:
                return res
            return res.success(variableAssignNode(variableName,expr)) # Parse using the operation method with logical operators
        node = res.register(self.operation(self.comparisonExpr,((KEYWORD, "and"),(KEYWORD,"or"))))
        if res.error:
            sys.exit("Missing or Expected 'var','int','float','identifier','+','-' ,'(' ")
        return res.success(node)

    def operation(self,func,op):# Method to parse operations
        res = parserResult()
        left = res.register(func())
        if res.error:
            return res
        while self.currentToken.type in op or (self.currentToken.type, self.currentToken.value) in op:
            opToken = self.currentToken
            res.resgisterAdvance()
            self.advance()
            right = res.register(func())
            if res.error:
                return res
            left = operationNode(left, opToken, right)
        return res.success(left)

#Evaluator/Interpreter-----------------------------------------------------------------------
class runtimeResult:  # Result class for the runtime evaluator
    def __init__(self):
        self.value = None
        self.error = None

    def register(self,res): # Method to register the result of an operation
        if res.error: self.error = res.error
        return res.value

    def success(self,value):# Method to indicate successful evaluation
        self.value = value
        return self

    def failure(self, error): # Method to indicate evaluation failure
        self.error=error
        return self

class Number: # Class to represent numeric values in the interpreter
    def __init__(self,value):
        self.value = value
        self.setPosition()
        self.setContext()

    def setPosition(self, posStart=None, posEnd=None):  # Method to set the position of the Number instance
        self.posStart=posStart
        self.posEnd = posEnd
        return self

    def setContext(self,context=None):# Method to set the context of the Number instance
        self.context = context
        return self

    def addition(self,other):# Methods for arithmetic operations  # Method for addition
        if isinstance(other,Number):
            return Number(self.value + other.value).setContext(self.context), None

    def subtraction(self,other): # Method for subtraction
        if isinstance(other,Number):
            return Number(self.value - other.value).setContext(self.context), None

    def multiplication(self,other):
        if isinstance(other,Number):
            return Number(self.value * other.value).setContext(self.context), None

    def division(self,other):
        if isinstance(other,Number):
            if other.value == 0:
                sys.exit('Division by zero is a Runtime Error')
            return Number(self.value / other.value).setContext(self.context), None

    def getComparisonEqual(self,other):# Methods for comparison operations
        if isinstance(other,Number):# Method for equality comparison
            return Number(int(self.value == other.value)).setContext(self.context), None

    def getComparisonNotEqual(self,other):# Method for inequality comparison
        if isinstance(other,Number):
            return Number(int(self.value != other.value)).setContext(self.context), None

    def getComparisonLessThan(self,other): # Method for less than comparison
        if isinstance(other,Number):
            return Number(int(self.value < other.value)).setContext(self.context), None

    def getComparisonGreaterThan(self,other):# Method for greater than comparison
        if isinstance(other,Number):
            return Number(int(self.value > other.value)).setContext(self.context), None

    def getComparisonLessThanEqual(self,other):# Method for less than or equal to comparison
        if isinstance(other,Number):
            return Number(int(self.value <= other.value)).setContext(self.context), None

    def getComparisonGreaterThanEqual(self,other): # Method for greater than or equal to comparison
        if isinstance(other,Number):
            return Number(int(self.value >= other.value)).setContext(self.context), None
# Logical operation methods
    def andBy(self,other): # Method for logical AND operation
        if isinstance(other,Number):
            return Number(int(self.value and other.value)).setContext(self.context), None

    def orBy(self,other):
        if isinstance(other,Number):
            return Number(int(self.value or other.value)).setContext(self.context), None

    def notBy(self):
        return Number(1 if self.value == 0 else 0).setContext(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.setPosition(self.posStart,self.posEnd)
        copy.setContext(self.context)
        return copy

    def isTrue(self):# Method to check if the Number instance represents a true value
        return self.value != 0

    def __repr__(self):
        return str(self.value)

class traceBackHandling:  # Class to handle tracebacks in the interpreter
    def __init__(self,displayName,parent = None, parentEntryPos=None ):
        self.displayName = displayName
        self.parent = parent
        self.parentEntryPos = parentEntryPos
        self.symbolTable = None
# Class representing a symbol table for variable storage
class symbolTable:
    def __init__(self):
        self.symbol = {}
        self.parent = None

    def get(self,name):  # Method to retrieve a variable value from the symbol table
        value = self.symbol.get(name,None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self,name,value): # Method to set a variable value in the symbol table
        self.symbol[name]=value

    def remove(self,name):
        del self.symbol[name]

class eval:# Class for the compiler evaluator
    def visit(self,node,context):# Method to visit and evaluate a node in the abstract syntax tree
        functionName = f'visit_{type(node).__name__}'
        function = getattr(self, functionName,self.unknownFunction)
        return function(node,context)

    def unknownFunction(self,node,context):  # Method to handle unknown node types
        raise Exception(f'No visit {type(node).__name__} function determined')

    def visit_numNode(self,node,context):# Method to evaluate a numeric node
        return runtimeResult().success(Number(node.tok.value).setContext(context).setPosition(node.posStart, node.posEnd))

    def visit_varAccessNode(self,node,context):# Method to evaluate a variable access node
        res = runtimeResult()
        varName = node.varNameTok.value
        value = context.symbolTable.get(varName)
        if not value:
            sys.exit(varName + ' is not defined')
        return res.success(value)

    def visit_variableAssignNode(self,node,context):  # Method to evaluate a variable assignment node
        res = runtimeResult()
        varName = node.varNameTok.value
        value = res.register(self.visit(node.valueNode,context))
        if res.error:
            return res
        context.symbolTable.set(varName,value)
        return res.success(value)

    def visit_operationNode(self,node,context): # Method to evaluate an operation node
        res = runtimeResult()
        left = res.register(self.visit(node.left,context))
        if res.error:return res
        right = res.register(self.visit(node.right,context))
        if res.error:return res
        if node.opToken.type == PLUS:
            result, error = left.addition(right)
        elif node.opToken.type == MINUS:
            result, error = left.subtraction(right)
        elif node.opToken.type == MUL:
            result, error = left.multiplication(right)
        elif node.opToken.type == DIV:# Perform the operation based on the operator type
            result, error = left.division(right)
        elif node.opToken.type == EE:
            result, error = left.getComparisonEqual(right)
        elif node.opToken.type == NE:
            result, error = left.getComparisonNotEqual(right)
        elif node.opToken.type == LT:
            result, error = left.getComparisonLessThan(right)
        elif node.opToken.type == GT:
            result, error = left.getComparisonGreaterThan(right)
        elif node.opToken.type == LTE:
            result, error = left.getComparisonLessThanEqual(right)
        elif node.opToken.type == GTE:
            result, error = left.getComparisonGreaterThanEqual(right)
        elif node.opToken.matches(KEYWORD,'and'):
            result, error = left.andBy(right)
        elif node.opToken.matches(KEYWORD,'or'):
            result, error = left.orBy(right)
        if error: # ... (other operators)
            return res.failure(error)
        else:
            return res.success(result.setPosition(node.posStart, node.posEnd))

    def visit_ifNode(self,node,context):  # Method to evaluate an if statement node
        res = runtimeResult()
        for condition,expr in node.case:
            conditionValue = res.register(self.visit(condition,context))
            if res.error:return res
            if conditionValue.isTrue():
                expressionValue = res.register(self.visit(expr,context))
                if res.error:return res
                return res.success(expressionValue)
        if node.elseCase:# Evaluate the else case if no condition is true
            elseValue = res.register(self.visit(node.elseCase,context))
            if res.error:return res
            return res.success(elseValue)
        return res.success(None)

#Executable----------------------------------------------------------------------------------
staticSymbolTable = symbolTable()
staticSymbolTable.set("null",Number(0))
staticSymbolTable.set("true",Number(1))
staticSymbolTable.set("false",Number(0))

def run(text): # Function to run the interpreter on a given input text
    lex = lexer(text)
    tokens, error = lex.makeTokens()
    if error: return None, error
    par = parser(tokens)
    tree = par.parse()
    if tree.error: return None, tree.error
    evaluate = eval()
    context = traceBackHandling('<program>')
    context.symbolTable = staticSymbolTable
    result = evaluate.visit(tree.node, context)
    return result.value, result.error


# Defining the shell function--------------------------------------------------------------------
def main():
    print("Welcome to nexascript,i tries making this compiler ")
    while True:
        text = input('Result > ')
        result, error = run(text)
        if error:
            print(error.log())
        elif result:
            print(result)

if __name__ == "__main__":
    main()