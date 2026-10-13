(* x = {x+1} - 1 *)

%{
  open Formula
  open Prog

  let proof_status_of_string = function
    | "to-prove" -> To_prove
    | "prove-later" -> To_prove_later
    | "proved..." -> Proved_conditionally
    | "proved" -> Proved
    | s -> failwith ("Unknown annotation: " ^ s)

  let or_empty_prog = function
    | None -> Prog []
    | Some p -> p

%}

%token EOF
%token <string> IDENTIFIER
%token <int> INT_CONST
%token <string> STRING_LIT

%token LEFT_PAREN RIGHT_PAREN LEFT_CURLY RIGHT_CURLY
%token COMMA COLON SEMICOLON

%token PLUS MINUS TIMES
%token EQ NE LE LT GE GT
%token TRUE FALSE UNKNOWN AND OR IMPLIES NOT

%token ASSUME ASSERT IF ELSE WHILE ASSIGN INVARIANT

%start <Formula.t> single_formula
%start <Prog.t> single_program
%start <Term.t> single_term

%right IMPLIES
%left OR
%left AND
%nonassoc NOT
%left PLUS MINUS
%left TIMES
%nonassoc UNARY_MINUS

%%

single_term: t=term EOF { t }

single_formula: f=formula EOF { f }

single_program: p=program EOF { p }

program: is=list(instr) { Prog is }

proof_status: st=option(STRING_LIT) { Option.map proof_status_of_string st }

instr:
  | ASSUME cond=formula SEMICOLON { Assume cond }
  | ASSERT cond=formula SEMICOLON st=proof_status { Assert (cond, st) }
  | s=IDENTIFIER SEMICOLON { LabeledProg (s, None) }
  | s=IDENTIFIER COLON LEFT_CURLY p=program RIGHT_CURLY
    { LabeledProg (s, Some p) }
  | s=IDENTIFIER COLON i=instr { LabeledProg (s, Some (Prog [i])) }
  | x=IDENTIFIER ASSIGN e=term SEMICOLON { Assign (x, e) }
  | IF LEFT_PAREN cond=formula RIGHT_PAREN
    LEFT_CURLY tb=program RIGHT_CURLY
    fb=option(else_clause)
    { If (cond, tb, or_empty_prog fb) }
  | WHILE LEFT_PAREN cond=formula RIGHT_PAREN
    LEFT_CURLY invs=list(invariant) body=program RIGHT_CURLY
    { While (cond, invs, body) }

else_clause: ELSE LEFT_CURLY body=program RIGHT_CURLY { body }

invariant: INVARIANT inv=formula SEMICOLON st=proof_status { (inv, st) }

%inline comp_op:
  | EQ {Compop.EQ} | NE {Compop.NE}
  | LT {Compop.LT} | LE {Compop.LE} | GT {Compop.GT} | GE {Compop.GE}

formula:
  | UNKNOWN { Unknown }
  | s=IDENTIFIER { Labeled (s, None) }
  | TRUE { Bconst true }
  | FALSE { Bconst false }
  | NOT f=formula { Not f }
  | lhs=term op=comp_op rhs=term { Comp (lhs, op, rhs) }
  | lhs=formula AND rhs=formula { And [lhs; rhs] }
  | lhs=formula OR rhs=formula { Or [lhs; rhs] }
  | lhs=formula IMPLIES rhs=formula { Implies (lhs, rhs) }
  | LEFT_PAREN f=formula RIGHT_PAREN { f }
  | LEFT_PAREN l=IDENTIFIER COLON f=formula RIGHT_PAREN { Labeled (l, Some f) }

term:
  | i=INT_CONST { Term.const i }
  | x=IDENTIFIER { Term.var x }
  | lhs=term PLUS rhs=term { Term.add lhs rhs }
  | lhs=term MINUS rhs=term { Term.sub lhs rhs }
  | c=INT_CONST TIMES t=term { Term.mulc c t }
  | MINUS t=term %prec UNARY_MINUS { Term.mulc (-1) t }
  | f=IDENTIFIER LEFT_PAREN args=separated_list(COMMA, term) RIGHT_PAREN
    { Term.funapp f args }
  (* We cannot use normal parens as otherwise (x) could be
     interpreted as either a term or a formula. *)
  | LEFT_CURLY t=term RIGHT_CURLY { t }