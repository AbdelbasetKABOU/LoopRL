open Base

type t =
  (* Used for fixed-width encodings *)
  | PADDING
  (* Terms *)
  | VAR
  | VAR_HOLE
  | META_VAR
  | PARAM
  | FUN_APP
  | ADD
  | NEG
  | CMUL
  (* Numerical constants symbols *)
  | NEG_CONST
  | MINUS_THREE
  | MINUS_TWO
  | MINUS_ONE
  | ZERO
  | ONE
  | TWO
  | THREE
  | POS_CONST
  (* Comparison operators *)
  | LE
  | LT
  | GE
  | GT
  | EQ
  | NE
  (* Formulas *)
  | TRUE
  | FALSE
  | UNKNOWN
  | AND
  | OR
  | NOT
  | IMPLIES
  | LABELED
  (* Programs *)
  | PROG
  | ASSIGN
  | IF_ELSE
  | WHILE
  | ASSERT
  | ASSUME
  | INVARIANTS
  | INVARIANT
  | LABELED_PROG
  (* Solver agent *)
  | PROVE_POST  (* Probe titles *)
  | PROVE_INIT
  | PROVE_INDUCTIVE
  | NO_SUBTITLE  (* Probe subtitles *)
  | ADD_DISJUNCT_SUBTITLE
  | STRENGTHEN_ADDED_DISJUNCT_SUBTITLE
  | CONSTRAINTS  (* Env labels *)
  | INV_CANDIDATE  (* Action labels *)
  | ADD_CONSTR
  | PROVE_INV_CANDIDATE
  | ABDUCT_INV_DISJUNCT
  | CONJECTURE_INV_DISJUNCT
  | KEEP_INV_DISJUNCT
  | STRENGHTEN_INV_DISJUNCT
  (* Teacher probes *)
  | PROBE_SELECT_VAR
  | PROBE_REFINE_CONST
  | PROBE_REFINE_FORMULA
  | PROBE_REFINE_PROG
  | NO_PROBE_ARG
  (* Teacher specification *)
  | PROBLEM_SPEC
  | AVAILABLE_CONSTS
  | TWO_VARS_PRESERVED
  | THREE_VARS_PRESERVED
  | SIMPLE_MAIN_INV
  | DISJUNCTIVE_MAIN_INV
  | DISJUNCTIVE_POST
  | BODY_IMPLIES_MAIN_INV
  | USE_AUX_INV
  | USE_CONJUNCTIVE_AUX_INV
  | LOOP_GUARD_USEFUL_FOR_INV
  | LOOP_GUARD_USEFUL_FOR_POST
  | USE_PARAMS
  | USE_COND_GUARD
  | SINGLE_INSTR_CONDITIONAL
  | NO_CONDITIONALS
  | IF_THEN_CONDITIONAL
  | IF_THEN_ELSE_CONDITIONAL
  (* Template restrictions *)
  | ALLOW_VCOMP_IN_PRIM_INV
  | ONLY_CONST_INCR
  | NO_VAR_CONST_ASSIGN
  (* Pretraining *)
  | FIND_CONSEQUENCE
  [@@deriving enum, eq, show, sexp]

type flag =
  | ABDUCTED
  | CLOSING_ACTION
  | TO_PROVE
  | TO_PROVE_LATER
  | PROVED
  | PROVED_CONDITIONALLY
  [@@deriving enum, eq, show, sexp]

type augmented = {
  token: t;
  flags: flag list;
  name: string option;
  cval: int option }
  [@@deriving eq, show, sexp]

let tok ?(flags=[]) ?name ?cval token = {token; flags; name; cval}

(* Edge types are named from the perspective of the destination. *)
type edge =
  | CHILD
  | PARENT
  | NEXT_SIBLING
  | PREV_SIBLING
  | LAST_READ
  | LAST_WRITE
  | GUARDED_BY
  | GUARDED_BY_NEG
  | COMPUTED_FROM
  | NEXT_LEXICAL_USE
  | PREV_LEXICAL_USE
  | SAME_CONST
  | SMALLER_CONST
  | LARGER_CONST
  | PREV_CONST
  | NEXT_CONST
  [@@deriving enum, eq, compare, show, sexp]

let num_edges = max_edge + 1

let pp f t = Fmt.string f (String.drop_prefix ([%show: t] t) 6)
let pp_flag f t = Fmt.string f (String.drop_prefix ([%show: flag] t) 6)
let pp_edge  f t = Fmt.string f (String.drop_prefix ([%show:edge] t) 6)
let pp_augmented f t =
  pp f t.token;
  List.iter t.flags ~f:(fun flag -> Fmt.pf f ".%a" pp_flag flag);
  Option.iter t.name ~f:(fun s -> Fmt.pf f ".%s" s);
  Option.iter t.cval ~f:(fun v -> Fmt.pf f ".%d" v);