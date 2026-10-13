open Base

type proof_status =
  To_prove | To_prove_later | Proved | Proved_conditionally
  [@@deriving eq, compare, hash, sexp]

type t = Prog of instr list
  [@@deriving eq, compare, hash, sexp]
and instr =
  | LabeledProg of string * t option
  | Assume of Formula.t
  | Assert of Formula.t * proof_status option
  | Assign of string * Term.t
  | If of Formula.t * t * t
  | While of Formula.t * invariant list * t
  [@@deriving eq, compare, hash, sexp]
and invariant = Formula.t * proof_status option
  [@@deriving eq, compare, hash, sexp]

val is_empty_prog : t -> bool
val prog_instrs : t -> instr list

val vars_set : t -> Set.M(String).t
val pred_symbols : t -> Set.M(String).t
val prog_symbols : t -> Set.M(String).t
val modified_vars: t -> Set.M(String).t

val iter_instrs: f:(instr -> unit) -> t -> unit
val map_instrs: f:(instr -> t) -> t -> t
val apply_recursively: f:(instr -> instr) -> t -> t
val map_formulas: f:(Formula.t -> Formula.t) -> t -> t

val subst: from:string -> substituted:Term.t -> t -> t
val subst_multi: f:(string -> Term.t) -> t -> t
val rename_var : from:string -> renamed:string -> t -> t
val subst_prog_symbol : from:string -> substituted:t -> t -> t
val subst_pred_symbol : from:string -> substituted:Formula.t -> t -> t

val pp : t Fmt.t
val pp_instr : instr Fmt.t

exception Remaining_prog_symbol
val wlp : t -> Formula.t -> Formula.t
