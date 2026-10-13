open Base

module Compop :
  sig
    type t = EQ | NE | LE | LT | GE | GT
      [@@deriving eq, compare, hash, sexp]
    val not : t -> t
    val reverse : t -> t
    val interpret : t -> int -> int -> bool
    val to_string : t -> string
  end

type t =
  | Unknown
  | Labeled of string * t option
  | Bconst of bool
  | And of t list
  | Or of t list
  | Not of t
  | Implies of t * t
  | Comp of Term.t * Compop.t * Term.t
  [@@deriving eq, compare, hash, sexp]

val comp_eval : Term.t -> Compop.t -> Term.t -> bool option
val is_true: t -> bool
val is_false: t -> bool

val mk_conj: t list -> t
val mk_disj: t list -> t
val mk_implies: t list -> t -> t
val conjuncts: t -> t list
val disjuncts: t -> t list

val vars_set : t -> Set.M(String).t
val pred_symbols : t -> Set.M(String).t

val map_children : f:(t -> t) -> t -> t
val apply_recursively: f:(t -> t) -> t -> t
val iter_sub: f:(t -> unit) -> t -> unit
val subst: from:string -> substituted:Term.t -> t -> t
val subst_multi: f:(string -> Term.t) -> t -> t
val subst_pred_symbol: from:string -> substituted:t -> t -> t

val pp: t Fmt.t

module Infix :
  sig
    val ( == ) : Term.t -> Term.t -> t
    val ( <> ) : Term.t -> Term.t -> t
    val ( <= ) : Term.t -> Term.t -> t
    val ( < )  : Term.t -> Term.t -> t
    val ( >= ) : Term.t -> Term.t -> t
    val ( > )  : Term.t -> Term.t -> t
  end