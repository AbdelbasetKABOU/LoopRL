(** Linear term representation for Looprl.
    Invariant: the [Term.t] type can only represent simplified terms
    with nonzero coeffs and an optional constant term at the end. *)

open Base

type atom =
  | Var of string
  | FunApp of string * t list
  | One
  [@@deriving eq, compare, hash, sexp]

and t [@@deriving eq, compare, hash, sexp]

val to_alist: t -> (atom * int) list
val coeff: atom -> t -> int option

val add: t -> t -> t
val zero: t
val one: t
val const: int -> t
val var: string -> t
val funapp: string -> t list -> t
val sum: t list -> t
val neg: t -> t
val mulc: int -> t -> t
val coeffs_gcd: t -> int
val get_const: t -> int option
val divc: t -> int -> t
val sub: t -> t -> t
val atom: atom -> t

val subst: from:string -> substituted:t -> t -> t
val subst_multi: f:(string -> t) -> t -> t

val eval: ?valuation:(string -> int option) -> t -> int option
val vars_set: t -> Set.M(String).t
val partition: f:(atom -> bool) -> t -> t * t

module type DOMAIN = sig
  type t
  val zero: t
  val one: t
  val add: t -> t -> t
  val cmul: int -> t -> t
end

val eval_domain:
  (module DOMAIN with type t = 'num) ->
  ?valuation:(string -> 'num option) -> t -> 'num option

val to_string: t -> string
val pp: t Fmt.t

module Infix: sig
  val ( + ): t -> t -> t
  val ( - ): t -> t -> t
  val ( * ): int -> t -> t
  val (~- ): t -> t
  val ( / ): t -> int -> t
end