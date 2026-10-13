type fm_settings = {
  timeout: int;
  max_clauses: int }

type abduction_settings = {
  fm: fm_settings;
  abduct_var_diff: fm_settings option }

val fm_default_settings: fm_settings
val default_abduction_settings: abduction_settings

val normalize: Formula.t -> Formula.t
val remove_unknowns_and_symbols: Formula.t -> Formula.t
exception Remaining_symbol_or_unknown
val simplify: Formula.t -> Formula.t
val prettify: ?meta:bool -> Formula.t -> Formula.t

val abduct_conjunctive:
  settings:abduction_settings -> Formula.t list -> Formula.t list list
val abduct:
  settings:abduction_settings -> Formula.t -> Formula.t list list

val surely_valid: ?settings:fm_settings -> Formula.t -> bool
val possibly_sat: ?settings:fm_settings -> Formula.t -> bool

type bound_type = Lower_bound | Upper_bound | Equality
val flip_bound: bound_type -> bound_type
val bound_type: string -> Formula.t -> bound_type option

val suggest_refinements:
  string -> bound_type option -> Formula.t list ->
  (Term.t * Formula.t list) list

val elim_constraint: Formula.t -> (string * Term.t) list list