open Base

val sub_formula_with_label : string -> Formula.t -> Formula.t option
val remove_label : string -> Formula.t -> Formula.t
val is_redundant : ?assuming:Formula.t -> Formula.t list -> bool
val remove_redundant : ?assuming:Formula.t -> Formula.t list -> Formula.t list
val remove_redundant_conjuncts : Formula.t -> Formula.t
val possible_consequences : Formula.t -> Formula.t list

val add_conjunct : conj:Formula.t -> Formula.t -> Formula.t
val add_disjunct : disj:Formula.t -> Formula.t -> Formula.t

val randomize_comparisons : Base.Random.State.t -> Formula.t -> Formula.t

val simplify_to_atomic: ?assuming:Formula.t -> Formula.t -> Formula.t option

val exists_in_formula: f:(Formula.t -> bool) -> Formula.t  -> bool

val vars_appearing_with_unit_coeff: Formula.t -> Set.M(String).t