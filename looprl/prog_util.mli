open Base

val first_loop : (Prog.t, Prog.instr) Lens.t
val invariants : (Prog.instr, Prog.invariant list) Lens.t
val initial_param_assums : (Prog.t, Formula.t list) Lens.t

val append_instr : string -> Prog.instr -> Prog.t -> Prog.t
val replace_final_instr : final:Prog.t -> Prog.t -> Prog.t

val formula_with_label : string -> (Prog.t, Formula.t) Lens.t
val prog_with_label : string -> (Prog.t, Prog.t) Lens.t

val remove_prog_label : string -> Prog.t -> Prog.t
val remove_formula_label : string -> Prog.t -> Prog.t
val remove_all_labels : ?except:(string list) -> Prog.t -> Prog.t

val parameter_vars : Prog.t -> Set.M(String).t

val normalize_task : Prog.t -> Prog.t

val exists_in_prog: f:(Prog.instr -> bool) -> Prog.t -> bool