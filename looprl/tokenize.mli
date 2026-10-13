val program : Prog.t -> Token_graph.t
(** Tokenize a program and add the semantic edges. *)

val instr : Prog.instr -> Token_graph.t
(** Tokenize an instruction. *)

val formula : Formula.t -> Token_graph.t
(** Tokenize a formula. *)

val term : Term.t -> Token_graph.t
(** Tokenize a term. *)

val var : string -> Token_graph.t
(** Tokenize a variable. *)

val example_prog: Prog.t
val simple_example_prog: Prog.t