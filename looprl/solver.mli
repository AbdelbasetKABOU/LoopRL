open Base
open Search_intf
open Event_intf

module Single_loop_prog: sig
  type t =
    { init : Prog.t
    ; guard : Formula.t
    ; invs : Prog.invariant list
    ; body : Prog.t
    ; post : Prog.t
    ; final_assertion : Formula.t
    ; final_assertion_status : Prog.proof_status option }
  val to_prog : t -> Prog.t
  val of_prog : Prog.t -> t
end

type inv_disjunct_type =
  | Conjectured
  | Abducted
  | Kept
  | Strengthened
  [@@deriving show]

type proof_action =
  | Add_constr of {constr: Formula.t; closing: bool}
  | Prove_inv_candidate of {closing: bool}
  | Add_candidate_disjunct of
      {inv: Formula.t; ty: inv_disjunct_type; closing: bool}
  [@@deriving show]

module Choice_summary: Graphable.S with type t = proof_action

type env = {prog: Single_loop_prog.t; constrs: Formula.t list;}

type probe_title = Prove_post | Prove_init | Prove_inductive [@@deriving show]
and probe_subtitle = Add_disjunct | Strengthen_added_disjunct [@@deriving show]

type probe =
  { title: probe_title
  ; subtitle: probe_subtitle option
  ; env: env
  ; inv_candidate: Formula.t option
  ; obligation: Formula.t option }

module Probe: Graphable.S with type t = probe

module Event: EVENT

module Search: SEARCH_MONAD with
  type probe := Probe.t and
  type summary := Choice_summary.t and
  type event := Event.event and
  type outcome := Event.outcome

val init_solver: Prog.t -> Prog.t Search.tree