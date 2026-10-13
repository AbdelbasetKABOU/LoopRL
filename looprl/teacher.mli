open Search_intf
open Event_intf

module Probe: Graphable.S

module Choice_summary: Graphable.S

module Event: EVENT

module Search: SEARCH_MONAD with
  type probe := Probe.t and
  type summary := Choice_summary.t and
  type event := Event.event and
  type outcome := Event.outcome

type result = {
  problem: Prog.t;
  nonprocessed: Prog.t }

val init_teacher: Util.Random.rng -> result Search.tree

val init_teacher_with_spec: Util.Random.rng -> string -> result Search.tree