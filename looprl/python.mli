open Base
open Python_lib
open Event_intf

module Make (): sig

  val rng_class: Random.State.t Class_wrapper.t
  val rng_param: Random.State.t Defunc.Of_python.t

  val prog_class: Prog.t Class_wrapper.t
  val prog_param: Prog.t Defunc.Of_python.t
  val python_of_prog: Prog.t -> pyobject

  type teacher_result = Teacher.result [@@deriving python, sexp]

  val uid_map_class: Tensorize.Uid_map.t Class_wrapper.t
  val tensorizer_config_param: Tensorize.config Defunc.Of_python.t

  type graphable = Graphable.t
  val graphable_class: graphable Class_wrapper.t
  val python_of_graphable: graphable -> pyobject
  val python_unserialize_graphable:
    (module Graphable.S with type t = 'a) -> (unit -> pyobject) Defunc.t

  module type WITH_SEARCH = sig
    module Probe: Graphable.S
    module Choice_summary: Graphable.S
    module Event: EVENT
    module Search: Search_intf.SEARCH_MONAD with
      type summary := Choice_summary.t and
      type probe := Probe.t and
      type event := Event.event and
      type outcome := Event.outcome
  end

  module Search_wrapper (M: WITH_SEARCH): sig
    val make_class:
      string ->
      ('a -> pyobject) -> ('a -> Sexp.t) -> (Sexp.t -> 'a) ->
      'a M.Search.tree Class_wrapper.t * (unit -> pyobject) Defunc.t
    val agent_spec: pyobject
  end

end
