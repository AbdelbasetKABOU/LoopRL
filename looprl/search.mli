open Search_intf

module Make
  (Probe: sig type t end) (Summary: sig type t end)
  (Event: sig type event type outcome val default_failure: outcome end):
  SEARCH_MONAD with
    type probe := Probe.t and type summary := Summary.t and
    type event := Event.event and type outcome := Event.outcome