open Base

module type EVENT =
sig
  type outcome [@@deriving enum, show, sexp]
  type event [@@deriving enum, show, sexp]

  val success: outcome
  val default_failure: outcome
  val size_limit_exceeded: outcome

  val outcome_reward: outcome -> float
  val event_reward: event -> float
  val max_event_occurences: event -> int

  val min_success_reward: float
end