val sample_task: ?random_sigils:bool -> true_false:bool ->
  Util.Random.rng -> Formula.t * Formula.t * Formula.t

module Probe: Graphable.S with type t = Formula.t
module Choice_summary: Graphable.S with type t = Formula.t