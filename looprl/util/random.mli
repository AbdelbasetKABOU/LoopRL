type rng = Base.Random.State.t

val shuffle_array : rng -> 'a array -> unit
val shuffle : rng -> 'a list -> 'a list
val sample_subset: rng -> 'a list -> 'a list
val pick: rng -> 'a list -> 'a

type 'a distr
val make_distr : ('a * float) list -> 'a distr
val sample : 'a distr -> rng -> 'a
val bernouilli : ?p:float -> rng -> bool

val filtered : ('a -> bool) -> (rng -> 'a) -> (rng -> 'a)