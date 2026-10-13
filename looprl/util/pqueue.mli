type cost = int

type 'a t

val empty: 'a t

val is_empty: 'a t -> bool

val singleton: cost -> 'a -> 'a t

val insert: 'a t -> cost -> 'a -> 'a t

val extract: 'a t -> cost * 'a * 'a t

val size: 'a t -> int

exception Queue_is_empty