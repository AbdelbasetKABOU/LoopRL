type 'a tree = Node of 'a * 'a tree list
type edge = { typ : Token.edge; src : int; dst : int; } [@@deriving eq, compare]
type t = { tree : Token.augmented tree; edges : edge list; }

val num_tokens : 'a tree -> int
val label_tree_with_indexes : 'a tree -> (int * 'a) tree
val label_with_tree_rev_pos : 'a tree -> (int list * 'a) tree

val iter_nodes : 'a tree -> f:('a tree -> unit) -> unit
val tree_to_string : ('a -> string) -> ?width:int -> 'a tree -> string

val map : f:(Token.augmented -> Token.augmented) -> t -> t

val pp_tree : 'a Fmt.t -> 'a tree Fmt.t
val pp : t Fmt.t

val compose : Token.augmented -> t list -> t
val create : Token.augmented tree -> t
val singleton : Token.augmented -> t
val add_edges : t -> edge list -> t
val add_children_edges : t -> t
val add_next_sibling_edges : t -> t
val add_reverse_edges : edge:Token.edge -> reversed:Token.edge -> t -> t
val add_next_lexical_use_edges : t -> t
val add_numerical_constants_edges : ?min_abs:int -> t -> t
val add_all_reverse_edges : t -> t

val add_canonical_edges : t -> t
(** Add the following edges: children, sibling and last lexical *)