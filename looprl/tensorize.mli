type config = {
  d_model : int;
  pos_enc_size : int;
  uid_emb_size : int;
  const_emb_size : int;
} [@@deriving python]

val token_encoding_size: config -> int
val uid_encoding_offset: config -> int

type tensor2f
type tensor2i
val python_of_tensor2f : tensor2f -> Pytypes.pyobject
val python_of_tensor2i : tensor2i -> Pytypes.pyobject

module Uid_map :
  sig
    type t
    val empty : t
    val get : t -> string -> int * t
    val rget : t ref -> string -> int
    val vars : t -> string list
  end

module GraphTensors :
  sig
    type t = { nodes : tensor2f; edges : tensor2i; pos_emb : tensor2f; }
    val python_of_t : t -> Pytypes.pyobject
    val num_tokens : t -> int
    val create :
      num_toks:int -> num_edges:int -> num_features:int -> d_model:int -> t
  end

val tensorize :
  config:config -> Uid_map.t -> Token_graph.t -> GraphTensors.t * Uid_map.t

val pp_bin_tensor :
  ?row_nums:bool ->
  ?seps:int list ->
  Format.formatter -> tensor2f -> unit

val pp_graph_tensors : config -> GraphTensors.t Fmt.t