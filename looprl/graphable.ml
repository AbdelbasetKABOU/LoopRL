open Base
open Python_lib

module type S = sig
  type t
  val to_string: t -> string
  val to_meta: t -> (string, string) List.Assoc.t
  val to_graph: t -> Token_graph.t
  val sexp_of_t: t -> Sexp.t
  val t_of_sexp: Sexp.t -> t
end

type t = Pack: 'a * (module S with type t = 'a) -> t

type tokenizer_config = {
  enable_numerical_edges: bool;
  add_reverse_edges: bool
} [@@deriving python]

let apply_config config graph =
  let graph =
    if config.enable_numerical_edges
    then Token_graph.add_numerical_constants_edges graph
    else graph in
  let graph =
    if config.add_reverse_edges
    then Token_graph.add_all_reverse_edges graph
    else graph in
  graph

let tensorize
  (type a) (module G: S with type t = a)
  ~config ~tokenizer_config uids (x: a) =
  G.to_graph x
  |> Token_graph.add_canonical_edges
  |> apply_config tokenizer_config
  |> Tensorize.tensorize ~config uids