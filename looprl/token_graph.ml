open Base

(* ////////////////////////////////////////////////////////////////////////// *)
(* Types                                                                      *)
(* ////////////////////////////////////////////////////////////////////////// *)

(* We represent programs and queries as trees with aditional edges. *)
type 'a tree = Node of 'a * 'a tree list

(* Information flows from the source to the destination.
   Edge types are names from the perspective of the source. *)
type edge = {typ: Token.edge; src: int; dst: int}
  [@@deriving eq, compare]

(* A tree with additional edges. *)
type t = {
  tree: Token.augmented tree;
  edges: edge list }

(* ////////////////////////////////////////////////////////////////////////// *)
(* Utility functions                                                          *)
(* ////////////////////////////////////////////////////////////////////////// *)

let rec num_tokens (Node (_, children)) =
  1 + List.fold children ~init:0 ~f:(fun acc c -> acc + num_tokens c)

let label_tree_with_indexes t =
  let id = ref (-1) in
  let fresh () = Int.incr id; !id in
  let rec aux (Node (x, children)) =
    let x = (fresh (), x) in
    let children = List.map ~f:aux children in
    Node (x, children) in
  aux t

let label_with_tree_rev_pos t =
  let rec aux pos (Node (x, children)) =
    Node ((pos, x), List.mapi children ~f:(
      fun i c -> aux (i::pos) c)) in
  aux [] t

let rec iter_nodes (Node (_, children) as t) ~f =
  f t; List.iter children ~f:(iter_nodes ~f)

let rec map_tree (Node (hd, children)) ~f =
  Node (f hd, List.map children ~f:(map_tree ~f))

let map ~f {tree; edges} = {tree = map_tree ~f tree; edges}

(* ////////////////////////////////////////////////////////////////////////// *)
(* Pretty printing                                                            *)
(* ////////////////////////////////////////////////////////////////////////// *)

let tree_to_string node_to_string ?(width=50) t =
  let rec aux indent (Node (x, children)) =
    (* Return list strings *)
    let children = List.map children ~f:(aux (indent+1)) in
    let w = List.map children ~f:String.length |> List.fold ~init:0 ~f:(+) in
    let sep = if w > width then "\n" ^ (String.make (indent*2) ' ') else " " in
    "(" ^ String.concat ~sep (node_to_string x :: children) ^ ")" in
  aux 1 t

let pp_tree pp_node =
  Fmt.of_to_string (tree_to_string (Fmt.to_to_string pp_node))

let pp_index_token f (idx, tok) =
  Fmt.pf f "%d:%a" idx Token.pp_augmented tok

let pp_edge_src_dst f e = Fmt.pf f "%d<-%d" e.dst e.src

let pp f graph =
  let tree = label_tree_with_indexes graph.tree in
  pp_tree (pp_index_token) f tree;
  Fmt.pf f "@;";
  Fmt.pf f "@[<v>";
  let edge_types =
    graph.edges
    |> List.map ~f:(fun e -> e.typ)
    |> List.dedup_and_sort ~compare:Token.compare_edge in
  List.iter edge_types ~f:(fun typ ->
    Fmt.pf f "@[<hov 2>(%a@;" Token.pp_edge typ;
    List.filter graph.edges ~f:(fun e -> Token.equal_edge e.typ typ)
    |> Fmt.list ~sep:Fmt.sp pp_edge_src_dst f;
    Fmt.pf f "@])@;");
  Fmt.pf f "@]"

(* ////////////////////////////////////////////////////////////////////////// *)
(* Composing graphs                                                           *)
(* ////////////////////////////////////////////////////////////////////////// *)

let edge_shift_token_ids shift e = {e with src=e.src+shift; dst=e.dst+shift}

let shift_token_ids shift g =
  {g with edges = List.map g.edges ~f:(edge_shift_token_ids shift)}

let compose head_token children =
  let shift = ref 1 in
  let children = List.map children ~f:(fun c ->
    let cur = !shift in
    shift := cur + num_tokens c.tree;
    shift_token_ids cur c) in
  let tree = Node (head_token, List.map children ~f:(fun g -> g.tree)) in
  let edges = List.concat_map children ~f:(fun c -> c.edges) in
  {tree; edges}

(* ////////////////////////////////////////////////////////////////////////// *)
(* Adding edges                                                               *)
(* ////////////////////////////////////////////////////////////////////////// *)

let create tree = {tree; edges=[]}

let singleton tok = create (Node (tok, []))

let add_edges g edges = {g with edges = g.edges @ edges}

let add_children_edges g =
  let edges = Queue.create () in
  let rec aux (Node ((id, _), children)) =
    List.iter children ~f:(fun (Node ((child_id, _), _)) ->
      Queue.enqueue edges {typ=Token.CHILD; src=child_id; dst=id});
    List.iter children ~f:aux in
  aux (label_tree_with_indexes g.tree);
  add_edges g (Queue.to_list edges)

let add_next_sibling_edges g =
  let edges = Queue.create () in
  let tree = label_tree_with_indexes g.tree in
  iter_nodes tree ~f:(fun (Node (_, children)) ->
    let prev = ref None in
    List.iter children ~f:(fun (Node ((id, _), _)) ->
      Option.iter !prev ~f:(fun prev ->
        Queue.enqueue edges
          {typ=Token.NEXT_SIBLING; src=id; dst=prev});
      prev := Some id)
  );
  add_edges g (Queue.to_list edges)

let add_reverse_edges ~edge ~reversed g =
  let edges = Queue.create () in
  List.iter g.edges ~f:(fun e ->
    if Token.equal_edge e.typ edge then
      Queue.enqueue edges {typ=reversed; src=e.dst; dst=e.src});
  add_edges g (Queue.to_list edges)

let add_next_lexical_use_edges g =
  let tree = label_tree_with_indexes g.tree in
  let edges = Queue.create () in
  let prev = Hashtbl.create (module String) in
  iter_nodes tree ~f:(fun (Node ((id, x), _)) ->
    Option.iter x.Token.name ~f:(fun name ->
      Option.iter (Hashtbl.find prev name) ~f:(fun prev ->
        Queue.enqueue edges {typ=Token.NEXT_LEXICAL_USE; src=id; dst=prev});
      Hashtbl.set prev ~key:name ~data:id));
  add_edges g (Queue.to_list edges)

let find_numerical_constant ?(min_abs=4) g =
  let tree = label_tree_with_indexes g.tree in
  let consts = ref [] in
  let () = iter_nodes tree ~f:(fun (Node ((i, tok), _)) ->
    Option.iter tok.cval ~f:(fun c ->
      if Int.abs c >= min_abs then consts := (c, i) :: !consts)) in
  !consts

let add_numerical_constants_edges ?min_abs g =
  (* Warning: this generates a quadratic number of edges. *)
  let open Token in
  let consts = find_numerical_constant ?min_abs g in
  let edges = Queue.create () in
  List.iter consts ~f:(fun (c, i) ->
    List.iter consts ~f:(fun (c', i') ->
      if i <> i' then begin
        let typ =
          if c' = c then SAME_CONST
          else if c' = c + 1 then NEXT_CONST
          else if c' = c - 1 then PREV_CONST
          else if c' < c then SMALLER_CONST
          else if c' > c then LARGER_CONST
          else assert false in
        Queue.enqueue edges {typ; src=i'; dst=i}
      end));
  add_edges g (Queue.to_list edges)

let add_all_reverse_edges g = g
  |> add_reverse_edges ~edge:NEXT_LEXICAL_USE ~reversed:PREV_LEXICAL_USE
  |> add_reverse_edges ~edge:CHILD ~reversed:PARENT
  |> add_reverse_edges ~edge:NEXT_SIBLING ~reversed:PREV_SIBLING

let add_canonical_edges g = g
  |> add_next_lexical_use_edges
  |> add_children_edges
  |> add_next_sibling_edges
