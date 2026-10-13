open Base
open Python_lib
open Bigarray

(* ////////////////////////////////////////////////////////////////////////// *)
(* Configuration                                                              *)
(* ////////////////////////////////////////////////////////////////////////// *)

type config = {
  d_model: int;
  pos_enc_size: int;
  uid_emb_size: int;
  const_emb_size: int}
  [@@deriving python]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Tensor utilities                                                           *)
(* ////////////////////////////////////////////////////////////////////////// *)

type tensor_int = int64
let tensor_int = Int64.of_int_exn
let tensor_int_elt = Int64

type tensor2f = (float, float32_elt, c_layout) Array2.t
type tensor2i = (tensor_int, int64_elt, c_layout) Array2.t
let python_of_tensor2f t = Numpy.of_bigarray (genarray_of_array2 t)
let python_of_tensor2i t = Numpy.of_bigarray (genarray_of_array2 t)

(* We use Owl because [Array2.create] does not initialize the array and
   [Array2.init] is extremely slow.  *)
let create_tensor2f dx dy =
  Owl.Dense.Ndarray.Generic.zeros Float32 [|dx; dy|]
  |> array2_of_genarray

let create_tensor2i dx dy = Array2.create tensor_int_elt C_layout dx dy

let set = Array2.unsafe_set

(* ////////////////////////////////////////////////////////////////////////// *)
(* Layout                                                                     *)
(* ////////////////////////////////////////////////////////////////////////// *)

type features_layout = {
  tok_start: int;
  flags_start: int;
  uid_emb_start: int;
  const_emb_start: int;
  num_features: int }

let num_tokens = Token.max + 1
let num_flags = Token.max_flag + 1

let features_layout conf =
  let tok_start = 0 in
  let flags_start = tok_start + num_tokens in
  let uid_emb_start = flags_start + num_flags in
  let const_emb_start = uid_emb_start + conf.uid_emb_size in
  let num_features = const_emb_start + conf.const_emb_size in
  {tok_start; flags_start; uid_emb_start; const_emb_start; num_features}

let token_encoding_size conf = (features_layout conf).num_features
let uid_encoding_offset conf = (features_layout conf).uid_emb_start

(* ////////////////////////////////////////////////////////////////////////// *)
(* Unique IDs Utilities                                                       *)
(* ////////////////////////////////////////////////////////////////////////// *)

module Uid_map = struct

  type t = {
    next: int;
    map: int Map.M(String).t }

  let empty = {next = 0; map = Map.empty (module String)}

  let get t name =
    match Map.find t.map name with
    | Some uid -> (uid, t)
    | None ->
      let uid = t.next in
      let t = {
        map = Map.add_exn t.map ~key:name ~data:uid;
        next = uid + 1 } in
      (uid, t)

  let rget rt name =
    let (uid, t) = get !rt name in
    rt := t;
    uid

  let vars m = Map.keys (m.map)

  let%expect_test "uid_map" =
    let r = ref empty in
    ["x"; "y"; "x"; "x"; "z"; "y"]
    |> List.map ~f:(fun s -> rget r s)
    |> [%show: int list]
    |> Stdio.print_string;
    [%expect {| [0; 1; 0; 0; 2; 1] |}]

end

(* ////////////////////////////////////////////////////////////////////////// *)
(* Defining graph tensors                                                     *)
(* ////////////////////////////////////////////////////////////////////////// *)

module GraphTensors = struct

  (** A tensor representing a network input:
      - [nodes] has shape (num_toks, num_features)
      - [edges] has shape (num_edges, 3) and consists (src, dst, typ) triples
      - [pos_emb] has shape (num_toks, d_model) *)
  type t = {
    nodes: tensor2f;
    edges: tensor2i;
    pos_emb: tensor2f; }
    [@@deriving python_of]

  let num_tokens ts = Array2.dim1 ts.nodes

  let create ~num_toks ~num_edges ~num_features ~d_model = {
    nodes = create_tensor2f num_toks num_features;
    edges = create_tensor2i num_edges 3;
    pos_emb = create_tensor2f num_toks d_model }

  let src_idx, dst_idx, typ_idx = 0, 1, 2

end

open GraphTensors

(* ////////////////////////////////////////////////////////////////////////// *)
(* Embedding utilities                                                        *)
(* ////////////////////////////////////////////////////////////////////////// *)

(* Allocation-free iterators *)

let list_of_iter it =
  let l = ref [] in
  it (fun v -> l := v::!l);
  List.rev !l

let write_iter ~maxlen it arr i j =
  let exception Break in
  let k = ref 0 in
  try
    it (fun v ->
      if !k >= maxlen then raise Break
      else Array2.unsafe_set arr i (j + !k) v;
      Int.incr k)
  with Break -> ()

(* Binary encoding *)

let binary_encoding n f =
  assert (n >= 0);
  let rec aux = function
    | 0 -> ()
    | n -> f (Float.of_int (n%2)) ; aux (n/2) in
  aux n

let%expect_test "binary_encoding" =
  let ex n =
    let res = list_of_iter (binary_encoding n) |> List.map ~f:Int.of_float in
    Fmt.pr "%d -> %s\n" n ([%show: int list] res) in
  ex 0; ex 1; ex 2; ex 3; ex 4; ex 5;
  ex 10; ex 32; ex 100;
  [%expect {|
    0 -> []
    1 -> [1]
    2 -> [0; 1]
    3 -> [1; 1]
    4 -> [0; 0; 1]
    5 -> [1; 0; 1]
    10 -> [0; 1; 0; 1]
    32 -> [0; 0; 0; 0; 0; 1]
    100 -> [0; 0; 1; 0; 0; 1; 1] |}]

let encode_tree_pos rpos f =
  let rec encode = function
    | [] -> ()
    | i::rpos -> f 1.; f 0.; sibling rpos i
  and sibling rpos = function
    | 0 -> encode rpos
    | i -> f 0.; f 1.; sibling rpos (i-1) in
  encode rpos

let%expect_test "encode_tree_pos" =
  let ex rpos =
    let res = encode_tree_pos rpos
      |> list_of_iter |> List.map ~f:Int.of_float in
    Fmt.pr "%s -> %s\n" ([%show: int list] rpos) ([%show: int list] res) in
  ex []; ex [0]; ex [1]; ex [2];
  ex [0; 0]; ex [0; 1]; ex [1; 0]; ex [2; 0; 1];
  [%expect {|
    [] -> []
    [0] -> [1; 0]
    [1] -> [1; 0; 0; 1]
    [2] -> [1; 0; 0; 1; 0; 1]
    [0; 0] -> [1; 0; 1; 0]
    [0; 1] -> [1; 0; 1; 0; 0; 1]
    [1; 0] -> [1; 0; 0; 1; 1; 0]
    [2; 0; 1] -> [1; 0; 0; 1; 0; 1; 1; 0; 1; 0; 0; 1] |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Tensorizing function                                                       *)
(* ////////////////////////////////////////////////////////////////////////// *)

let tensorize ~config uids graph =
  let open Token_graph in
  let uids = ref uids in
  let tree = graph.tree |> label_with_tree_rev_pos |> label_tree_with_indexes in
  let num_toks = num_tokens graph.tree in
  let layout = features_layout config in
  let num_features = layout.num_features in
  let num_edges = List.length graph.edges in
  let d_model = config.d_model in
  let ts = GraphTensors.create ~num_toks ~num_edges ~num_features ~d_model in
  (* Nodes encoding *)
  iter_nodes tree ~f:(fun (Node ((tid, (tpos, tok)), _)) ->
    (* Encode token type *)
    let tok_id = Token.to_enum tok.token in
    set ts.nodes tid (layout.tok_start + tok_id) 1.0;
    (* Encode flags *)
    List.iter tok.flags ~f:(fun flag ->
      let flag_id = Token.flag_to_enum flag in
      set ts.nodes tid (layout.flags_start + flag_id) 1.0);
    (* Encode constant value *)
    Option.iter tok.cval ~f:(fun cval ->
      let const_emb = binary_encoding (Int.abs cval) in
      write_iter ~maxlen:config.const_emb_size
        const_emb ts.nodes tid layout.const_emb_start);
    (* Encode uid *)
    Option.iter tok.name ~f:(fun name ->
      let uid = Int.min (Uid_map.rget uids name) (config.uid_emb_size-1) in
      set ts.nodes tid (layout.uid_emb_start + uid) 1.0);
    (* Position encoding *)
    let pos_emb = encode_tree_pos tpos in
    write_iter ~maxlen:config.pos_enc_size pos_emb ts.pos_emb tid 0
  );
  (* Edges encoding *)
  List.iteri graph.edges ~f:(fun i {typ; src; dst} ->
    set ts.edges i src_idx @@ tensor_int src;
    set ts.edges i dst_idx @@ tensor_int dst;
    set ts.edges i typ_idx @@ tensor_int (Token.edge_to_enum typ)
  );
  (ts, !uids)

(* ////////////////////////////////////////////////////////////////////////// *)
(* Tests                                                                      *)
(* ////////////////////////////////////////////////////////////////////////// *)

let pp_bin_tensor ?(row_nums=true) ?(seps=[]) f t =
  for i = 0 to Array2.dim1 t - 1 do
    if row_nums then Fmt.pf f "%3s|  " (Int.to_string i);
    for j = 0 to Array2.dim2 t - 1 do
      if List.mem ~equal:Int.equal seps j then Fmt.pf f "| ";
      let v = Array2.get t i j in
      Fmt.pf f "%s " (if Float.(equal v zero) then "." else "1")
    done;
    Fmt.pf f "@;"
  done

let pp_graph_tensors conf f ts =
  let layout = features_layout conf in
  let seps = [
    layout.flags_start; layout.uid_emb_start; layout.const_emb_start] in
  Fmt.pf f "NODE FEATURES:@;";
  pp_bin_tensor f ~seps ts.nodes;
  Fmt.pf f "@;";
  Fmt.pf f "TREE POS ENCODING:@;";
  pp_bin_tensor f ts.pos_emb

let%expect_test "graph_tensors" =
  let prog = Tokenize.simple_example_prog in
  let graph = Tokenize.program prog in
  let config = {
    d_model=32; uid_emb_size=4; const_emb_size=4; pos_enc_size=32} in
  let uids = Uid_map.empty in
  let ts, _ = tensorize ~config uids graph in
  Fmt.pr "@[<v>%a@]" (pp_graph_tensors config) ts;
  [%expect {|
    NODE FEATURES:
      0|  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . . . . | . . . .
      1|  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . . . . | . . . .
      2|  . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | 1 . . . | . . . .
      3|  . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . . . . | . . . .
      4|  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . . . . | . . . .
      5|  . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . . . . | . . . .
      6|  . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | 1 . . . | . . . .
      7|  . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . 1 . . | . . . .
      8|  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . . . . | . . . .
      9|  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . 1 . . . | . . . . | . . . .
     10|  . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . . . . | . . . .
     11|  . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | 1 . . . | . . . .
     12|  . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . 1 . . | . . . .
     13|  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . . . . | . . . .
     14|  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . . . . | . . . .
     15|  . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | 1 . . . | . . . .
     16|  . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . . . . | . . . .
     17|  . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | 1 . . . | . . . .
     18|  . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . . . . | 1 1 . .
     19|  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . 1 | . . . . | . . . .
     20|  . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . . . . | . . . .
     21|  . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | 1 . . . | . . . .
     22|  . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . . . . . . | . 1 . . | . . . .

    TREE POS ENCODING:
      0|  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
      1|  1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
      2|  1 . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
      3|  1 . . 1 1 . . . . . . . . . . . . . . . . . . . . . . . . . . .
      4|  1 . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . .
      5|  1 . 1 . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . .
      6|  1 . 1 . 1 . . 1 . . . . . . . . . . . . . . . . . . . . . . . .
      7|  1 . . 1 1 . 1 . . 1 . . . . . . . . . . . . . . . . . . . . . .
      8|  1 . . 1 1 . . 1 . . . . . . . . . . . . . . . . . . . . . . . .
      9|  1 . 1 . . 1 1 . . 1 . . . . . . . . . . . . . . . . . . . . . .
     10|  1 . 1 . 1 . . 1 1 . . 1 . . . . . . . . . . . . . . . . . . . .
     11|  1 . 1 . 1 . 1 . . 1 1 . . 1 . . . . . . . . . . . . . . . . . .
     12|  1 . . 1 1 . 1 . 1 . . 1 1 . . 1 . . . . . . . . . . . . . . . .
     13|  1 . . 1 . 1 1 . . 1 . . . . . . . . . . . . . . . . . . . . . .
     14|  1 . 1 . . 1 . 1 1 . . 1 . . . . . . . . . . . . . . . . . . . .
     15|  1 . 1 . 1 . . 1 . 1 1 . . 1 . . . . . . . . . . . . . . . . . .
     16|  1 . . 1 1 . 1 . . 1 . 1 1 . . 1 . . . . . . . . . . . . . . . .
     17|  1 . 1 . . 1 1 . 1 . . 1 . 1 1 . . 1 . . . . . . . . . . . . . .
     18|  1 . . 1 1 . . 1 1 . 1 . . 1 . 1 1 . . 1 . . . . . . . . . . . .
     19|  1 . . 1 . 1 . . . . . . . . . . . . . . . . . . . . . . . . . .
     20|  1 . 1 . . 1 . 1 . . . . . . . . . . . . . . . . . . . . . . . .
     21|  1 . 1 . 1 . . 1 . 1 . . . . . . . . . . . . . . . . . . . . . .
     22|  1 . . 1 1 . 1 . . 1 . 1 . . . . . . . . . . . . . . . . . . . . |}]