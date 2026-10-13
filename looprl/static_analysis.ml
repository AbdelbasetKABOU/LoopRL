open Base
open Token
open Token_graph

(* ////////////////////////////////////////////////////////////////////////// *)
(* Analysis state                                                             *)
(* ////////////////////////////////////////////////////////////////////////// *)

module State = struct

  type t = {
    last_read: (int list) Hashtbl.M(String).t;
    last_written: (int list) Hashtbl.M(String).t;
    guards: (int list) Hashtbl.M(String).t;
    neg_guards: (int list) Hashtbl.M(String).t }

  let create () =
    let size = 10 in {
    last_read = Hashtbl.create ~size (module String);
    last_written = Hashtbl.create ~size (module String);
    guards = Hashtbl.create ~size (module String);
    neg_guards = Hashtbl.create ~size (module String)}

  let copy s = {
    last_read = Hashtbl.copy s.last_read;
    last_written = Hashtbl.copy s.last_written;
    guards = Hashtbl.copy s.guards;
    neg_guards = Hashtbl.copy s.neg_guards }

  let append_hashtbl h h' =
    Hashtbl.iter_keys h' ~f:(fun key ->
      Hashtbl.find_multi h' key |> List.iter ~f:(fun data ->
        Hashtbl.add_multi h ~key ~data))

  let join_with st st' =
    append_hashtbl st.last_read st'.last_read;
    append_hashtbl st.last_written st'.last_written;
    append_hashtbl st.guards st'.guards;
    append_hashtbl st.neg_guards st'.neg_guards

end

(* ////////////////////////////////////////////////////////////////////////// *)
(* Utilities                                                                  *)
(* ////////////////////////////////////////////////////////////////////////// *)

open State

let tree_tid (Node ((id, _), _)) = id
let tree_token (Node ((_, tok), _)) = tok
let tree_token_type tree = (tree_token tree).token
let tree_children (Node ((_, _), children)) = children
let tree_child tree i = List.nth_exn (tree_children tree) i
let assert_token tree tok = assert (Token.equal (tree_token tree).token tok)

let make_edges ~emit ~state (var, v_tid) =
  (* Make all the edges ending up to [v_tid] using the current state *)
  let aux typ st =
    Hashtbl.find_multi st var |> List.iter ~f:(fun i_tid ->
      emit {typ; src=i_tid; dst=v_tid}) in
  let open State in
  aux LAST_READ state.last_read;
  aux LAST_WRITE state.last_written;
  aux GUARDED_BY state.guards;
  aux GUARDED_BY_NEG state.neg_guards

let var_symbol (Node ((id, x), _)) =
  match x.token with
  | VAR | META_VAR | VAR_HOLE | PARAM -> Some (Option.value_exn x.name, id)
  | _ -> None

let rec iter_tree_vars ~f (Node (_, children) as t) =
  (* Call [f] on all the [(v_name, v_tid)] pairs *)
  Option.iter ~f (var_symbol t);
  List.iter children ~f:(iter_tree_vars ~f)

let make_all_edges ~emit ~state =
  (* Create edges for all variables in a formula *)
  iter_tree_vars ~f:(make_edges ~emit ~state)

let read_all ~state instr_id tree =
  (* Record that instruction [instr_id] reads all variables in [tree] *)
  iter_tree_vars tree ~f:(fun (var_name, _) ->
    Hashtbl.add_multi state.last_read ~key:var_name ~data:instr_id)

let assume_bval b_guards notb_guards fml_tree =
  let fml_id = tree_tid fml_tree in
  iter_tree_vars fml_tree ~f:(fun (v_name, _) ->
    Hashtbl.add_multi b_guards ~key:v_name ~data:fml_id;
    Hashtbl.change notb_guards v_name ~f:(
      Option.map ~f:(List.filter ~f:(fun id -> id <> fml_id))))

let assume_true ~state = assume_bval state.guards state.neg_guards
let assume_false ~state = assume_bval state.neg_guards state.guards

let assume_loop_guard_true ~emit ~state loop =
  assert_token loop WHILE;
  let guard = tree_child loop 0 in
  make_all_edges ~emit ~state guard;
  read_all ~state (tree_tid loop) guard;
  assume_true ~state guard;
  tree_children (tree_child loop 1) |> List.iter ~f:(fun inv ->
    assert_token inv INVARIANT;
    let fml = tree_child inv 0 in
    make_all_edges ~emit ~state fml;
    assume_true ~state fml)

let assume_loop_guard_false ~state loop =
  let guard = tree_child loop 0 in
  assume_false ~state guard

(* ////////////////////////////////////////////////////////////////////////// *)
(* Static analysis                                                            *)
(* ////////////////////////////////////////////////////////////////////////// *)

let analyze_assignment ~emit ~state assign =
  assert_token assign ASSIGN;
  let instr_id = tree_tid assign in
  let lhs = tree_child assign 0 in
  let lhs_v_name, lhs_v_tid = var_symbol lhs |> Option.value_exn in
  let rhs = tree_child assign 1 in
  (* For every var in the rhs, we create semantic edges *)
  iter_tree_vars rhs ~f:(fun (v_name, v_tid) ->
    emit {typ=COMPUTED_FROM; src=v_tid; dst=lhs_v_tid};
    make_edges ~emit ~state (v_name, v_tid));
  (* We also read vars in the RHS *)
  read_all ~state instr_id rhs;
  (* Update the state *)
  Hashtbl.set state.last_written ~key:lhs_v_name ~data:[instr_id];
  Hashtbl.remove state.last_read lhs_v_name;
  Hashtbl.remove state.guards lhs_v_name;
  Hashtbl.remove state.neg_guards lhs_v_name;
  (* We annotate the LHS *)
  make_edges ~emit ~state (lhs_v_name, lhs_v_tid)

let analyze_assume_or_assert ~emit ~state instr =
  let cond = tree_child instr 0 in
  make_all_edges ~emit ~state cond;
  assume_true ~state cond

let rec analyze_program ~emit ~state prog =
  List.iter (tree_children prog) ~f:(analyze_instr ~emit ~state)

and analyze_instr ~emit ~state instr =
  match tree_token_type instr with
  | ASSIGN -> analyze_assignment ~emit ~state instr
  | ASSUME | ASSERT -> analyze_assume_or_assert ~emit ~state instr
  | IF_ELSE -> analyze_if_else ~emit ~state instr
  | WHILE -> analyze_while ~emit ~state instr
  | LABELED_PROG ->
    begin match tree_children instr with
    | [] -> ()
    | [p] -> analyze_program ~emit ~state p
    | _ -> assert false
    end
  | t -> failwith ("Expected instr, got: " ^ [%show: Token.t] t)

and analyze_if_else ~emit ~state instr =
  let cond = tree_child instr 0 in
  let tb, fb = tree_child instr 1, tree_child instr 2 in
  make_all_edges ~emit ~state cond;
  read_all ~state (tree_tid instr) cond;
  (* TRUE branch *)
  let state_f = State.copy state in
  assume_true ~state cond;
  analyze_program ~emit ~state tb;
  (* FALSE branch *)
  assume_false ~state:state_f cond;
  analyze_program ~emit ~state:state_f fb;
  (* Merging *)
  State.join_with state state_f

and analyze_while ~emit ~state instr =
  let body = tree_child instr 2 in
  (* We explore three representative cases: 0, 1 and 2 iterations *)
  (* Zero iterations *)
  let state_0 = State.copy state in
  assume_loop_guard_false ~state:state_0 instr;
  (* One iteration *)
  assume_loop_guard_true ~emit ~state instr;
  analyze_program ~emit ~state body;
  let state_1 = State.copy state in
  assume_loop_guard_false ~state:state_1 instr;
  (* Two iterations *)
  assume_loop_guard_true ~emit ~state instr;
  analyze_program ~emit ~state body;
  assume_loop_guard_false ~state instr;
  (* Merging *)
  State.join_with state state_0;
  State.join_with state state_1

(* ////////////////////////////////////////////////////////////////////////// *)
(* External interface                                                         *)
(* ////////////////////////////////////////////////////////////////////////// *)

let add_prog_semantic_edges g =
  let edges = Queue.create () in
  let emit e = Queue.enqueue edges e in
  let tree = Token_graph.(label_tree_with_indexes g.tree) in
  let state = State.create () in
  analyze_program ~emit ~state tree;
  let edges =
    Queue.to_list edges
    |> List.dedup_and_sort ~compare:Token_graph.compare_edge in
  Token_graph.add_edges g edges
