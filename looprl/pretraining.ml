(* Generate testing tasks for the neural network. *)

open Base
open Util.Random
open Formula

exception Reject

let ensure p = if not p then raise Reject

let rec sample f rng =
  try f rng with
  | Reject -> sample f rng

let pick = Util.Random.pick

let const ~max_const rng =
  Random.State.int_incl rng (-max_const) max_const

let comparison ~max_const ~vars rng =
  let op = pick rng Compop.[EQ; NE; GE; GT; LE; LT] in
  let lhs = Term.var (pick rng vars) in
  let rhs =
    if bernouilli rng ~p:0.5
    then Term.var (pick rng vars)
    else Term.const (const ~max_const rng) in
  let comp = Comp (lhs, op, rhs) |> Arith.simplify in
  ensure (not (is_true comp || Term.equal lhs rhs));
  comp

let theorem_with ~max_const ~num_assums ~num_irrelevant ~vars rng =
  let sample_clauses k =
    List.init k ~f:(fun _ -> sample (comparison ~max_const ~vars) rng) in
  let clauses = sample_clauses (num_assums + 1) in
  (* The clauses are contradictory but no strict subset is *)
  ensure (Arith.surely_valid (Not (And clauses)));
  ensure (
    List.for_all (Util.Combinatorics.strict_subsets clauses) ~f:(fun cs ->
      Arith.possibly_sat (And cs)));
  match shuffle rng clauses with
  | [] -> assert false
  | c::cs ->
    (* Adding some irrelevant assumptions *)
    let assums = shuffle rng (sample_clauses num_irrelevant @ cs) in
    ensure (Arith.possibly_sat (And assums));
    (assums, Arith.simplify (Not c))

let non_theorem ~max_const ~assums ~vars rng =
  let non_concl = sample (comparison ~max_const ~vars) rng in
  ensure (not (Arith.surely_valid (Implies (And assums, non_concl))));
  non_concl

let sample_task ?(random_sigils=true) rng =
  let vars = ["x"; "y"; "z"] in
  let vars =
    if not random_sigils then vars
    else List.map vars ~f:(fun v ->
      Var.(make (pick rng [Var; Meta_var; Var_hole; Param]) v)) in
  let num_assums = pick rng [1; 2] in
  let num_irrelevant = pick rng [0; 1] in
  let max_const = Util.Random.pick rng [1; 10; 100; 10_000] in
  (* let max_const = Util.Random.pick rng [1; 10] in *)
  let assums, concl =
    sample (theorem_with ~max_const ~num_assums ~num_irrelevant ~vars) rng in
  let vars =
    Set.union_list (module String) (List.map assums ~f:Formula.vars_set)
    |> Set.to_list in
  let non_concl = sample (non_theorem ~max_const ~assums ~vars) rng in
  mk_conj assums, concl, non_concl

let sample_tf_task ?random_sigils rng =
  (* An implication is proposed and the user has to respond 'true' or 'false' *)
  let assums, concl, non_concl = sample_task ?random_sigils rng in
  match Util.Random.bernouilli ~p:0.5 rng with
  | true -> Implies(assums, concl), Bconst true, Bconst false
  | false -> Implies(assums, non_concl), Bconst false, Bconst true

let sample_task ?random_sigils ~true_false rng =
  if true_false then sample_tf_task ?random_sigils rng
  else sample_task ?random_sigils rng

module Probe = struct
  type t = Formula.t [@@deriving sexp]
  let to_string = [%show: Formula.t]
  let to_meta _ = []
  let to_graph f =
    Token_graph.compose
      (Token.tok FIND_CONSEQUENCE) [Tokenize.formula f]
end

module Choice_summary = struct
  type t = Formula.t [@@deriving sexp]
  let to_string = [%show: Formula.t]
  let to_meta _ = []
  let to_graph = Tokenize.formula
end

let%expect_test "network_test_tasks" =
  let rng = Random.State.make [||] in
  let show = [%show: Formula.t] in
  List.init 20 ~f:(fun _ ->
    sample_task ~random_sigils:false ~true_false:false rng)
  |> List.iter ~f:(fun (assum, concl, non_concl) ->
      Fmt.pr "%40s %15s %15s\n" (show assum) (show concl) (show non_concl));
  [%expect {|
        z > x && x <= 6925 && x >= y          z >= y          y >= z
                   x == 35 && x != z         x != 66        z == -54
                   z <= 37 && z != x          z < 91          x > 15
         z >= y && y < -2 && z == -7           y < 5         z >= -5
                              y == z          z == y          z <= 3
            x != 0 && y < z && x > z           y < x         x == -7
    y != -5503 && x >= z && x == 295       z != 8048       x == 5887
                 y == 1867 && y == x        x < 2115       y <= 1560
                 z == 9667 && y >= z      z >= -5613      z == -2734
                              x == 1          x == 1          x == 0
          x != -1 && z >= x && x > 0         z != -1           z > 1
                     x > y && z <= 0          y <= x          x == z
                           x == 2123       x != 5857      x == -2677
                    y > z && z >= 54         z > -66           y < z
                 x == -440 && y <= x      x >= -6616          y >= x
                  z <= -5 && y >= -7           z < 4          y >= z
                   y == z && y >= 10          z >= 6          z < -6
                 z >= 1190 && z == x       x > -6176        x > 2000
                   z == -1 && x != z          z <= 0          z >= x
                            y <= -18          y < -9        y <= -77 |}]