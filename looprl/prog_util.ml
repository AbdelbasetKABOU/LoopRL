open Base
open Prog

(* ////////////////////////////////////////////////////////////////////////// *)
(* Program manipulation utilities                                             *)
(* ////////////////////////////////////////////////////////////////////////// *)

(* Lenses to access invariants *)

let get_first_loop (Prog instrs) =
  List.find instrs ~f:(function While _ -> true | _ -> false)

let set_first_loop repl (Prog instrs) =
  let rec aux = function
  | [] -> []
  | (While _):: rest -> repl::rest
  | i::is -> i::aux is in
  Prog (aux instrs)

let first_loop = Lens.{
  get = (fun p -> get_first_loop p |> Option.value_exn);
  set = set_first_loop;}

let get_invariants = function
  | While (_, invs, _) -> invs
  | _ -> assert false

let set_invariants invs = function
  | While (guard, _, body) -> While (guard, invs, body)
  | _ -> assert false

let invariants = Lens.{
  get = get_invariants;
  set = set_invariants}

(* Adding instructions *)

let append_instr label instr prog =
  Prog.apply_recursively prog ~f:(function
  | LabeledProg (l, arg) when equal_string label l ->
    LabeledProg (l, Some (Prog
      begin match arg with
      | None -> [instr]
      | Some (Prog instrs) -> instrs @ [instr]
      end))
  | i -> i)

let replace_final_instr ~final (Prog instrs) =
  let instrs = List.rev instrs |> List.tl_exn |> List.rev in
  Prog (instrs @ prog_instrs final)

(* Manipulating labels *)

let remove_prog_label label =
  Prog.map_instrs ~f:(function
  | LabeledProg (l, Some p) when equal_string l label -> p
  | i -> Prog [i])

let remove_formula_label label =
  Prog.map_formulas ~f:(Formula_util.remove_label label)

let get_prog_with_label label p =
  let exception Found of Prog.t in
  try
    let _: Prog.t = Prog.apply_recursively p ~f:(function
      | LabeledProg (l, Some sub) when equal_string l label ->
          raise (Found sub)
      | i -> i
    ) in None
  with Found i -> Some i

let get_formula_with_label label p =
  let exception Found of Formula.t in
  try
    let _: Prog.t =
      Prog.map_formulas p ~f:(
        Formula.apply_recursively ~f:(function
        | Formula.Labeled (l, Some sub) when equal_string l label ->
            raise (Found sub)
        | f -> f)
    ) in None
  with Found f -> Some f

let remove_all_labels_in_formula ~except =
  Formula.apply_recursively ~f:(function
    | Formula.Labeled (l, Some f)
      when not (List.mem ~equal:String.equal except l) -> f
    | f -> f)

let remove_all_labels ?(except=[]) p =
  Prog.map_instrs p ~f:(function
    | LabeledProg (l, Some p)
      when not (List.mem ~equal:String.equal except l) -> p
    | i -> Prog [i])
  |> Prog.map_formulas ~f:(remove_all_labels_in_formula ~except)

let set_formula_with_label label sub =
  let substituted = Formula.Labeled (label, Some sub) in
  Prog.subst_pred_symbol ~from:label ~substituted

let set_prog_with_label label p =
  let substituted = Prog [LabeledProg (label, Some p)] in
  Prog.subst_prog_symbol ~from:label ~substituted

let formula_with_label label = Lens.{
  get = (fun p -> get_formula_with_label label p |> Option.value_exn);
  set = set_formula_with_label label}

let prog_with_label label = Lens.{
  get = (fun p -> get_prog_with_label label p |> Option.value_exn);
  set = set_prog_with_label label}

(* Manipulating param assumptions *)

let split_initial_param_assums (Prog instrs) =
  let is_param_constr c =
    Formula.vars_set c |> Set.for_all ~f:Var.(has_kind Param) in
  let rec aux = function
  | (Assume c)::instrs when is_param_constr c ->
    let assums, rest = aux instrs in
    c::assums, rest
  | rest -> [], Prog rest in
  aux instrs

let get_initial_param_assums p =
  fst (split_initial_param_assums p)

let set_initial_param_assums assums p =
  let _, (Prog instrs) = split_initial_param_assums p in
  Prog ((List.map assums ~f:(fun a -> Assume a)) @ instrs)

let initial_param_assums = Lens.{
  get = get_initial_param_assums;
  set = set_initial_param_assums }

(* Parameter variables *)

(* Returns the set of variables that are not modified in the loop body. *)
let parameter_vars p =
  let modified_in_loop =
    match first_loop.get p with
    | While (_, _, body) -> Prog.modified_vars body
    | _ -> assert false in
  Set.diff (Prog.vars_set p) modified_in_loop

(* Task normalization *)

(* Transform "if P {assert Q;}" into "assume P; assert Q;" *)
let rec remove_post_cond p =
  match List.rev (prog_instrs p) with
  | (If (c, tb, Prog []))::rev_is ->
    Prog (List.rev rev_is @ [Assume c] @ prog_instrs (remove_post_cond tb))
  | _ -> p

let normalize_task p = remove_post_cond p

(* ////////////////////////////////////////////////////////////////////////// *)
(* Tests                                                                      *)
(* ////////////////////////////////////////////////////////////////////////// *)

let test_program = Parse.program
  {|
    init: {}
    while ((guard: x>=1 && (C: y>=0))) {
      invariant (I: x + y == ?c);
      body: {
        x = x + 1;
      }
    }
    assert (post: x>=0);
  |}

let%expect_test "remove_labels" =
  test_program
    |> remove_formula_label "guard"
    |> remove_prog_label "body"
    |> [%show: Prog.t] |> Stdio.print_endline;
  [%expect {|
    init: { }
    while (x >= 1 && (C: y >= 0)) {
        invariant (I: x + y == ?c);
        x = x + 1;
    }
    assert (post: x >= 0); |}]

let%expect_test "remove_all_labels" =
  test_program
    |> remove_all_labels ~except:["I"]
    |> [%show: Prog.t] |> Stdio.print_endline;
  [%expect {|
    while (x >= 1 && y >= 0) {
        invariant (I: x + y == ?c);
        x = x + 1;
    }
    assert x >= 0; |}]


let%expect_test "formula_with_label" =
  let ex label =
    get_formula_with_label label test_program
      |> [%show: Formula.t option] |> Stdio.print_endline in
  ex "C"; ex "guard"; ex "I"; ex "absent";
  [%expect {|
    (Some y >= 0)
    (Some x >= 1 && (C: y >= 0))
    (Some x + y == ?c)
    None |}]

let%expect_test "prog_with_label" =
let ex label =
 get_prog_with_label label test_program
    |> [%show: Prog.t option] |> Stdio.print_endline in
ex "guard"; ex "init"; ex "body"; ex "absent";
[%expect {|
  None
  (Some )
  (Some x = x + 1;)
  None |}]

let%test_unit "params_transformations" =
  let prog = {|
      n = 15;
      x = 0;
      while (x < n) { x = x + 1; }
      assert x == n;
    |} |> Parse.program in
  let params = parameter_vars prog in
  assert (Set.equal params (Set.of_list (module String) ["n"]))


(* ////////////////////////////////////////////////////////////////////////// *)
(* Check patterns presence                                                    *)
(* ////////////////////////////////////////////////////////////////////////// *)

let exists_in_prog ~f prog =
  let exception Found in
  try Prog.iter_instrs prog ~f:(fun instr -> if f instr then raise Found); false
  with Found -> true