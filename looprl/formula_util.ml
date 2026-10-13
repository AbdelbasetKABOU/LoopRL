open Base

let sub_formula_with_label label f =
  let exception Found of Formula.t in
  try
    let _: Formula.t =
      Formula.apply_recursively f ~f:(function
      | Formula.Labeled (l, Some sub) when equal_string l label ->
          raise (Found sub)
      | f -> f
    ) in None
  with Found f -> Some f

let remove_label label =
  Formula.apply_recursively ~f:(function
  | Labeled (l, Some fml) when equal_string l label -> fml
  | e -> e)

(* Simplifies a conjunction *)
let remove_redundant ?(assuming=(Formula.Bconst true)) constrs =
  let implies f f' = Arith.surely_valid (Implies ((And [assuming; f]), f')) in
  let rec insert cur = function
    | [] -> cur
    | c::cs ->
      if List.exists cur ~f:(fun cur -> implies cur c)
      then insert cur cs
      else insert (c :: List.filter cur ~f:(fun cur -> not (implies c cur))) cs
  in insert [] constrs |> List.rev

(* A conjunction or disjunction is redundant *)
let is_redundant ?assuming constrs =
  List.length constrs > List.length (remove_redundant ?assuming constrs)

let remove_redundant_conjuncts fml =
  Formula.conjuncts fml |> remove_redundant |> Formula.mk_conj

let%expect_test "remove_redundant" =
  let ex fmls =
    List.map ~f:Parse.formula fmls
    |> remove_redundant ~assuming:(Parse.formula "#n<=0")
    |> [%show: Formula.t list]
    |> Stdio.print_endline in
  ex ["x >= 0"; "x >= 1"; "y != 0"];
  ex ["x >= 1"; "y != 0"; "x >= 0"];
  ex ["x >= 0"; "x >= #n"];
  [%expect {|
    [x >= 1; y != 0]
    [x >= 1; y != 0]
    [x >= 0] |}]

let add_conjunct ~conj fml =
  Formula.mk_conj (Formula.conjuncts fml @ [conj])

let add_disjunct ~disj fml =
  Formula.mk_disj (Formula.disjuncts fml @ [disj])

(* Randomize inequality strictness *)

let swap_strictness (lhs, op, rhs) =
  let open Formula.Compop in
  let open Term in
  let open Term.Infix in
  match op with
  | GE -> (lhs, GT, rhs - one)
  | GT -> (lhs, GE, rhs + one)
  | LE -> (lhs, LT, rhs + one)
  | LT -> (lhs, LE, rhs - one)
  | _ -> (lhs, op, rhs)

let comp_complexity (lhs, _, rhs) =
  List.length (Term.(to_alist (sub lhs rhs)))

let randomize_strictness rng comp =
  if Util.Random.bernouilli ~p:0.5 rng then comp
  else
    let new_comp = swap_strictness comp in
    if comp_complexity new_comp <= comp_complexity comp then new_comp else comp

let randomize_strictness rng = function
  | Formula.Comp (lhs, op, rhs) ->
    let (lhs, op, rhs) = randomize_strictness rng (lhs, op, rhs) in
    Formula.Comp (lhs, op, rhs)
  | f -> f

(* Randomizing the aspect of formulas *)

let shuffle_comparison rng (lhs, op, rhs) =
  let open Term in let open Term.Infix in
  let lhs =
    to_alist (lhs - rhs)
    |> Util.Random.shuffle rng
    |> List.fold ~init:zero ~f:(fun acc (a, c) -> acc + mulc c (atom a)) in
  Arith.prettify (Comp (lhs, op, zero))

let randomize_comparisons rng =
  Formula.apply_recursively ~f:(function
    | Comp (lhs, op, rhs) ->
      (lhs, op, rhs) |> shuffle_comparison rng |> randomize_strictness rng
    | f -> f)

let%expect_test "randomize_comparisons" =
  let fml = Parse.formula "x > y + 3 && i + j - k == 42 && x > n && x <= 4" in
  let rng = Random.State.make [||] in
  for _ = 1 to 10 do
    randomize_comparisons rng fml |> [%show: Formula.t] |> Stdio.print_endline
  done;
  [%expect {|
    ((y <= x - 4 && k - i - j == -42) && n < x) && x < 5
    ((y < x - 3 && k - i - j == -42) && x > n) && x <= 4
    ((x > y + 3 && i + j - k == 42) && n < x) && x <= 4
    ((y < x - 3 && i - k + j == 42) && x > n) && x <= 4
    ((y <= x - 4 && k - j - i == -42) && x > n) && x < 5
    ((y < x - 3 && j + i - k == 42) && n < x) && x <= 4
    ((y < x - 3 && j - k + i == 42) && n < x) && x <= 4
    ((y <= x - 4 && j - k + i == 42) && x > n) && x <= 4
    ((x >= y + 4 && i + j - k == 42) && x > n) && x <= 4
    ((x > y + 3 && i + j - k == 42) && n < x) && x < 5 |}]

(* Simplifying disjunctions *)

let simplify_to_atomic ?(assuming=(Formula.Bconst true)) fml =
  let open Arith in
  let open Formula in
  let settings = {default_abduction_settings with abduct_var_diff = None} in
  match abduct ~settings fml with
  | [] -> Some (Formula.Bconst true)
  | _::_::_ -> None
  | [cands] ->
    List.filter cands ~f:(
    fun f -> surely_valid (Implies (And [assuming; fml], f)))
    |> List.hd

let%expect_test "simplify_to_atomic" =
  let ex i a f =
    let res =
      simplify_to_atomic ~assuming:(Parse.formula a) (Parse.formula f)
      |> [%show: Formula.t option] in
    Stdio.print_endline (Int.to_string i ^ ": " ^ res) in
  ex 1 "true" "x == 0 || x < 0";
  ex 2 "true" "x == 0 || x > 0";
  ex 3 "true" "x == 0 || x >= 1";
  ex 3 "true" "x == y || y > x";
  ex 5 "true" "x == 0 || x >= 2";
  ex 6 "true" "x >= 3 && x != 3";  (* we can make it work in the future *)
  ex 7 "n == 1" "x == n || x > 1";
  [%expect {|
    1: (Some x <= 0)
    2: (Some x >= 0)
    3: (Some x >= 0)
    3: (Some y >= x)
    5: None
    6: None
    7: None |}]

(* Check subpattern *)

let exists_in_formula ~f fml =
  let exception Found in
  try Formula.iter_sub fml ~f:(fun sub -> if f sub then raise Found); false
  with Found -> true

(* Possible consequences *)

let possible_consequences fml =
  (* We have F -> A iff !A -> !F *)
  let settings = Arith.{
    default_abduction_settings
    with abduct_var_diff = None} in
  Arith.abduct ~settings (Not fml)
  |> List.concat
  |> List.dedup_and_sort ~compare:[%compare: Formula.t]
  |> List.map ~f:(fun f -> Arith.(simplify (Not f)))


(* Unit coeffs *)

let vars_appearing_with_unit_coeff fml =
  let vars = ref (Set.empty (module String)) in
  Formula.iter_sub fml ~f:(function
    | Comp (lhs, _, rhs) ->
      let t = Term.sub lhs rhs in
      List.iter (Term.to_alist t) ~f:(function
        | Var x, (-1 | 1) -> vars := Set.add !vars x
        | _ -> ())
    | _ -> ());
  !vars

let%expect_test "vars_appearing_with_unit_coeff" =
  let ex f =
    vars_appearing_with_unit_coeff (Parse.formula f)
    |> Set.to_list
    |> [%show: string list]
    |> Stdio.print_endline in
  ex "3*x + 2*y == 0";
  ex "x <= 2*y";
  ex "-x - 3*y == k -> -x + ?c3 - 3*y - 3*?c2 == k";
  [%expect {|
    []
    ["x"]
    ["?c3"; "k"; "x"] |}]
