(** Simplification engine.
    In FM normal form, variables, parameters and function symbols are on
    the left and constants are on the right.
    The GCD of all coeffs is also equal to 1.
    Right now, we only use FM to eliminate vars on the LHS. *)

open Term
open Formula
open Base

(* ////////////////////////////////////////////////////////////////////////// *)
(* Parameters                                                                 *)
(* ////////////////////////////////////////////////////////////////////////// *)

type fm_settings = {
  timeout: int;
  max_clauses: int}

type abduction_settings = {
  fm: fm_settings;
  abduct_var_diff: fm_settings option }

let fm_default_settings = {
  timeout = 100;
  max_clauses = 20 }

let default_abduction_settings = {
  fm = fm_default_settings;
  abduct_var_diff = Some {timeout=10; max_clauses = 20;} }

(* In comparison, the age of a clause has a cost factor of 1 *)
let fm_cost_per_var = function
  | Var.Var -> 2
  | _ -> 1

let fm_max_allowed_coeff = 5
let fm_max_allowed_const = 1_000_000

(* ////////////////////////////////////////////////////////////////////////// *)
(* Working with comparisons                                                   *)
(* ////////////////////////////////////////////////////////////////////////// *)

type comparison = Comp of Term.t * Compop.t * Term.t

let normalize_compop (Comp (lhs, op, rhs)) =
  let open Compop in
  let open Term.Infix in
  match op with
  | GT -> Comp (lhs, GE, rhs + const 1) (* X > Y <-> X >= Y + 1 *)
  | LT -> Comp (-lhs, GE, - rhs + const 1) (* X < Y <-> -X >= -Y+1 *)
  | LE -> Comp (-lhs, GE, -rhs) (* X <= Y <-> -X >= -Y *)
  | GE | EQ | NE -> Comp (lhs, op, rhs)

let move_all_left (Comp (lhs, op, rhs)) =
  Comp (Term.sub lhs rhs, op, Term.zero)

let normalize_coeffs (Comp (lhs, op, rhs) as comp) =
  let d = Util.Math.gcd (coeffs_gcd lhs) (coeffs_gcd rhs) in
  assert (d >= 0);
  if d > 1 then Comp (Term.divc lhs d, op, Term.divc rhs d)
  else comp

let is_meta v = Var.(has_kind Meta_var v)

let separate_constants (Comp (lhs, op, rhs)) =
  assert Term.(equal rhs zero);
  let remain_left = function
    | One -> false
    | Var _ | FunApp _ -> true in
  let lhs, neg_rhs = Term.partition lhs ~f:remain_left in
  Comp (lhs, op, Term.Infix.(-neg_rhs))

let comp_of_formula = function
  | Formula.Comp (lhs, op, rhs) -> Some (Comp (lhs, op, rhs))
  | _ -> None

let comp_of_formula_exn = function
  | Formula.Comp (lhs, op, rhs) -> Comp (lhs, op, rhs)
  | _ -> assert false

let formula_of_comp (Comp (lhs, op, rhs)) = Formula.Comp (lhs, op, rhs)

let pp_comparison = Fmt.using formula_of_comp Formula.pp

let normalize_comparison comp =
  comp
  |> move_all_left
  |> normalize_coeffs
  |> separate_constants
  |> normalize_compop

let%expect_test "normalize_comparison" =
  let ex s =
    let comp = Parse.formula s in
    let comp' =
      comp |> comp_of_formula_exn
      |> normalize_comparison |> formula_of_comp in
    Fmt.pr "%a  ~>  %a\n" Formula.pp comp Formula.pp comp' in
  ex "x >= y + 2";
  ex "y + ?c < x + ?d";
  [%expect {|
    x >= y + 2  ~>  x - y >= 2
    y + ?c < x + ?d  ~>  -y - ?c + x + ?d >= 1 |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* FM reasoning steps                                                         *)
(* ////////////////////////////////////////////////////////////////////////// *)

let fm_combine_compop =
  let open Compop in
  function
  | EQ, op | op, EQ -> Some op
  | GE, GE -> Some GE
  | _ -> None

let fm_pos_coeffs_only =
  let open Compop in
  function
  | GE -> true
  | _ -> false

let fm_rework_coeffs (c, pos) (c', pos') =
  let c, c' = if pos  && c < 0  then (-c, -c') else (c, c') in
  let c, c' = if pos' && c' < 0 then (-c, -c') else (c, c') in
  if (not pos || c > 0) && (not pos' || (c' > 0))
  then Some (c, c')
  else None

let fm_critical_pairs lhs lhs' =
  let rec aux = function
    | [] -> []
    | (Var x, c)::lhs ->
      begin match Term.coeff (Var x) lhs' with
      | None -> aux lhs
      | Some c' -> (c', -c) :: aux lhs
      end
    | _::lhs -> aux lhs in
  aux (Term.to_alist lhs)

let fm_step (Comp (lhs, op, rhs)) (Comp (lhs', op', rhs')) =
  match fm_combine_compop (op, op') with
  | None -> []
  | Some new_op ->
    let pos, pos' = fm_pos_coeffs_only op, fm_pos_coeffs_only op' in
    let crit =
      fm_critical_pairs lhs lhs'
      |> List.filter_map ~f:(
        fun (c, c') -> fm_rework_coeffs (c, pos) (c', pos'))
      |> List.dedup_and_sort ~compare:[%compare: int * int] in
    List.map crit ~f:(fun (c', c) ->
      let open Term.Infix in
      Comp (c' * lhs + c * lhs', new_op, c' * rhs + c * rhs')
      |> normalize_coeffs)

let read_normalized_comp s =
  Parse.formula s |> comp_of_formula_exn |> normalize_comparison

let%expect_test "fm_step" =
  let ex s s' =
    let comp = read_normalized_comp s in
    let comp' = read_normalized_comp s' in
    let res = fm_step comp comp' in
    Fmt.pr "%s\n" ([%show: comparison list] res) in
  ex "x >= 0" "x + y >= 0";
  ex "x >= 0" "x + y <= 0";
  ex "x >= ?c" "x < ?d";
  (* missing reasoning principle. *)
  ex "x >= 0" "x != 0";
  ex "x + y == 1" "x + y == 2";
  ex "x == y" "y != z";
  ex "x == y + 2" "y >= z";
  ex "x >= y" "y != z";
  ex "x + y >= a" "x + y <= b";
  ex "-j + i >= 1" "j - i >= -3";
  ex "x>=0" "x!=0";
  [%expect {|
    []
    [-y >= 0]
    [-?c + ?d >= 1]
    []
    [0 == -1]
    [x - z != 0]
    [x - z >= 2]
    []
    [-a + b >= 0]
    [0 >= -1]
    [] |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Optimization: deriving equalities from integer bounds                      *)
(* ////////////////////////////////////////////////////////////////////////// *)

type multiples = NoMultiples | OneMultiple of int | SeveralMultiples
  [@@deriving show]

let multiples_of_num_in ~num (lb, ub) =
  assert (lb <= ub);
  let v = Int.(ub /% num) * num in
  if v < lb then NoMultiples
  else if v - num < lb then OneMultiple v
  else SeveralMultiples

let%expect_test "multiples_of_num_in" =
  let ex num (lb, ub) =
    let res = multiples_of_num_in ~num (lb, ub) in
    Stdio.print_endline ([%show: multiples] res) in
  ex 3 (-2, 2);
  ex 3 (-3, 2);
  ex 3 (1, 2);
  [%expect {|
    (Arith.OneMultiple 0)
    Arith.SeveralMultiples
    Arith.NoMultiples |}]

let derive_const_equality (Comp (lhs, op, rhs)) (Comp (lhs', op', rhs')) =
  (* if we have aX >= b and a'X >= b' with sig(a) <> sign(a') *)
  match
    (Term.to_alist lhs, op, Term.get_const rhs),
    (Term.to_alist lhs', op', Term.get_const rhs')
  with
  | ([Var x, a], GE, Some b), ([Var x', a'], GE, Some b')
    when equal_string x x' && a*a'< 0 ->
      (* Make sure that a > 0 and a' > 0 *)
      let a, b, a', b' = if a < 0 then a', b', a, b else a, b, a', b' in
      (* Put in in form aX >= b, -aX >= b' with a > 0 *)
      let a, b, b' = (-a')*a, (-a')*b, a*b' in
      (* Put it in form lb <= aX <= ub *)
      let lb, ub = b, -b' in
      (* there is a single multiple of a in an interval of length  *)
      if lb <= ub then
        (* v is the only possible value for x *)
        begin match multiples_of_num_in ~num:a (lb, ub) with
          | NoMultiples ->
              Some (Comp (const 0, EQ, const 1))  (* inconsistency detected *)
          | OneMultiple m ->
             Some (Comp (var x, EQ, const (m / a)))
          | SeveralMultiples -> None
        end
      else None
  | _ -> None

let%expect_test "derive_const_equality" =
  let ex c c' =
    let c, c' = read_normalized_comp c, read_normalized_comp c' in
    let res = derive_const_equality c c' |> Option.map ~f:formula_of_comp in
    Stdio.print_endline ([%show: Formula.t option] res) in
  ex "-2 <= 2*x" "2*x <= -1";
  ex "-1 <= 2*x" "2*x <=  0";
  ex " 0 <= 2*x" "2*x <=  1";
  ex " 1 <= 2*x" "2*x <=  2";
  ex " 1 <= 3*x" "3*x <=  2";
  ex " 1 <= 3*x" "3*x <=  4";
  ex " 1 <= 3*x" "3*x <=  3";
  ex " 0 <= 3*x" "3*x <=  3";
  [%expect{|
    (Some x == -1)
    (Some x == 0)
    (Some x == 0)
    (Some x == 1)
    (Some 0 == 1)
    (Some x == 1)
    (Some x == 1)
    None |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Optimization: X>=Y & X!=Y -> X>=Y+1                                        *)
(* ////////////////////////////////////////////////////////////////////////// *)

let rec combine_ge_ne (Comp (lhs, op, rhs)) (Comp (lhs', op', rhs')) =
  let open Term in
  let open Term.Infix in
  match op, op' with
  | GE, NE ->
    if equal (lhs - lhs') zero && equal (rhs - rhs') zero ||
       equal (lhs + lhs') zero && equal (rhs + rhs') zero
    then Some (Comp (lhs, GE, rhs + one))
    else None
  | NE, GE -> combine_ge_ne (Comp (lhs', op', rhs')) (Comp (lhs, op, rhs))
  | _ -> None

(* ////////////////////////////////////////////////////////////////////////// *)
(* Optimization: X>=Y & -X>=-Y -> X=Y                                         *)
(* ////////////////////////////////////////////////////////////////////////// *)

let double_ineq_to_eq (Comp (lhs, op, rhs)) (Comp (lhs', op', rhs')) =
  let open Term.Infix in
  if Term.(equal (lhs + lhs') zero) && Term.(equal (rhs + rhs') zero) then
  match op, op' with
  | GE, GE -> Some (Comp (lhs, EQ, rhs))
  | _ -> None
  else None

(* ////////////////////////////////////////////////////////////////////////// *)
(* FM Utilities                                                               *)
(* ////////////////////////////////////////////////////////////////////////// *)

(* Classifying clause types *)

type clause_status =
  | Normal
  | Boolean of bool

let clause_status (Comp (lhs, op, rhs)) =
  if not Term.(equal lhs zero) then Normal
  else begin match get_const rhs with
  | None -> failwith "The comparison is not in FM-normal form."
  | Some c ->
    let v = Compop.interpret op 0 c in
    Boolean v
  end

(* Testing clause compatibility *)

let incompatible c c' =
  List.exists (fm_step c c') ~f:(fun res ->
    match clause_status res with
    | Boolean false -> true
    | _ -> false)

let negate_comp (Comp (lhs, op, rhs)) =
  Comp (lhs, Compop.not op, rhs) |> normalize_compop

let implies c c' = incompatible c (negate_comp c')

(* Clauses with age annotations *)

type annotated_comparison = {comp: comparison; age: int}

let clause_cost {comp=Comp (lhs, _, _); age} =
  let vars_cost =
    Set.fold (Term.vars_set lhs)
      ~init:0 ~f:(fun c v -> c + fm_cost_per_var (Var.kind v)) in
  age + vars_cost

let term_has_small_consts t =
  List.for_all (Term.to_alist t) ~f:(function
    | (One, c) -> Int.abs c <= fm_max_allowed_const
    | (_, c) -> Int.abs c <= fm_max_allowed_coeff)

let has_small_consts (Comp (lhs, _, rhs)) =
  term_has_small_consts lhs && term_has_small_consts rhs

let derive_consequences c c' =
  let age = c.age + c'.age + 1 in
  let derived =
    fm_step c.comp c'.comp @
    Option.to_list (derive_const_equality c.comp c'.comp) @
    Option.to_list (combine_ge_ne c.comp c'.comp) @
    Option.to_list (double_ineq_to_eq c.comp c'.comp) in
  let derived = List.filter derived ~f:has_small_consts in
  List.map derived ~f:(fun comp -> {comp; age})

(* ////////////////////////////////////////////////////////////////////////// *)
(* FM state                                                                   *)
(* ////////////////////////////////////////////////////////////////////////// *)

module State = struct

  module Pqueue = Util.Pqueue

  type t = {
    mutable pending: annotated_comparison Pqueue.t;
    clauses: annotated_comparison Queue.t;
    other: Formula.t Queue.t;
    mutable false_derived: bool}

  let create () = {
    pending = Pqueue.empty;
    clauses = Queue.create ();
    other = Queue.create ();
    false_derived = false}

  let copy st = {
    pending = st.pending;
    clauses = Queue.copy st.clauses;
    other = Queue.copy st.other;
    false_derived = st.false_derived}

  let add_pending st c =
    let cost = clause_cost c in
    st.pending <- Pqueue.insert st.pending cost c

  let has_pending st = not (Pqueue.is_empty st.pending)

  let pop_pending st =
    let _, c, pending = Pqueue.extract st.pending in
    st.pending <- pending;
    c

end

(* ////////////////////////////////////////////////////////////////////////// *)
(* FM (main algorithm)                                                        *)
(* ////////////////////////////////////////////////////////////////////////// *)

exception False_derived

let redundant_clause clauses c =
  Queue.exists clauses ~f:(fun old -> implies old.comp c.comp)

let insert_clause clauses c =
  Queue.filter_inplace clauses ~f:(fun old -> not (implies c.comp old.comp));
  Queue.enqueue clauses c

let add_pending_clause st c =
  let open State in
  match clause_status c.comp with
  | Boolean true -> ()
  | Boolean false -> (st.false_derived <- true; raise False_derived)
  | Normal -> State.add_pending st c

let assert_formulas st fmls =
  (* Normalize comparisons and add them in the pending queue. *)
  let open State in
  try
  List.iter fmls ~f:(fun f ->
    match comp_of_formula f with
    | Some comp ->
      let comp = normalize_comparison comp in
      add_pending_clause st {comp; age=0}
    | None -> Queue.enqueue st.other f)
  with False_derived -> ()

let saturate ~settings st =
  let open State in
  let step = ref 0 in
  if not st.false_derived then begin
  try
    while
        !step < settings.timeout &&
        Queue.length st.clauses < settings.max_clauses &&
        State.has_pending st
    do
      Int.incr step;
      let c = State.pop_pending st in
      (* We have a formula. We assume it is not trivial *)
      (* We ensure that it is not redundant. *)
      if not (redundant_clause st.clauses c) then begin
        (* We add consequences to the pending list.*)
        Queue.iter st.clauses ~f:(fun ex ->
          List.iter (derive_consequences ex c) ~f:(fun res ->
            add_pending_clause st res));
        (* We remove weaker versions me may already have and add the clause. *)
        insert_clause st.clauses c
      end
    done
  with False_derived -> ()
  end

(* Tests *)

let fourrier_motzkin ?(settings=fm_default_settings) fmls =
  let st = State.create () in
  assert_formulas st fmls;
  saturate ~settings st;
  st

let saturate_constraints ?(settings=fm_default_settings) fmls =
  let st = fourrier_motzkin ~settings fmls in
  if st.false_derived then [Bconst false]
  else
    st.State.clauses
    |> Queue.to_list
    |> List.map ~f:(fun c -> formula_of_comp c.comp)

let string_of_clauses clauses =
  Queue.to_list clauses |> List.map ~f:(fun c -> formula_of_comp c.comp)

let%expect_test "fm" =
  let ex clauses =
    let clauses = List.map clauses ~f:Parse.formula in
    let settings = {max_clauses=10; timeout=200} in
    let st = fourrier_motzkin ~settings clauses in
    let rem_clauses = string_of_clauses st.State.clauses in
    (st.false_derived, rem_clauses)
    |> [%show: bool * Formula.t list] |> Stdio.print_endline in
  ex ["x>=0"; "y>=0"; "-x-y>=1"];
  ex ["x>=0"; "y>=0"; "x+y>=0"];
  ex ["x>=1"; "x>=0"];
  ex ["j<i"; "j>=i-3"; "2*j+i==41"; "j!=13"];
  ex ["j<i"; "j>=i-3"; "2*j+i==41"];
  ex ["j<i"; "j>=i-3"];
  ex ["x>=0"; "x!=0"];
  ex ["?c >= -2"; "?c >= 0"; "?c != 0"];
  ex ["x<=y+2"; "x>y"; "x-3*y==7"];
  [%expect{|
    (true, [x >= 0; y >= 0; -x - y >= 1])
    (false, [x >= 0; y >= 0; x + y >= 0])
    (false, [x >= 1])
    (true,
     [j != 13; j - i >= -3; -j + i >= 1; 2*j + i == 41; -i != -15; -3*i >= -47;
       3*j >= 38; -3*j >= -40; 3*i >= 43])
    (false, [-j + i >= 1; 2*j + i == 41; j - i >= -3; j == 13; i == 15])
    (false, [-j + i >= 1; j - i >= -3])
    (false, [x >= 1])
    (false, [?c >= 1])
    (false, [-x + y >= -2; x - 3*y == 7; x - y >= 1; y == -3; x == -2]) |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Abducting x >= y + ?c                                                      *)
(* ////////////////////////////////////////////////////////////////////////// *)

let find_vars ~f clauses =
  Queue.fold clauses
    ~init:(Set.empty (module String))
    ~f:(fun vars c -> Set.union (f c) vars)

let bounded_dyn_vars pcoeff clauses =
  find_vars clauses ~f:(fun {comp=(Comp (lhs, op, _)); _} ->
    let empty = Set.empty (module String) in
    match op with
    | Compop.GE ->
      List.fold (Term.to_alist lhs) ~init:empty ~f:(fun acc (a, c) ->
        match a with
        | Var x when pcoeff c && Var.(has_kind Var x) -> Set.add acc x
        | _ -> acc)
    | _ -> empty)

let lower_bounded_vars = bounded_dyn_vars (fun c -> c > 0)
let upper_bounded_vars = bounded_dyn_vars (fun c -> c < 0)

let find_ge_candidates clauses =
  (* If ... >= X and Y >= ... we suggest X >= Y *)
  let ub_vars = upper_bounded_vars clauses |> Set.to_list in
  let lb_vars = lower_bounded_vars clauses |> Set.to_list in
  List.cartesian_product ub_vars lb_vars
    |> List.filter ~f:(fun (x, y) -> not (equal_string x y))

let recognize_fresh_bound fresh (Comp (lhs, op, rhs)) =
  (* Recognize a bound of the form '?c>=K' with K an int literal *)
  if Term.equal lhs (var fresh) && Compop.(equal op GE)
  then Term.get_const rhs
  else None

(* Tests *)

let read_comparisons clauses =
  clauses
  |> List.map ~f:Parse.formula
  |> List.map ~f:comp_of_formula_exn
  |> List.map ~f:(fun comp -> {comp; age=0})
  |> Queue.of_list

let%expect_test "lower_bounded_vars" =
  let ex clauses =
    let clauses = read_comparisons clauses in
    let lb_vars = lower_bounded_vars clauses in
    let ub_vars = upper_bounded_vars clauses in
    Fmt.pr "Lower bounded: %s, upper bounded: %s\n"
      ([%show:string list] (Set.to_list lb_vars))
      ([%show:string list] (Set.to_list ub_vars)) in
  ex ["x - y >= 0"; "y - z >= 0"];
  [%expect {| Lower bounded: ["x"; "y"], upper bounded: ["y"; "z"] |}]

let%expect_test "find_ge_candidates" =
  let ex clauses =
    let clauses = read_comparisons clauses in
    let cands = find_ge_candidates clauses in
    Stdio.print_endline ([%show: (string * string) list] cands) in
  ex ["x - y >= 0"; "y - z >= 0"];
  [%expect {| [("y", "x"); ("z", "x"); ("z", "y")] |}]

let ge_candidates ~settings st =
  List.filter_map (find_ge_candidates st.State.clauses) ~f:(fun (x, y) ->
    let open Term in
    let st = State.copy st in
    let fresh = "??c" in
    let comp = Comp (sub (var x) (var y), Compop.GE, var fresh) in
    assert_formulas st [formula_of_comp comp];
    saturate ~settings st;
    Queue.find_map st.clauses ~f:(fun c ->
      Option.map (recognize_fresh_bound fresh (negate_comp c.comp)) ~f:(fun k ->
        Formula.Comp (sub (var x) (var y), Compop.GE, const k)
      )))

(* ////////////////////////////////////////////////////////////////////////// *)
(* Simplification                                                             *)
(* ////////////////////////////////////////////////////////////////////////// *)

module Build = struct

  let bconst b = Bconst b

  let conj args =
    (* Step 1: flatten *)
    let args =
      if List.exists args ~f:(function
        | And _ -> true
        | _ -> false)
      then List.concat_map args ~f:(function
        | And cs -> cs
        | a -> [a])
      else args in
    (* 2. test if a conjunct is false *)
    if List.exists ~f:is_false args then bconst false
    else
      (* 3. remove the true conjuncts *)
      let args = List.filter ~f:(fun a -> not (is_true a)) args in
      (* 4. handle the cases where 0 or 1 conjuncts remain *)
      match args with
      | [] -> bconst true
      | [x] -> x
      | _ -> And args

  let disj args =
    let args =
      if List.exists args ~f:(function
        | Or _ -> true
        | _ -> false)
      then List.concat_map args ~f:(function
        | Or cs -> cs
        | a -> [a])
      else args in
    if List.exists ~f:is_true args then bconst true
    else
      let args = List.filter ~f:(fun a -> not (is_false a)) args in
      match args with
      | [] -> bconst false
      | [x] -> x
      | _ -> Or args

  let rec lnot = function
    | Unknown -> Unknown
    | Labeled _ as e -> Not e
    | Bconst b -> Bconst (not b)
    | And args -> disj (List.map ~f:lnot args)
    | Or args -> conj (List.map ~f:lnot args)
    | Not arg -> arg
    | Implies (lhs, rhs) -> conj [lhs; lnot rhs]
    | Comp (lhs, op, rhs) -> Comp (lhs, Compop.not op, rhs)

  let implies lhs rhs =
    if is_true rhs then bconst true
    else if is_false lhs then bconst true
    else if is_false rhs then lnot lhs
    else if is_true lhs then rhs
    else
      match rhs with
      (* | Implies (lhs', rhs') ->
        Implies (Annot.neutral, conj [lhs; lhs'], rhs') *)
      | _ -> Implies (lhs, rhs)

end

let rec normalize e =
  match e with
  | And args -> Build.conj (List.map ~f:normalize args)
  | Or args -> Build.disj  (List.map ~f:normalize args)
  | Not arg -> Build.lnot (normalize arg)
  | Implies (lhs, rhs) ->
      Build.implies (normalize lhs) (normalize rhs)
  | Comp (lhs, op, rhs) ->
    (* Evaluate expressions that only feature constants *)
    begin match comp_eval lhs op rhs with
    | None -> e
    | Some b -> Bconst b
    end
  | Labeled (s, Some arg) -> Labeled (s, Some (normalize arg))
  | Bconst _ | Unknown | Labeled (_, None) -> e

exception Remaining_symbol_or_unknown

let remove_unknowns_and_symbols e =
  let open Formula in
  let rec aux cov = function
    | And args -> And (List.map args ~f:(aux cov))
    | Or args -> Or (List.map args ~f:(aux cov))
    | Not arg -> Not (aux (-cov) arg)
    | Implies (lhs, rhs) -> Implies (aux (-cov) lhs, aux cov rhs)
    | Unknown ->
      if cov = 1 then Bconst false
      else if cov = -1 then Bconst true
      else raise Remaining_symbol_or_unknown
    | Labeled (_, None) -> raise Remaining_symbol_or_unknown
    | Labeled (_, Some f) -> aux cov f
    | Bconst _ | Comp _ as e -> e in
  aux 1 e

let propagate_assumptions e =
  let implies c c' =
    implies (normalize_comparison c) (normalize_comparison c') in
  let get_assumptions = function
    | And args -> List.filter_map args ~f:comp_of_formula
    | e -> comp_of_formula e |> Option.to_list in
  let rec aux assums = function
    | And args -> And (List.map ~f:(aux assums) args)
    | Or args -> Or (List.map ~f:(aux assums) args)
    | Implies (lhs, rhs) ->
        let assums = assums @ get_assumptions lhs in
        Implies (lhs, aux assums rhs)
    | Formula.Comp (lhs, op, rhs) as e ->
        let c = Comp (lhs, op, rhs) in
        if List.exists assums ~f:(fun a -> implies a c)
        then Bconst true else e
    | Labeled (l, Some arg) -> Labeled (l, Some (aux assums arg))
    | Bconst _ | Unknown | Labeled (_, None) | Not _ as e -> e in
  aux [] e

let simplify e = normalize e |> propagate_assumptions |> normalize

let%expect_test "simplify" =
  let ex s =
    let fml = Parse.formula s in
    let simplified = simplify fml in
    Fmt.pr "   %a\n~> %a\n"
      Formula.pp fml Formula.pp simplified in
  ex "x >= 1 -> x >= 0";
  ex "x>z && y>=0 -> y>=1 && x>z";
  ex "1==1 && !(x<0) -> y>=1 || (0==1)";
  ex "x==1 -> x>=0";
  [%expect{|
       x >= 1 -> x >= 0
    ~> true
       x > z && y >= 0 -> y >= 1 && x > z
    ~> x > z && y >= 0 -> y >= 1
       1 == 1 && !(x < 0) -> y >= 1 || 0 == 1
    ~> x >= 0 -> y >= 1
       x == 1 -> x >= 0
    ~> true |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Human readable form                                                        *)
(* ////////////////////////////////////////////////////////////////////////// *)

let leading_coeff t =
  match Term.to_alist t with
  | [] -> 0
  | (_, c) :: _ -> c

let ensure_pos_leading_coeff (Comp (lhs, op, rhs) as c) =
  let open Term.Infix in
  let open Compop in
  if leading_coeff lhs < 0 then Comp (-lhs, reverse op, -rhs)
  else c

let prettify_comp_move ~meta  c =
  let Comp (lhs, op, _) =
    normalize_comparison c |> move_all_left |> ensure_pos_leading_coeff in
  (* In general, we move everything but the constants and constant symbols
     on the left. There is one exception though in cases like "x-y op ...",
     which are rewritten "x op y ..." *)
  let to_lhs =
      if meta then function
      | Var x when is_meta x -> true
      | _ -> false
      else function
      | Var x when not (is_meta x) -> true
      | FunApp _ -> true
      | _ -> false in
  let lhs, neg_rhs = Term.partition lhs ~f:to_lhs in
  (* Exception for cases like 'x - y op ...' *)
  let open Term.Infix in
  let lhs, neg_rhs =
    match Term.to_alist lhs with
      | [(Var x, 1); (Var y, -1)] -> var x, - (var y) + neg_rhs
      | _ -> lhs, neg_rhs in
  Comp (lhs, op, -neg_rhs) |> ensure_pos_leading_coeff

let const_coeff t =
  List.find_map (Term.to_alist t) ~f:(
    function
    | One, c -> Some c
    | _ -> None)
  |> Option.value ~default:0

let recover_strict (Comp (lhs, op, rhs) as comp) =
  let open Term.Infix in
  match op, const_coeff rhs with
  | GE, 1 -> Comp (lhs, GT, rhs - const 1)
  | LE, -1 -> Comp (lhs, LT, rhs + const 1)
  | _ -> comp

let prettify_comp ~meta c =
  prettify_comp_move ~meta c |> recover_strict

let prettify ?(meta=false) =
  Formula.apply_recursively ~f:(function
    | Formula.Comp _ as e ->
      comp_of_formula_exn e
      |> prettify_comp ~meta
      |> formula_of_comp
    | e -> e)

let%expect_test "prettify" =
  let ex ?(meta=false) s =
    Parse.formula s |> prettify ~meta
    |> [%show: Formula.t] |> Stdio.print_endline in
  ex "x < 5";
  ex "-x + y >= 3";
  ex "x + y + z >= 1";
  ex "x <= y - 1";
  ex "x + y + 3 + ?c != 0";
  ex "0 >= ?c - ?d";
  ex "n - ?c >= 0" ~meta:true;
  [%expect {|
    x <= 4
    x <= y - 3
    x + y + z > 0
    x < y
    x + y != -?c - 3
    0 <= -?c + ?d
    ?c <= n |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Abduction tactic (conjunctive form)                                        *)
(* ////////////////////////////////////////////////////////////////////////// *)

(* Abduction takes a series of clauses and returns a
   missing assumption that enables deriving false. *)

type abduction_result = Formula.t list list [@@deriving sexp, show]

let map_abduction_result ~f = List.map ~f:(List.map ~f)

let negated_final c = formula_of_comp (negate_comp c.comp)

let non_redundant_suggestion prevs fml =
  not (List.exists prevs ~f:(fun prev ->
    match comp_of_formula prev, comp_of_formula fml with
    | Some prev, Some fml -> implies prev fml
    | _ -> false))

let abduct_conjunctive ~settings clauses =
  let st = State.create () in
  assert_formulas st clauses;
  saturate ~settings:(settings.fm) st;
  if st.false_derived then []
  else begin
    let candidates =
      Queue.to_list (st.clauses) |> List.map ~f:negated_final in
    begin match settings.abduct_var_diff with
      | None -> [candidates]
      | Some fm ->
        let ge_cands =
          ge_candidates ~settings:fm st
          |> List.filter ~f:(non_redundant_suggestion candidates) in
        [candidates @ ge_cands]
    end
  end

let test_abduct_conjunctive =
  abduct_conjunctive ~settings:default_abduction_settings

let%expect_test "abduct_conjunctive" =
  let ex clauses =
    let clauses = List.map clauses ~f:Parse.formula in
    let res = test_abduct_conjunctive clauses in
    Stdio.print_endline ([%show: abduction_result] res) in
  ex ["x>=0"];
  ex ["x>=4"; "y<=-5"]; (* find_ge_cand *)
  ex ["x>=0"; "x+y<0"];
  ex ["x>=0"; "y>x"; "y<0"];
  ex ["x>=0"; "y<=0"];
  ex ["j<i"; "j>=i-3"; "2*j+i==41"];
  ex ["j<i"; "j>=i-3"; "2*j+i==41"; "j!=?c"];
  (* ?c1 ≤ 19 ⟶  i + 2*j = 41 ⟶  j ≥ i + ?c1 ⟶  j - 1 ≥ i + 2 + ?c1 *)
  ex ["?c<=19"; "i+2*j==41"; "j>=i"; "j-1<i+2+?c"];
  [%expect {|
    [[-x >= 1]]
    [[-x >= -3; y >= -4; y - x >= -8]]
    [[-x >= 1; x + y >= 0; y >= 0; y - x >= 0]]
    []
    [[-x >= 1; y >= 1; y - x >= 1]]
    [[j - i >= 0; 2*j + i != 41; -j + i >= 4; j != 13; i != 15]]
    [[j - ?c == 0; -j + i >= 4; j - i >= 0; 2*j + i != 41; -2*?c - i == -41;
       i != 15; j != 13; -?c == -13]
      ]
    [[?c >= 20; i + 2*j != 41; -j + i >= 1; 3*i >= 42; -3*j >= -40;
       j - i - ?c >= 3; -?c >= 3; 3*j - ?c >= 44; -3*i - 2*?c >= -36; 3*j >= 63;
       -3*i >= 2; -3*i - 3*?c >= -34; j - i >= 22; 3*j - 3*?c >= 48;
       3*i - ?c >= 44; -3*j - 2*?c >= -36; -3*i - 5*?c >= -30; 3*i - 3*?c >= 48;
       3*j - 4*?c >= 50; -3*j - 3*?c >= -34]
      ] |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Conjunctive Normal Forms                                                   *)
(* ////////////////////////////////////////////////////////////////////////// *)

type literal = bool * comparison  (* (negated, comp) *)

type cnf = literal list list

let negate_lit (negated, comp) = (not negated, comp)

let comp_of_lit (negated, comp) =
  if negated then negate_comp comp else comp

let rec negate_cnf = function
  | [] -> [[]]
  | disj::rest ->
    (* !((d1 | d2) & rest)  <->  (!d1 & !d2) | !rest *)
    let neg_rest = negate_cnf rest in
    List.concat_map (List.map disj ~f:negate_lit) ~f:(fun nd ->
      List.map neg_rest ~f:(fun r -> nd :: r))

let and_cnfs (args: cnf list) = List.concat args

(*  Note: using De Morgan's law to implement [or_cnfs] as follows
    would be TERRIBLY inefficient:

    let or_cnfs (args: cnf list) =
      negate_cnf (and_cnfs (List.map ~f:negate_cnf args))
*)

let binary_or_cnfs cnf cnf' =
  List.cartesian_product cnf cnf'
  |> List.map ~f:(fun (d, d') -> d @ d')

let or_cnfs = List.fold ~init:[[]] ~f:binary_or_cnfs

let implies_cnfs lhs rhs =
  or_cnfs [negate_cnf lhs; rhs]

let rec conjunctive_normal_form: Formula.t -> cnf =
  function
  | Bconst true -> []
  | Bconst false -> [[]]
  | Comp _ as c -> [[true, comp_of_formula_exn c]]
  | Unknown | Labeled _ -> raise Remaining_symbol_or_unknown
  | And args ->
    and_cnfs (List.map args ~f:conjunctive_normal_form)
  | Or args ->
    or_cnfs (List.map args ~f:conjunctive_normal_form)
  | Implies (lhs, rhs) ->
    let lhs = conjunctive_normal_form lhs in
    let rhs = conjunctive_normal_form rhs in
    implies_cnfs lhs rhs
  | Not arg -> conjunctive_normal_form arg |> negate_cnf

let clausify e =
  (* To prove [e], we have to find a contradiction for every
     set of clauses in [clausify e]. *)
  let cnf = conjunctive_normal_form e in
  List.map cnf ~f:(fun disj ->
    List.map disj ~f:(fun d -> comp_of_lit d |> formula_of_comp))

let%expect_test "clausify" =
  let ex s =
    Parse.formula s |> clausify
    |> [%show: Formula.t list list] |> Stdio.print_endline in
  ex "x > 1 -> x > 0";
  ex "x >= x && 0 <= 1";
  ex "x>=0 && y>=0 -> z>0 || z<5";
  ex "a1>0 || a2>0 -> c1>0 && c2>0 && c3>0";
  [%expect {|
    [[x > 1; -x >= 0]]
    [[-x >= -x + 1]; [0 >= 2]]
    [[x >= 0; y >= 0; -z >= 0; z >= 5]]
    [[a1 > 0; -c1 >= 0]; [a1 > 0; -c2 >= 0]; [a1 > 0; -c3 >= 0];
      [a2 > 0; -c1 >= 0]; [a2 > 0; -c2 >= 0]; [a2 > 0; -c3 >= 0]] |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Abduction tactic (general case)                                            *)
(* ////////////////////////////////////////////////////////////////////////// *)

(* TODO: Are there examples where we combine in a nontrivial way? *)

let combine_abduction_results res res' = res @ res'

let abduct ~settings fml =
  let fml = fml |> remove_unknowns_and_symbols |> simplify in
  let obligations = clausify fml in
  let res = List.map obligations ~f:(abduct_conjunctive ~settings) in
  List.fold res ~init:[] ~f:combine_abduction_results
  |> map_abduction_result ~f:prettify

let%expect_test "abduct" =
  let ex i s =
    let settings = default_abduction_settings in
    let res = Parse.formula s |> abduct ~settings in
    Fmt.pr "%d: %s\n" i ([%show: abduction_result] res) in
  ex 1 "true";
  ex 2 "false";
  ex 3 "x >= 0 & y >= 0";
  ex 4 "x >= 0 -> x + y >= 1";
  ex 5 "x >= ?c -> x >= ?d";
  ex 6 "i < n + 1 -> i >= n -> 3*n == x + y";
  ex 7 "x > #n & y >= 0 -> x + y >= ?c";
  (* Would easily lead to constant explosion *)
  ex 8 "3*x + 3*y >= 2 && y < z -> 2*y + x < z";
  [%expect{|
    1: []
    2: [[]]
    3: [[x >= 0]; [y >= 0]]
    4: [[x < 0; x + y > 0; y > 0; y > x]]
    5: [[x < ?c; x >= ?d; 0 >= -?c + ?d]]
    6: [[i != n; 3*n - x - y == 0; 3*i - x - y == 0]]
    7: [[y < 0; x <= #n; x + y >= ?c; x >= ?c; #n >= ?c - 1; #n + y >= ?c - 1]]
    8: [[3*x + 3*y <= 1; y >= z; 3*x + 3*z <= 4; 2*y + x - z < 0]] |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Sat and validity                                                           *)
(* ////////////////////////////////////////////////////////////////////////// *)

let surely_valid ?(settings=fm_default_settings) fml =
  let settings = {fm=settings; abduct_var_diff=None;} in
  let fml = remove_unknowns_and_symbols fml |> simplify in
  List.is_empty (abduct ~settings fml)

let possibly_sat ?(settings=fm_default_settings) fml =
  not (surely_valid ~settings (Not fml))

let%expect_test "sat_valid" =
  let ex fml =
    let fml = Parse.formula fml in
    let valid = surely_valid fml
    and sat = possibly_sat fml in
    assert (not valid || sat);
    let msg =
      if valid then "valid"
      else if sat then "sat"
      else "invalid" in
    Fmt.pr "%-30s  [%s] \n" ([%show: Formula.t] fml) msg in
  ex "x >= 1 -> x >= 0";
  ex "x == 0 && x == 1";
  ex "x != 0 && x + 2 >= 0";
  ex "0 < 2*x && 2*x < 2";
  ex "?c >= 1 -> ?c >= 0";
  ex "x>=0 & x!=0 -> x>=1";
  ex "x>=0 & x!=0 -> x>=2";
  ex "x >= #n || x <= #n";
  [%expect{|
    x >= 1 -> x >= 0                [valid]
    x == 0 && x == 1                [invalid]
    x != 0 && x + 2 >= 0            [sat]
    0 < 2*x && 2*x < 2              [invalid]
    ?c >= 1 -> ?c >= 0              [valid]
    x >= 0 && x != 0 -> x >= 1      [valid]
    x >= 0 && x != 0 -> x >= 2      [sat]
    x >= #n || x <= #n              [valid] |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Suggesting refinements                                                     *)
(* ////////////////////////////////////////////////////////////////////////// *)

let metavars_with_invertible_coeffs t =
  List.filter_map (Term.to_alist t) ~f:(function
    | (Var x, c) when (c = 1 || c = -1) && is_meta x -> Some (x, c = -1)
    | _ -> None)

let isolate_in_comp c =
  let Comp (lhs, op, _) = c |> normalize_compop |> move_all_left in
    metavars_with_invertible_coeffs lhs |> List.map ~f:(fun (x, negated) ->
    let lhs, op =
      if negated then Term.neg lhs, Compop.reverse op else lhs, op in
    let lhs, rhs_neg = Term.partition lhs ~f:(Term.equal_atom (Var x)) in
    assert (Term.equal lhs (Term.var x));
    (x, op, Term.neg rhs_neg))

let critical_values (op, e) =
  let open Term.Infix in
  match op with
  | Compop.NE -> [e + one; e - one]
  | _ -> [e]

let isolate_and_elim_metavar c =
  match isolate_in_comp c with
  | [] -> []
    (* It does not matter what var we eliminate *)
  | (x, op, e)::_ ->
      critical_values (op, e)
      |> List.map ~f:(fun c -> [(x, c)])

(* Particular case for constraints of the type 2*?c + 3*?d = 0 *)
let elim_two_metavars_at_once c =
  let (Comp (lhs, op, rhs)) = prettify_comp ~meta:true c in
  match Term.to_alist lhs, op, Term.to_alist rhs with
  | [(Var x, a); (Var y, b)], (GE | LE | EQ), [] when
    Int.abs a > 1 && Int.abs b > 1 &&
    is_meta x && is_meta y && not (equal_string x y) ->
    let open Term in
    let d = Util.Math.gcd a b in
    let a, b = a / d, b / d in
    [[(x, const b); (y, const (-a))]; [(x, const (-b)); (y, const a)]]
  | _ -> []

let elim_constraint c =
  match comp_of_formula c with
  | None -> [[]]
  | Some c ->
    isolate_and_elim_metavar c @ elim_two_metavars_at_once c

let%expect_test "elim_constraint" =
  let ex i fml =
    let refs = elim_constraint (Parse.formula fml) in
    Fmt.pr "%d: %s\n" i ([%show: (string * Term.t) list list] refs) in
  ex 1 "?c >= 2";
  ex 2 "?c + ?d >= 0";
  ex 3 "?c != n";
  ex 4 "2*?c + 3*?d == 0";
  [%expect {|
    1: [[("?c", 2)]]
    2: [[("?c", -?d)]]
    3: [[("?c", n + 1)]; [("?c", n - 1)]]
    4: [[("?c", 3); ("?d", -2)]; [("?c", -3); ("?d", 2)]] |}]

(* Find simple refinements *)

type bound_type = Lower_bound | Upper_bound | Equality [@@deriving eq]

let flip_bound = function
  | Lower_bound -> Upper_bound
  | Upper_bound -> Lower_bound
  | Equality -> Equality

let bound_type_of_compop = function
  | Compop.LE -> Some Upper_bound
  | Compop.GE -> Some Lower_bound
  | Compop.EQ -> Some Equality
  | _ -> None

let isolate_var_in_comp var c =
  List.find (isolate_in_comp c) ~f:(fun (x, _, _) -> equal_string x var)
  |> Option.map ~f:(fun (_, op, e) -> (op, e))

(* Return None if no bound was found. *)
let bound_type var fml =
  let open Option.Let_syntax in
  let%bind c = comp_of_formula fml in
  let%bind (op, _) = isolate_var_in_comp var c in
  bound_type_of_compop op

let update_constrs constrs (x, substituted) =
  List.map constrs ~f:(Formula.subst ~from:x ~substituted)
  |> List.filter ~f:(fun c -> not (surely_valid c))

(* Eliminate a var and return new constraints *)
let suggest_refinements var bound_type constrs =
  let open Option.Let_syntax in
  List.filter_map (saturate_constraints constrs) ~f:(fun f ->
    let%bind c = comp_of_formula f in
    let%bind (op, t) = isolate_var_in_comp var c in
    let%bind c_bound_type = bound_type_of_compop op in
    if Option.for_all bound_type ~f:(equal_bound_type c_bound_type)
  then Some (t, update_constrs constrs (var, t))
    else None)
  |> List.filter ~f:(fun (_, cs) -> possibly_sat (And cs))

let%expect_test "suggest_refinements" =
  let ex i var bound_type constrs =
    let constrs = List.map ~f:Parse.formula constrs in
    let res = suggest_refinements var bound_type constrs in
    Fmt.pr "%d: %s\n" i ([%show: (Term.t * Formula.t list) list] res) in
  ex 1 "?c" (Some Upper_bound) [];
  ex 2 "?c" (Some Upper_bound) ["0 <= ?c"; "?c <= 2"];
  ex 3 "?c" (Some Lower_bound) ["0 <= ?c"; "?c <= 2"];
  ex 4 "?c" (None) ["0 <= ?c"; "?c <= 2"];
  ex 5 "x"  (None) ["0 <= x"]; (* not a metavar *)
  ex 6 "?c" (Some Upper_bound) ["?c <= 1"; "?c <= 2"];
  ex 7 "?c" (Some Lower_bound) ["?c >= -2"; "?c >= 0"; "?c != 0"];
  ex 8 "?c" (Some Lower_bound) ["?c >= 0"; "?c != 0"];
  [%expect {|
    1: []
    2: [(2, [])]
    3: [(0, [])]
    4: [(0, []); (2, [])]
    5: []
    6: [(1, [])]
    7: [(1, [])]
    8: [(1, [])] |}]