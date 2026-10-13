open Base

type proof_status =
  To_prove | To_prove_later | Proved | Proved_conditionally
  [@@deriving eq, compare, hash, sexp]

let is_proved = function
  | Some Proved -> true
  | _ -> false

type t = Prog of instr list
  [@@deriving eq, compare, hash, sexp]
and instr =
  | LabeledProg of string * t option
  | Assume of Formula.t
  | Assert of Formula.t * proof_status option
  | Assign of string * Term.t
  | If of Formula.t * t * t
  | While of Formula.t * invariant list * t
  [@@deriving eq, compare, hash, sexp]
and invariant = Formula.t * proof_status option
  [@@deriving eq, compare, hash, sexp]

let is_empty_prog (Prog instrs) = List.is_empty instrs
let prog_instrs (Prog instrs) = instrs

(* ////////////////////////////////////////////////////////////////////////// *)
(* Mapping and traversing utilities                                           *)
(* ////////////////////////////////////////////////////////////////////////// *)

let map_children_formulas ~f = function
  | LabeledProg _ as e -> e
  | Assume c -> Assume (f c)
  | Assert (c, st) -> Assert (f c, st)
  | Assign _ as a -> a
  | If (cond, tb, fb) -> If (f cond, tb, fb)
  | While (cond, invs, body) ->
    let invs = List.map invs ~f:(fun (i, st) -> (f i, st)) in
    While (f cond, invs, body)

let map_children_programs ~f = function
  | LabeledProg (_, None) | Assume _ | Assert _ | Assign _ as i -> i
  | LabeledProg (s, Some p) -> LabeledProg (s, Some (f p))
  | If (cond, tb, fb) -> If (cond, f tb, f fb)
  | While (cond, invs, body) -> While (cond, invs, f body)

let rec apply_recursively ~f (Prog instrs) =
  Prog (List.map instrs ~f:(instr_apply_recursively ~f))
and instr_apply_recursively ~f instr =
  f (map_children_programs ~f:(apply_recursively ~f) instr)

let rec map_instrs ~f (Prog instrs) =
  Prog (List.concat_map instrs
    ~f:(fun i -> prog_instrs (instr_map_instrs ~f i)))
and instr_map_instrs ~f instr =
  f (map_children_programs ~f:(map_instrs ~f) instr)

let rec map_formulas ~f (Prog instrs) =
  Prog (List.map ~f:(instr_map_formulas ~f) instrs)
and instr_map_formulas ~f = function
  | LabeledProg (s, None) -> LabeledProg (s, None)
  | LabeledProg (s, Some p) -> LabeledProg (s, Some (map_formulas ~f p))
  | Assume c -> Assume (f c)
  | Assert (c, st) -> Assert (f c, st)
  | Assign _ as a -> a
  | If (cond, tb, fb) ->
      If (f cond, map_formulas ~f tb, map_formulas ~f fb)
  | While (cond, invs, body) ->
      let invs = List.map invs ~f:(invariant_map_formulas ~f) in
      While (f cond, invs, map_formulas ~f body)
and invariant_map_formulas ~f (inv, st) = (f inv, st)

let rec iter_instrs ~f (Prog instrs) =
  List.iter instrs ~f:(instr_iter_instrs ~f)
and instr_iter_instrs ~f instr =
  f instr;
  match instr with
    | LabeledProg (_, None) | Assume _ | Assert _ | Assign _ ->()
    | LabeledProg (_, Some p) -> iter_instrs ~f p
    | If (_, tb, fb) -> iter_instrs ~f tb; iter_instrs ~f fb
    | While (_, _, body) -> iter_instrs ~f body

(* ////////////////////////////////////////////////////////////////////////// *)
(* Querying Utilities                                                         *)
(* ////////////////////////////////////////////////////////////////////////// *)

let rec vars_set (Prog instrs) =
  Set.union_list (module String) (List.map ~f:instr_vars_set instrs)
and instr_vars_set = function
  | LabeledProg (_, None) -> Set.empty (module String)
  | LabeledProg (_, Some p) -> vars_set p
  | Assume cond | Assert (cond, _) -> Formula.vars_set cond
  | Assign (x, e) -> Set.add (Term.vars_set e) x
  | If (cond, tb, fb) ->
      Set.union
        (Formula.vars_set cond)
        (Set.union (vars_set tb) (vars_set fb))
  | While (cond, invs, body) ->
      let inv_vars =
        List.map invs ~f:(fun (i, _) -> Formula.vars_set i) in
      Set.union
        (Formula.vars_set cond)
        (Set.union
          (vars_set body)
          (Set.union_list (module String) inv_vars))

let rec pred_symbols (Prog instrs) =
  Set.union_list (module String) (List.map ~f:instr_pred_symbols instrs)
and instr_pred_symbols = function
  | LabeledProg (_, None) -> Set.empty (module String)
  | LabeledProg (_, Some p) -> pred_symbols p
  | Assign _ -> Set.empty (module String)
  | Assume cond | Assert (cond, _) -> Formula.pred_symbols cond
  | If (cond, tb, fb) ->
      Set.union
        (Formula.pred_symbols cond)
        (Set.union (pred_symbols tb) (pred_symbols fb))
  | While (cond, invs, body) ->
     let inv_symbs =
        List.map invs ~f:(fun (i, _) -> Formula.pred_symbols i) in
      Set.union
        (Formula.pred_symbols cond)
        (Set.union
          (pred_symbols body)
          (Set.union_list (module String) inv_symbs))

let rec prog_symbols (Prog instrs) =
  Set.union_list (module String) (List.map ~f:instr_prog_symbols instrs)
and instr_prog_symbols = function
  | LabeledProg (s, None) -> Set.singleton (module String) s
  | LabeledProg (s, Some p) -> Set.add (prog_symbols p) s
  | Assign _ | Assume _ | Assert _ -> Set.empty (module String)
  | If (_, tb, fb) -> Set.union (prog_symbols tb) (prog_symbols fb)
  | While (_, _, body) -> prog_symbols body

let subst ~from ~substituted =
  apply_recursively ~f:(
    function
    | Assign (x, t) -> Assign (x, Term.subst ~from ~substituted t)
    | instr ->
      map_children_formulas instr ~f:(Formula.subst ~from ~substituted))

let subst_multi ~f =
  apply_recursively ~f:(
    function
    | Assign (x, t) -> Assign (x, Term.subst_multi ~f t)
    | instr -> map_children_formulas instr ~f:(Formula.subst_multi ~f))

let rename_var ~from ~renamed =
  let substituted = Term.var renamed in
  apply_recursively ~f:(
    function
    | Assign (x, t) ->
      let x = if equal_string x from then renamed else x in
      Assign (x, Term.subst ~from ~substituted t)
    | instr -> map_children_formulas instr
        ~f:(Formula.subst ~from ~substituted))

let subst_prog_symbol ~from ~substituted =
  map_instrs ~f:(
    function
    | LabeledProg (l, _) when equal_string l from -> substituted
    | instr -> Prog [instr])

let subst_pred_symbol ~from ~substituted =
  map_formulas ~f:(Formula.subst_pred_symbol ~from ~substituted)

let modified_vars p =
  let vars = ref (Set.empty (module String)) in
  iter_instrs p ~f:(function
    | Assign (x, _) -> vars := Set.add !vars x
    | _ -> ());
  !vars

(* ////////////////////////////////////////////////////////////////////////// *)
(* Pretty printing                                                            *)
(* ////////////////////////////////////////////////////////////////////////// *)

let pp_proof_status_and_semicolon f = function
  | None -> Fmt.pf f ";"
  | Some To_prove -> Fmt.pf f " 'to-prove';"
  | Some To_prove_later -> Fmt.pf f " 'prove-later';"
  | Some Proved_conditionally -> Fmt.pf f " 'proved...';"
  | Some Proved -> Fmt.pf f " 'proved';"

let rec pp_instr f =
  function
  | LabeledProg (s, None) -> Fmt.pf f "%s;" s
  | LabeledProg (s, Some p) ->
    let n = List.length (prog_instrs p) in
    if n = 0 then Fmt.pf f "%s: { }" s
    else if n < 2 then Fmt.pf f "%s: %a" s pp p
    else Fmt.pf f "@[<v>%s: {@;<1 4>@[<v>%a@]@;}@]" s pp p
  | Assume c -> Fmt.pf f "assume %a;" Formula.pp c
  | Assert (c, st) ->
      Fmt.pf f "assert %a" Formula.pp c;
      pp_proof_status_and_semicolon f st
  | Assign (x, e) -> Fmt.pf f "%s = %a;" x Term.pp e
  | If (cond, tb, fb) ->
      begin
        Fmt.pf f "@[<v>if (%a) {@;<1 4>@[<v>%a@]@;}"
          Formula.pp cond pp tb;
        if not (is_empty_prog fb) then
          Fmt.pf f " else {@;<1 4>@[<v>%a@]@;}" pp fb;
        Fmt.pf f "@]"
      end
    | While (cond, invs, Prog body_instrs) ->
        Fmt.pf f "@[<v>while (%a) {@;<1 4>@[<v>%a@]@;} @]"
          Formula.pp cond
          (Fmt.list ~sep:Fmt.cut pp_loop_body_elt)
          (List.map ~f:Either.first invs @
           List.map ~f:Either.second body_instrs)

and pp_loop_body_elt f = function
    | Either.First (inv, st) ->
        Fmt.pf f "invariant %a" Formula.pp inv;
        pp_proof_status_and_semicolon f st
    | Either.Second instr -> pp_instr f instr

and pp f (Prog instrs) =
  Fmt.(vbox (list ~sep:cut pp_instr)) f instrs

(* ////////////////////////////////////////////////////////////////////////// *)
(* Proof obligations                                                          *)
(* ////////////////////////////////////////////////////////////////////////// *)

exception Remaining_prog_symbol

let rec wlp (Prog instrs) post =
  List.fold_right instrs ~init:post ~f:instr_wlp
and instr_wlp instr post =
  let open Formula in
  match instr with
  | Assume cond -> Implies (cond, post)
  | Assert (cond, st) ->
    if is_proved st then Implies (cond, post) else post
  | Assign (x, t) ->
    Formula.subst ~from:x ~substituted:t post
  | If (cond, tb, fb) ->
    And [Implies (cond, wlp tb post); Implies (Not cond, wlp fb post)]
  | LabeledProg (_, None) -> raise Remaining_prog_symbol
  | LabeledProg (_, Some p) -> wlp p post
  | While (guard, invs, _) ->
    let assums = List.filter_map invs ~f:(fun (i, st) ->
      if is_proved st then Some i else None) in
    let assums = (Not guard) :: assums in
    Implies (And assums, post)