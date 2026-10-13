open Base
open Prog
open Formula

let abduct_cost = 0.20
let conjecture_cost = 0.30
let max_abduction_candidates = 8
let max_conjecture_candidates = 6

(* ////////////////////////////////////////////////////////////////////////// *)
(* Events                                                                     *)
(* ////////////////////////////////////////////////////////////////////////// *)

module Event = struct

  type outcome =
    | SUCCESS
    | FAILURE
    | SIZE_LIMIT_EXCEEDED
    [@@deriving enum, show, sexp]

  type event =
    | ABDUCTION_EVENT
    | CONJECTURING_EVENT
    [@@deriving enum, show, sexp]

  let success = SUCCESS
  let default_failure = FAILURE
  let size_limit_exceeded = SIZE_LIMIT_EXCEEDED

  let outcome_reward = function
    | SUCCESS -> 1.
    | _ -> -1.

  let event_reward = function
    | ABDUCTION_EVENT -> -. abduct_cost
    | CONJECTURING_EVENT -> -. conjecture_cost

  let max_event_occurences _ = 4

  let min_success_reward = 0.

end

open Event

(* ////////////////////////////////////////////////////////////////////////// *)
(* Utilities for single loop programs                                         *)
(* ////////////////////////////////////////////////////////////////////////// *)

module Single_loop_prog = struct

  (* Convert a single loop program back and forth to a specific view. *)
  type t = {
    init: Prog.t;
    guard: Formula.t;
    invs: Prog.invariant list;
    body: Prog.t;
    post: Prog.t;
    final_assertion: Formula.t;
    final_assertion_status: proof_status option }
    [@@deriving sexp]

  let to_prog p =
    let loop = While (p.guard, p.invs, p.body) in
    let final = Assert (p.final_assertion, p.final_assertion_status) in
    Prog (prog_instrs p.init @ [loop] @ prog_instrs p.post @ [final])

  let split_post instrs =
    match List.rev instrs with
    | (Assert (fml, pf))::rev_rest -> fml, pf, Prog (List.rev rev_rest)
    | _ -> assert false

  let of_prog p =
    let rec aux = function
      | [] -> assert false
      | (While (guard, invs, body))::is ->
        let final_assertion, final_assertion_status, post = split_post is in
        { init=(Prog []); guard; invs; body; post;
          final_assertion; final_assertion_status }
      | i::is ->
        let rest = aux is in
        {rest with init = Prog (i :: prog_instrs rest.init)} in
    aux (prog_instrs p)

  let is_proved = function
    | Some Proved -> true
    | _ -> false

  let proved_invs p =
    List.filter_map p.invs ~f:(fun (inv, st) ->
      if is_proved st then Some inv else None)

end

open Single_loop_prog

(* ////////////////////////////////////////////////////////////////////////// *)
(* Forming conjectures                                                        *)
(* ////////////////////////////////////////////////////////////////////////// *)

(* f(x+y)=f(x+2,y+3) -> ax+by=a(x+2)+b(y+3) -> 2a+3b=0 *)
(* f(x1, x2, ...) = f(x1 + c1, x2 + c2, ...) *)
let find_coeffs vars fml =
  let cs = Queue.create () in
  Formula.iter_sub fml ~f:(function
    | Comp (lhs, _, _) ->
      begin match Term.to_alist lhs with
      | [Term.FunApp (_, args), _] ->
        begin try
          Queue.enqueue cs begin
          List.zip_exn vars args |>
          List.filter_map ~f:(fun (v, a) ->
            match Term.to_alist a with
            | [(Var x, 1); (One, c)] when equal_string x v -> Some (x, c)
            | _ -> None) end
        with _ -> () end
      | _ -> ()
      end
    | _ -> ());
  Queue.to_list cs
  |> List.dedup_and_sort ~compare:[%compare: (string * int) list]

let suggestion_from_coeffs cs =
  match cs with
  | [(x, a); (y, b)] ->
    let a, b = if b < 0 then -a, -b else a, b in
    let d = Util.Math.gcd a b in
    let a, b = a / d, b / d in
    Some (Term.(add (mulc b (var x)) (mulc (-a) (var y))))
  | _ -> None

let preserved_term p =
  let vars = Prog.modified_vars (to_prog p) |> Set.to_list in
  let f = Term.funapp "f" (List.map ~f:Term.var vars) in
  let pred = Formula.Infix.(f == Term.const 0) in
  let pre = Prog.wlp p.body pred in
  let coeffs = find_coeffs vars pre in
  List.filter_map coeffs ~f:suggestion_from_coeffs
  |> List.dedup_and_sort ~compare:Term.compare

let test_preserve pstr =
  Parse.program pstr |> of_prog |> preserved_term |>
  [%show: Term.t list] |> Stdio.print_endline

let preserved_term_conjectures fresh p =
  let open List.Let_syntax in
  let%bind lhs = preserved_term p in
  let%bind op = Compop.[GE; LE; EQ] in
  return (Comp (lhs, op, Term.var fresh))

let%expect_test "preserved_suggs" =
  test_preserve @@ {|
    while (x < #n) {
      z = 2;
      x = x + 2;
      y = y - 3;
    }
    assert x>=0; |};
  [%expect{| [3*x + 2*y] |}]

(* Generalized loop guard *)

let is_dyn_atom = function
  | Term.Var x -> Var.(has_kind Var x)
  | _ -> false

let generalize_comp fresh_meta = function
  | Comp (lhs, op, rhs) ->
    let rhs, _ = Term.partition rhs ~f:is_dyn_atom in
    let rhs = Term.(add rhs (Term.var fresh_meta)) in
    Some (Comp (lhs, op, rhs))
  | _ -> None

let generalized_loop_guard fresh_meta p =
  generalize_comp fresh_meta p.guard
  |> Option.to_list

(* Param constraints *)

let param_constraints p =
  let params = Prog_util.parameter_vars (to_prog p) in
  List.filter_map (prog_instrs p.init) ~f:(function
    | Assign (x, t) when Set.mem params x ->
      Some (Formula.Infix.(Term.var x == t))
    | Assume c when Set.is_subset (Formula.vars_set c) ~of_:params ->
      Some c
    | _ -> None)

(* Tests *)

let%expect_test "suggestions" =
  let p = {|
      n = 1;
      assume n > 0;
      while (i <= j + #m) {
        body;
      }
      assume #n > 0;
      assert i == #n;
  |} |> Parse.program |> of_prog in
  Stdio.print_endline ([%show: Formula.t list] (generalized_loop_guard "?c" p));
  Stdio.print_endline ([%show: Formula.t list] (param_constraints p));
  [%expect {|
    [i <= j + ?c]
    [n == 1; n > 0] |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Probes and proof actions                                                   *)
(* ////////////////////////////////////////////////////////////////////////// *)

type inv_disjunct_type =
  | Conjectured
  | Abducted
  | Kept
  | Strengthened
  [@@deriving show, sexp]

type proof_action =
  | Add_constr of {constr: Formula.t; closing: bool}
  | Prove_inv_candidate of {closing: bool}
  | Add_candidate_disjunct of
      {inv: Formula.t; ty: inv_disjunct_type; closing: bool}
  [@@deriving show, sexp]

let is_closing = function
  | Add_constr {closing; _}
  | Prove_inv_candidate {closing}
  | Add_candidate_disjunct {closing; _} -> closing

module Choice_summary = struct
  type t = proof_action [@@deriving sexp]

  let token_of_inv_disjunct_type = function
    | Conjectured -> Token.CONJECTURE_INV_DISJUNCT
    | Abducted -> ABDUCT_INV_DISJUNCT
    | Kept -> KEEP_INV_DISJUNCT
    | Strengthened -> STRENGHTEN_INV_DISJUNCT

  let string_of_inv_disjunct_type = function
    | Conjectured -> "conjecture"
    | Abducted -> "abduct"
    | Kept -> "keep"
    | Strengthened -> "strengthen"

  let to_graph a =
    let open Token in
    let open Token_graph in
    let flags = if is_closing a then [CLOSING_ACTION] else [] in
    match a with
    | Add_constr {constr; _} ->
      compose (tok ~flags ADD_CONSTR) [Tokenize.formula constr]
    | Prove_inv_candidate _ ->
      singleton (tok ~flags PROVE_INV_CANDIDATE)
    | Add_candidate_disjunct {inv; ty; _} ->
      let header = token_of_inv_disjunct_type ty in
      compose (tok ~flags header) [Tokenize.formula inv]

  let to_meta _ = []

  let closing_to_string = function
    | true -> "*"
    | false -> ""

  let to_string =
    let aux name closing fml =
      name ^ closing_to_string closing ^ " " ^ [%show: Formula.t] fml in
    function
    | Add_constr {constr; closing} ->
        aux "constrain" closing constr
    | Prove_inv_candidate {closing} ->
        "prove-inv-candidate" ^ closing_to_string closing
    | Add_candidate_disjunct {inv; ty; closing} ->
      aux (string_of_inv_disjunct_type ty) closing inv
end

type env =
  { prog: Single_loop_prog.t
  ; constrs: Formula.t list }
  [@@deriving sexp]

(* First conjecture or... and then we have add_disjunct *)

type probe_title =
  Prove_post | Prove_init | Prove_inductive [@@deriving show, sexp]
and probe_subtitle =
  Add_disjunct | Strengthen_added_disjunct [@@deriving show, sexp]

let caml_ctor_to_string s =
  String.drop_prefix s 7
  |> String.substr_replace_all ~pattern:"_" ~with_:"-"
  |> String.lowercase

type probe =
  { title: probe_title
  ; subtitle: probe_subtitle option
  ; env: env
  ; inv_candidate: Formula.t option
  ; obligation: Formula.t option }
  [@@deriving sexp]

let probe_header_to_string p =
  let title = p.title |> [%show: probe_title] |> caml_ctor_to_string in
  let s = "goal " ^ title in
  match p.subtitle with
  | None -> s
  | Some subtitle ->
    let subtitle =
      subtitle |> [%show: probe_subtitle] |> caml_ctor_to_string in
    s ^ " (" ^ subtitle ^ ")"

let probe_to_string p =
  let header = probe_header_to_string p in
  let prog = [%show: Prog.t] (to_prog p.env.prog) in
  let constrs =
    List.map p.env.constrs ~f:(fun f -> "constraint " ^ [%show: Formula.t] f)
    |> String.concat ~sep:"\n" in
  let elts =
    [header; prog] @ (if String.is_empty constrs then [] else [constrs]) in
  let elts =
    match p.inv_candidate with
    | None -> elts
    | Some f -> elts @ ["inv-candidate " ^ [%show: Formula.t] f ^ " || ..."] in
  String.concat ~sep:"\n\n" elts

module Probe = struct
  type t = probe [@@deriving sexp]

  let to_string = probe_to_string

  let token_of_probe_title = function
    | Prove_post -> Token.PROVE_POST
    | Prove_init -> PROVE_INIT
    | Prove_inductive -> PROVE_INDUCTIVE

  let token_of_probe_subtitle = function
    | None -> Token.NO_SUBTITLE
    | Some Add_disjunct -> ADD_DISJUNCT_SUBTITLE
    | Some Strengthen_added_disjunct -> STRENGTHEN_ADDED_DISJUNCT_SUBTITLE

  let to_graph p =
    let open Token in
    let open Token_graph in
    compose (tok (token_of_probe_title p.title)) [
      singleton (tok (token_of_probe_subtitle p.subtitle));
      compose (tok CONSTRAINTS) (List.map p.env.constrs ~f:Tokenize.formula);
      compose (tok INV_CANDIDATE) (
        Option.map p.inv_candidate ~f:Tokenize.formula |> Option.to_list);
      Tokenize.program (to_prog p.env.prog)]

  let to_meta p =
    begin match p.obligation with
      | Some f -> ["obligation", [%show: Formula.t] f]
      | None -> []
    end
end

(* ////////////////////////////////////////////////////////////////////////// *)
(* Action utilities                                                           *)
(* ////////////////////////////////////////////////////////////////////////// *)

let action_event = function
  | Add_constr _ -> None
  | Prove_inv_candidate _ -> None
  | Add_candidate_disjunct {ty; _} ->
    begin match ty with
    | Conjectured -> Some CONJECTURING_EVENT
    | Abducted -> Some ABDUCTION_EVENT
    | Kept -> None
    | Strengthened -> None
    end

(* ////////////////////////////////////////////////////////////////////////// *)
(* Environment utilities                                                      *)
(* ////////////////////////////////////////////////////////////////////////// *)

let used_names env =
  Set.union
    (Prog.vars_set (to_prog env.prog))
    (Set.union_list (module String)
      (List.map env.constrs ~f:Formula.vars_set))

let set_post_status st env =
  {env with prog = {env.prog with final_assertion_status=(Some st)}}

let add_inv inv st env =
  {env with prog = {env.prog with invs = env.prog.invs @ [inv, Some st]}}

(* Fresh metavariables *)

let fresh_meta env =
  let meta = Var.(make Meta_var) in
  let preferred = List.map ~f:meta ["c"; "d"; "a"; "b"] in
  let backup_prefix = meta "k" in
  let used = used_names env in
  Util.Fresh.fresh_id ~preferred ~backup_prefix ~used ()

let newly_introduced_meta env fml =
  Formula.vars_set fml
  |> Set.filter ~f:Var.(has_kind Meta_var)
  |> fun m -> Set.diff m (used_names env)

(* ////////////////////////////////////////////////////////////////////////// *)
(* Search monad                                                               *)
(* ////////////////////////////////////////////////////////////////////////// *)

module Search = Search.Make (Probe) (Choice_summary) (Event)
open Search

(* Abduction utilities *)

type abduct_result =
  | Valid
  | Suggs of {choices: Formula.t list; closing: bool}

let smallest_candidates ~max ~show fmls =
  if List.length fmls <= max then fmls
  else
    let metric f = String.length (show f) in
    let compare f f' = Int.compare (metric f) (metric f') in
    List.take (List.sort ~compare fmls) max

let abduct fml =
  let settings = Arith.default_abduction_settings in
  match Arith.abduct ~settings fml with
  | [] -> Valid
  | choices::rest ->
    let closing = List.is_empty rest in
    let choices =
      smallest_candidates
        ~max:max_abduction_candidates ~show:[%show: Formula.t] choices in
    Suggs {choices; closing}

(* Search monad utilities *)

let probe ?obligation title env =
  let obligation = Option.map obligation ~f:(
    fun f -> Arith.remove_unknowns_and_symbols f |> Arith.simplify) in
  {title; subtitle=None; env; inv_candidate=None; obligation}

let choose probe items =
  let choices = List.map items ~f:(fun a ->
    let e = action_event a in
    Search.{item=(a, e); summary=a; weight=0.}) in
  let* selected, e = Search.choose ~probe choices in
  let* () =
    match e with
    | None -> return ()
    | Some e -> Search.event e in
  return selected

(* ////////////////////////////////////////////////////////////////////////// *)
(* Proof result                                                               *)
(* ////////////////////////////////////////////////////////////////////////// *)

type outcome =
  | Proved_inv of Formula.t
  | Constraint of Formula.t

module Proof_res = struct

  let partition res =
    let constrs = List.filter_map res ~f:(function
      | Constraint f -> Some f
      |  _ -> None) in
    let others = List.filter res ~f:(function
      | Constraint _ -> false
      | _ -> true) in
    constrs, others

  let join ~constrs others =
    List.map constrs ~f:(fun c -> Constraint c) @ others

  let subst ~from ~substituted =
    List.map ~f:(function
      | Proved_inv f -> Proved_inv (Formula.subst ~from ~substituted f)
      | Constraint f -> Constraint (Formula.subst ~from ~substituted f))

  let update_env res env =
    List.fold res ~init:env ~f:(fun env outcome ->
      match outcome with
      | Proved_inv inv -> add_inv inv Proved env
      | Constraint c -> {env with constrs = env.constrs @ [c]})

end

(* ////////////////////////////////////////////////////////////////////////// *)
(* Main strategy                                                              *)
(* ////////////////////////////////////////////////////////////////////////// *)

let suggestion_to_action ~param_vars ~constrs ~propose_invs ~closing sugg =
  let vars = Formula.vars_set sugg in
  let is_dyn_var v = Var.(has_kind Var v) && not (Set.mem param_vars v) in
  if propose_invs &&
     Set.exists vars ~f:Var.(fun v -> has_kind Var v || has_kind Param v)
  then Some (Add_candidate_disjunct {inv=sugg; closing; ty=Abducted})
  else if
    (* There is some metavar left but no dynvar and the
       constraint is neither incompatible nor redundant. *)
    Set.exists vars ~f:Var.(has_kind Meta_var) &&
    not (Set.exists vars ~f:is_dyn_var) &&
    not (Arith.surely_valid (mk_implies constrs sugg)) &&
    Arith.possibly_sat (And (constrs @ [sugg]))
  then Some (Add_constr {constr=Arith.prettify ~meta:true sugg; closing})
  else None

let rec instantiate_meta to_instantiate res =
  match to_instantiate with
  | [] -> return res
  | (var, bound_type)::to_instantiate ->
    let constrs, others = Proof_res.partition res in
    begin match Arith.suggest_refinements var bound_type constrs with
    | [(substituted, constrs)] ->
      let others = Proof_res.subst ~from:var ~substituted others in
      let res = Proof_res.join ~constrs others in
      instantiate_meta to_instantiate res
    | suggs -> fail (
        "No unique refinement suggestion:\n" ^
        [%show: (Term.t * Formula.t list) list] suggs)
    end


let maybe_strengthen probe inv closing =
  match inv with
  | Comp (lhs, Compop.NE, rhs) ->
    let mk ty inv = Add_candidate_disjunct {inv; ty; closing} in
    let choices =
      [ mk Kept inv
      ; mk Strengthened (Comp (lhs, Compop.GT, rhs))
      ; mk Strengthened (Comp (lhs, Compop.LT, rhs)) ] in
    let probe = {probe with subtitle = Some Strengthen_added_disjunct} in
    let* selected = choose probe choices in
    begin match selected with
    | Add_candidate_disjunct {inv; _} -> return inv
    | _ -> assert false
    end
  | _ -> return inv


let add_disjuncts_and_strengthen probe all_actions inv closing =
  let rec aux budget prev_disjs closing =
    if budget <= 0 then return (`Inv (Formula.mk_disj prev_disjs), closing)
    else
      let actions = List.filter_map all_actions ~f:(function
        | Add_candidate_disjunct {inv; ty=Abducted; closing=closing'}
          when not (List.mem ~equal:Formula.equal prev_disjs inv) ->
          Some (Add_candidate_disjunct {
            inv; ty=Abducted;
            closing = (closing && closing')})
        | _ -> None) in
      let actions = Prove_inv_candidate {closing} :: actions in
      let probe = {probe with
        subtitle = Some Add_disjunct;
        inv_candidate = Some (Formula.mk_disj prev_disjs)} in
      let* selected = choose probe actions in
      begin match selected with
          | Add_constr _ -> assert false
          | Prove_inv_candidate {closing} ->
            return (`Inv (Formula.mk_disj prev_disjs), closing)
          | Add_candidate_disjunct {inv; closing; _} ->
            let* inv = maybe_strengthen probe inv closing in
            aux (budget-1) (prev_disjs @ [inv]) closing
      end in
    let* inv = maybe_strengthen probe inv closing in
    aux 2 [inv] closing

let conjectures env =
    let fresh = fresh_meta env in
    let invs =
      generalized_loop_guard fresh env.prog @
      preserved_term_conjectures fresh env.prog @
      param_constraints env.prog in
    let invs =
      smallest_candidates
        ~max:max_conjecture_candidates ~show:[%show: Formula.t] invs in
    List.map invs ~f:(fun inv ->
      Add_candidate_disjunct {inv; ty=Conjectured; closing=false})

let select_missing ~allow_disjs:(_) probe actions =
  let* selected = choose probe actions in
  match selected with
  | Add_constr {constr; closing} -> return (`Constr constr, closing)
  | Prove_inv_candidate _ -> assert false
  | Add_candidate_disjunct {inv; closing; _} ->
    add_disjuncts_and_strengthen probe actions inv closing


let rec prove_obligation
  ~compute_obligation
  ~final_obligation ~propose_invs
  ~allow_disjs ~allow_conjectures
  ~title ~annotate_status =
  let open Proof_res in
  let rec self env =
    assert (propose_invs || not allow_conjectures);
    let obligation = mk_implies env.constrs (compute_obligation env.prog) in
    match abduct obligation with
    | Valid -> return []
    | Suggs {choices; closing} ->
      let param_vars = Prog_util.parameter_vars (to_prog env.prog) in
      let choices =
        List.filter_map choices ~f:(suggestion_to_action
          ~param_vars ~constrs:env.constrs ~propose_invs ~closing) in
      let choices =
        if allow_conjectures then choices @ conjectures env else choices in
      let probe = probe ~obligation title (annotate_status To_prove env) in
      let* missing, closing = select_missing ~allow_disjs probe choices in
      (* let* selected = strengthen_inv probe selected in *)
      let aux_env_status =
        if final_obligation && closing then Proved_conditionally
        else To_prove_later in
      let aux_env = annotate_status aux_env_status env in
      let* aux_res = prove_missing aux_env missing in
      let* res = self (update_env aux_res env) in
      return (aux_res @ res)
  and prove_missing env action =
    match action with
    | `Inv inv ->
      let introduced_meta =
        newly_introduced_meta env inv
        |> Set.to_list
        |> List.map ~f:(fun c -> (c, Arith.bound_type c inv)) in
      let* res = prove_inv env inv in
      instantiate_meta introduced_meta res
    | `Constr constr ->
      return [Constraint constr]
  in self

and prove_post env =
  let compute_obligation p =
    mk_implies
      (proved_invs p @ [Not p.guard])
      (Prog.wlp p.post p.final_assertion) in
  prove_obligation env
    ~final_obligation:true ~propose_invs:true
    ~allow_disjs:true ~allow_conjectures:true
    ~compute_obligation
    ~annotate_status:set_post_status
    ~title:Prove_post

and prove_inv_init env inv =
  let compute_obligation prog = Prog.wlp prog.init inv in
  prove_obligation env
    ~final_obligation:false ~propose_invs:false
    ~allow_disjs:false ~allow_conjectures:false
    ~compute_obligation
    ~annotate_status:(add_inv inv)
    ~title:Prove_init

and prove_inv_inductive env inv =
  let compute_obligation p =
    mk_implies (proved_invs p @ [env.prog.guard; inv]) (Prog.wlp p.body inv) in
  prove_obligation env
    ~final_obligation:true ~propose_invs:true
    ~allow_disjs:false ~allow_conjectures:true
    ~compute_obligation
    ~annotate_status:(add_inv inv)
    ~title:Prove_inductive

and prove_inv env inv =
  let* res_init = prove_inv_init env inv in
  let env_ind = Proof_res.update_env res_init env in
  let* res_ind = prove_inv_inductive env_ind inv in
  return (res_init @ res_ind @ [Proved_inv inv])

(* ////////////////////////////////////////////////////////////////////////// *)
(* Main driver                                                                *)
(* ////////////////////////////////////////////////////////////////////////// *)

let main prog =
  let env = {prog; constrs=[]} in
  let* res = prove_post env in
  let final_env = Proof_res.update_env res (set_post_status Proved env) in
  return (to_prog final_env.prog)

let init_solver p = Search.search_tree (main (of_prog p))