(* ////////////////////////////////////////////////////////////////////////// *)
(* Teacher v2: untangling correctness and diversity                           *)
(* ////////////////////////////////////////////////////////////////////////// *)

open Base
open Prog
open Formula
open Util.Monads
open Lens.Infix
open Formula_util
open Prog_util

(* ////////////////////////////////////////////////////////////////////////// *)
(* Global parameters                                                          *)
(* ////////////////////////////////////////////////////////////////////////// *)

let max_num_params = 2
let max_num_assums = 2
let max_num_additional_init_statements = 2
let num_available_random_pos_consts = 2

let debug_mode = true
let enable_difficulty_control = true
let no_param_reuse_outside_abduction = true

let soft_constraint_violation_cost = 0.5
let minor_violation_cost = 0.2
let min_success_reward = -0.5
let max_abduction_candidates = 6
let max_abducted_terms_candidates = 4

(* ////////////////////////////////////////////////////////////////////////// *)
(* Problem specification                                                      *)
(* ////////////////////////////////////////////////////////////////////////// *)

type problem_spec = {
  num_preserved_term_vars:     [`Two | `Three] option;
  num_main_inv_disjuncts:      [`One | `Two]   option;
  num_aux_inv_conjuncts:       [`One | `Two]   option;
  disjunctive_post:            bool;
  body_structure:              body_structure;
  body_implies_main_inv:       bool;
  loop_guard_useful_for_inv:   bool;
  loop_guard_useful_for_post:  bool;
  available_consts:            int list;
  use_params:                  bool;
  require_param_assums:        bool;
  equalities_only_for_init:    bool;
  loop_guard_template:         [`Ltc | `Lec | `Gtc | `Gec | `Lev | `Nec | `Nev];
  allow_vcomp_in_prim_inv:     bool;
  assignment_templates:        [`Only_const_incr | `No_var_const_assign | `All]}
  [@@deriving sexp]

and body_structure =
  | No_cond | Cond of {cond_guard: bool; else_branch: bool; single_instr: bool}
  [@@deriving sexp]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Structured programs                                                        *)
(* ////////////////////////////////////////////////////////////////////////// *)

type problem = {
  param_assums: Formula.t list;
  init: Formula.t list;
  post: Formula.t list;
  loop_guard: Formula.t;
  preserved: Formula.t option;
  inv_main: (Formula.t list) option;
  inv_aux: Formula.t list;
  body_common: Prog.t;
  body_cond: body_cond option;
  extra_before_loop: Prog.t;
  extra_after_loop: Prog.t }
  [@@deriving sexp]

and body_cond = {
  cond_guard: Formula.t;
  cond_at_start: bool;
  tbranch: Prog.t;
  fbranch: Prog.t }
  [@@deriving sexp]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Events                                                                     *)
(* ////////////////////////////////////////////////////////////////////////// *)

module Event = struct

  type outcome =
    | SUCCESS
    | FAILURE
    | SIZE_LIMIT_EXCEEDED
    | LOOP_DOES_NOT_TERMINATE
    | LOOP_NEVER_ENTERED
    | INVARIANT_USELESS
    | INVARIANT_UNSAT
    | FAILED_TO_PROVE_INIT
    | FAILED_TO_PROVE_INV_PRESERVED
    | FAILED_TO_PROVE_POST
    [@@deriving enum, show, sexp]

  type event =
    | NO_PARAM_ASSUMS
    | NO_PARAM_USED
    | AUX_INV_IRRELEVANT
    | COND_GUARD_IRRELEVANT
    | LOOP_GUARD_IRRELEVANT_IN_PROVING_INV
    | LOOP_GUARD_IRRELEVANT_IN_PROVING_POST
    | PRESERVED_TERM_NOT_USEFUL
    | MAIN_INV_NOT_USEFUL
    | USELESS_POST_DISJUNCTS
    | USELESS_INVARIANT_DISJUNCTS
    | USELESS_INIT_CONJUNCTS
    [@@deriving enum, show, sexp]

  let success = SUCCESS
  let default_failure = FAILURE
  let size_limit_exceeded = SIZE_LIMIT_EXCEEDED

  let outcome_reward = function
    | SUCCESS -> 1.
    | _ -> -1.

  let event_reward = function
    | NO_PARAM_ASSUMS -> -. minor_violation_cost
    | _ -> -. soft_constraint_violation_cost

  let max_event_occurences _ = 1

  let min_success_reward = min_success_reward

end

open Event

(* ////////////////////////////////////////////////////////////////////////// *)
(* Specification utilities                                                    *)
(* ////////////////////////////////////////////////////////////////////////// *)

let max_num_vars spec =
  match spec.num_preserved_term_vars with
  | Some `Three -> 3
  | _ -> 2

let has_invariant spec =
  Option.is_some spec.num_preserved_term_vars ||
  Option.is_some spec.num_main_inv_disjuncts

let has_loop_guard spec =
  spec.loop_guard_useful_for_inv || spec.loop_guard_useful_for_post

let has_cond_guard spec =
  match spec.body_structure with
  | Cond {cond_guard=true; _} -> true
  | _ -> false

let has_single_instr_cond spec =
  match spec.body_structure with
  | Cond {single_instr=true; _} -> true
  | _ -> false

let has_disjunctive_inv spec =
  match spec.num_main_inv_disjuncts with
  | Some `Two -> true | Some `One | None -> false

let has_if_else spec =
  match spec.body_structure with
  | Cond {else_branch;_} -> else_branch
  | _ -> false

let has_single_atomic_inv_relevant_to_post spec =
  match spec.num_main_inv_disjuncts with
  | None -> Option.is_some spec.num_preserved_term_vars
  | Some `One -> Option.is_none spec.num_preserved_term_vars
  | Some `Two -> false

let nontrivial_available_consts spec =
  List.filter spec.available_consts ~f:(fun i -> Int.abs i > 1)

(* ////////////////////////////////////////////////////////////////////////// *)
(* Problem manipulation utilities                                             *)
(* ////////////////////////////////////////////////////////////////////////// *)

let hole_name = "..."
let fml_hole = Labeled (hole_name, None)
let prog_hole = Prog [LabeledProg (hole_name, None)]

let problem_body p =
  let cond_at_start =
    match p.body_cond with Some c -> c.cond_at_start | _ -> true in
  let cond =
    match p.body_cond with
    | None -> []
    | Some {cond_guard; tbranch; fbranch; _} ->
      [If (cond_guard, tbranch, fbranch)] in
  let common = prog_instrs p.body_common in
  Prog (if cond_at_start then cond @ common else common @ cond)

let problem_post_invs p =
  (Option.to_list p.preserved) @
  (match p.inv_main with Some ds -> [mk_disj ds] | None -> [])

let problem_invs p =
  problem_post_invs p @
  (if List.is_empty p.inv_aux then [] else [mk_conj p.inv_aux])

let problem_program p =
  let pre =
    (List.map p.param_assums ~f:(fun f -> Assume f)) @
    (List.map p.init ~f:(fun f -> Assume f)) in
  let invs = problem_invs p in
  let post = [Assert (mk_disj p.post, None)] in
  let loop = While (
    p.loop_guard, List.map invs ~f:(fun i -> i, None), problem_body p) in
  let pre_extra = prog_instrs p.extra_before_loop in
  let post_extra = prog_instrs p.extra_after_loop in
  Prog (pre @ pre_extra @ [loop] @ post_extra @ post)

let map_children_formula ~f p = {
  p with
  param_assums = List.map ~f p.param_assums;
  init = List.map ~f p.init;
  post = List.map ~f p.post;
  loop_guard = f p.loop_guard;
  preserved = Option.map ~f p.preserved;
  inv_main = Option.map ~f:(List.map ~f) p.inv_main;
  inv_aux = List.map ~f p.inv_aux;
  body_common = p.body_common;
  body_cond = Option.map p.body_cond ~f:(fun bc ->
    {bc with cond_guard = f bc.cond_guard})}

let map_children_progs ~f p = {
  p with
  body_common = f p.body_common;
  extra_before_loop = f p.extra_before_loop;
  extra_after_loop = f p.extra_after_loop;
  body_cond = Option.map p.body_cond ~f:(fun bc -> {
    bc with
    tbranch = f bc.tbranch;
    fbranch = f bc.fbranch })}

let subst ~from ~substituted p =
  map_children_formula p ~f:(Formula.subst ~from ~substituted)
  |> map_children_progs ~f:(Prog.subst ~from ~substituted)

let rename_var ~from ~renamed p =
  let substituted = Term.var renamed in
  map_children_formula p ~f:(Formula.subst ~from ~substituted)
  |> map_children_progs ~f:(Prog.rename_var ~from ~renamed)

(* ////////////////////////////////////////////////////////////////////////// *)
(* State definition and associated lenses                                     *)
(* ////////////////////////////////////////////////////////////////////////// *)

type cconstr = Abduct_tight | One_of_or_abducted of int list

type stage = After_guard | After_inv | After_body | After_post | After_init
  [@@deriving enumerate]

module State = struct
  type t = {
    spec: problem_spec option;
    problem: problem;
    cconstrs: cconstr Map.M(String).t;
    stage: stage option;
    rng: Random.State.t }
end

(* ////////////////////////////////////////////////////////////////////////// *)
(* Lenses                                                                     *)
(* ////////////////////////////////////////////////////////////////////////// *)

module Lenses = struct

  (* State specific *)
  open State
  let problem = Lens.{
    get = (fun st -> st.problem);
    set = (fun problem st -> {st with problem})}
  let rng = Lens.{
    get = (fun st -> st.rng);
    set = (fun rng st -> {st with rng})}
  let spec = Lens.{
    get = (fun st -> st.spec);
    set = (fun spec st -> {st with spec})}
  let cconstrs = Lens.{
    get = (fun st -> st.cconstrs);
    set = (fun cconstrs st -> {st with cconstrs})}
  let stage = Lens.{
    get = (fun st -> st.stage);
    set = (fun stage st -> {st with stage})}

  (* Problem specific *)
  let param_assums = Lens.{
    get = (fun p -> p.param_assums);
    set = (fun param_assums p -> {p with param_assums})}
  let init = Lens.{
    get = (fun p -> p.init);
    set = (fun init p -> {p with init})}
  let post = Lens.{
    get = (fun p -> p.post);
    set = (fun post p -> {p with post})}
  let loop_guard = Lens.{
    get = (fun p -> p.loop_guard);
    set = (fun loop_guard p -> {p with loop_guard})}
  let preserved = Lens.{
    get = (fun p -> p.preserved);
    set = (fun preserved p -> {p with preserved})}
  let inv_main = Lens.{
    get = (fun p -> p.inv_main);
    set = (fun inv_main p -> {p with inv_main})}
  let inv_aux = Lens.{
    get = (fun p -> p.inv_aux);
    set = (fun inv_aux p -> {p with inv_aux})}
  let body_common = Lens.{
    get = (fun p -> p.body_common);
    set = (fun body_common p -> {p with body_common})}
  let body_cond = Lens.{
    get = (fun p -> p.body_cond);
    set = (fun body_cond p -> {p with body_cond})}
  let extra_before_loop = Lens.{
    get = (fun p -> p.extra_before_loop);
    set = (fun extra_before_loop p -> {p with extra_before_loop})}
  let extra_after_loop = Lens.{
    get = (fun p -> p.extra_after_loop);
    set = (fun extra_after_loop p -> {p with extra_after_loop})}
  let cond_at_start = Lens.{
    get = (fun bc -> bc.cond_at_start);
    set = (fun cond_at_start bc -> {bc with cond_at_start})}
  let cond_guard = Lens.{
    get = (fun bc -> bc.cond_guard);
    set = (fun cond_guard bc -> {bc with cond_guard})}
  let tbranch = Lens.{
    get = (fun bc -> bc.tbranch);
    set = (fun tbranch bc -> {bc with tbranch})}
  let fbranch = Lens.{
    get = (fun bc -> bc.fbranch);
    set = (fun fbranch bc -> {bc with fbranch})}

  (* Utilities *)
  let append lens v = Lens.modify lens (fun l -> l @ [v])
  let set lens = lens.Lens.set
  let get lens = lens.Lens.get
  let append_prog lens p' = Lens.modify lens (fun p ->
    Prog (prog_instrs p @ prog_instrs p'))
  let opt = Lens.option_get

end

(* ////////////////////////////////////////////////////////////////////////// *)
(* Probes and choices definitions                                             *)
(* ////////////////////////////////////////////////////////////////////////// *)

type const_sugg_type = Guessed | Abducted [@@deriving sexp]

type _ probe_type =
  | Sample_spec: string -> string probe_type
  | Select_var: string -> string probe_type
  | Refine_const: string -> (Term.t * const_sugg_type) probe_type
  | Refine_formula: Formula.t probe_type
  | Refine_prog: Prog.t probe_type

type probe = Probe: 'a probe_type * problem_spec option * Prog.t -> probe

type choice = Choice: 'a probe_type * 'a -> choice

(* ////////////////////////////////////////////////////////////////////////// *)
(* Probes and choices utilities                                               *)
(* ////////////////////////////////////////////////////////////////////////// *)

type some_probe_type = Some_probe_type: 'a probe_type -> some_probe_type

let sexp_of_probe_type (type a) (ty: a probe_type) =
  let open Sexp in
  let str = [%sexp_of: string] in
  match ty with
  | Sample_spec s -> List [Atom "sample-spec"; str s]
  | Select_var l -> List [Atom "select-var"; str l]
  | Refine_const l -> List [Atom "refine-const"; str l]
  | Refine_formula -> Atom "refine-formula"
  | Refine_prog -> Atom "refine-prog"

let sexp_of_probe (Probe (ty, spec, prog)) =
  let open Sexp in
  List [
    Atom "teacher-probe"; sexp_of_probe_type ty;
    [%sexp_of: problem_spec option] spec; [%sexp_of: Prog.t] prog]
    let some_probe_type_of_sexp ty_sexp =
      let open Sexp in
      let mk ty = Some_probe_type ty in
      let str = [%of_sexp: string] in
      match ty_sexp with
      | List [Atom "sample-spec"; l] -> mk (Sample_spec (str l))
      | List [Atom "select-var"; l] -> mk (Select_var (str l))
      | List [Atom "refine-const"; l] -> mk (Refine_const (str l))
      | Atom "refine-formula" -> mk Refine_formula
      | Atom "refine-prog" -> mk Refine_prog
      | _ -> failwith
        ("Invalid sexp for teacher probe type:\n\n" ^ to_string_hum ty_sexp)

let probe_of_sexp sexp =
  let open Sexp in
  let err msg = failwith (msg ^ ":\n\n" ^ to_string_hum sexp) in
  match sexp with
  | List [Atom "teacher-probe"; ty_sexp; spec_sexp; prog_sexp] ->
    let spec = [%of_sexp: problem_spec option] spec_sexp in
    let prog = [%of_sexp: Prog.t] prog_sexp in
    let (Some_probe_type ty) = some_probe_type_of_sexp ty_sexp in
    Probe (ty, spec, prog)
  | _ -> err "Invalid sexp for teacher probe"

let varint_to_string = function
  | `One -> "1" | `Two -> "2" | `Three -> "3"

let varint_option_to_string = function
  | None -> "none"
  | Some n -> varint_to_string n

module Probe = struct

  type t = probe [@@deriving sexp]

  let body_structure_to_string = function
    | No_cond -> "no-cond"
    | Cond {cond_guard; else_branch; single_instr} ->
      String.concat ~sep:" " @@
        ["cond"] @
        (if cond_guard then ["guard"] else []) @
        (if else_branch then ["else"] else []) @
        (if single_instr then ["single-instr"] else [])

  let show_assignment_templates = function
    | `Only_const_incr -> "only-constr-incr"
    | `No_var_const_assign -> "no-var-const-assign"
    | `All -> "all-templates"

  let problem_spec_to_string spec =
    let mk label show v = [label ^ " " ^ show v] in
    let mb label b = if b then [label] else [] in
    let mo label show v = match v with Some v -> mk label show v | _ -> [] in
    let show_int_list ns =
      String.concat ~sep:" " (List.map ~f:Int.to_string ns) in
    String.concat ~sep:"\n" @@
      mo "preserved-term" varint_to_string spec.num_preserved_term_vars @
      mo "main-inv" varint_to_string spec.num_main_inv_disjuncts @
      mb "disjunctive-post" spec.disjunctive_post @
      mb "body-implies-inv" spec.body_implies_main_inv @
      mo "use-aux-inv" varint_to_string spec.num_aux_inv_conjuncts @
      mk "body-structure" body_structure_to_string spec.body_structure @
      mb "loop-guard-useful-for-inv"
        spec.loop_guard_useful_for_inv @
      mb "loop-guard-useful-for-post" spec.loop_guard_useful_for_post @
      mb "use-params" spec.use_params @
      mb "allow-vcomp-in-prim-inv" spec.allow_vcomp_in_prim_inv @
      mk "assignment-templates"
        show_assignment_templates spec.assignment_templates @
      mk "available-consts" show_int_list (nontrivial_available_consts spec)

  let to_string (Probe (ty, spec, prog)) =
    let header =
      match sexp_of_probe_type ty with
      | Atom s -> s
      | List [Atom s; Atom l] -> s ^ " " ^ l
      | _ -> failwith "Invalid sexp for probe type." in
    match spec with
    | None -> header
    | Some spec ->
      let prog = [%show: Prog.t] prog in
      String.concat ~sep:"\n\n" [header; problem_spec_to_string spec; prog]

  let probe_type_to_tokens (type a) (ty: a probe_type) =
    let open Token in
    match ty with
    | Sample_spec _ ->
      tok PADDING, tok NO_PROBE_ARG  (* not supposed to be tokenized *)
    | Select_var l -> tok PROBE_SELECT_VAR, tok ~name:l VAR
    | Refine_const l -> tok PROBE_REFINE_CONST, tok ~name:l META_VAR
    | Refine_formula -> tok PROBE_REFINE_FORMULA, tok NO_PROBE_ARG
    | Refine_prog -> tok PROBE_REFINE_PROG, tok NO_PROBE_ARG

  let problem_spec_to_graph spec =
    let open Token_graph in
    let open Token in
    let stok t = singleton (tok t) in
    let mkb b t = if b then [stok t] else [] in
    let available_consts =
      let cs = nontrivial_available_consts spec in
      compose (tok AVAILABLE_CONSTS)
        (List.map cs ~f:(fun c -> Tokenize.term (Term.const c))) in
    compose (tok PROBLEM_SPEC) @@
      [available_consts] @
      begin match spec.num_preserved_term_vars with
      | None -> []
      | Some `Two -> [stok TWO_VARS_PRESERVED]
      | Some `Three -> [stok THREE_VARS_PRESERVED]
      end @
      begin match spec.num_main_inv_disjuncts with
      | None -> []
      | Some `One -> [stok SIMPLE_MAIN_INV]
      | Some `Two -> [stok DISJUNCTIVE_MAIN_INV]
      end @
      mkb spec.disjunctive_post DISJUNCTIVE_POST @
      mkb spec.body_implies_main_inv BODY_IMPLIES_MAIN_INV @
      begin match spec.num_aux_inv_conjuncts with
      | None -> []
      | Some `One -> [stok USE_AUX_INV]
      | Some `Two -> [stok USE_CONJUNCTIVE_AUX_INV]
      end @
      begin match spec.body_structure with
      | No_cond -> [stok NO_CONDITIONALS]
      | Cond {cond_guard; else_branch; single_instr} ->
        let t =
          if else_branch then stok IF_THEN_ELSE_CONDITIONAL
          else stok IF_THEN_CONDITIONAL in
        [t] @
          (if cond_guard then [stok USE_COND_GUARD] else []) @
          (if single_instr then [stok SINGLE_INSTR_CONDITIONAL] else [])
      end @
      mkb spec.loop_guard_useful_for_inv LOOP_GUARD_USEFUL_FOR_INV @
      mkb spec.loop_guard_useful_for_post LOOP_GUARD_USEFUL_FOR_POST @
      mkb spec.use_params USE_PARAMS @
      mkb spec.allow_vcomp_in_prim_inv ALLOW_VCOMP_IN_PRIM_INV @
      begin match spec.assignment_templates with
      | `Only_const_incr -> [stok ONLY_CONST_INCR]
      | `No_var_const_assign -> [stok NO_VAR_CONST_ASSIGN]
      | `All -> []
      end

  let to_graph (Probe (ty, spec, prog)) =
    let open Token_graph in
    let header, arg = probe_type_to_tokens ty in
    let spec =
      match spec with
      | Some spec -> problem_spec_to_graph spec
      | None -> singleton (Token.tok PADDING) in
    compose header [singleton arg; spec; Tokenize.program prog]

  let probe_focus (Probe (ty, _, _)) =
    match ty with
    | Select_var l | Refine_const l -> Some l
    | Refine_formula | Refine_prog -> Some hole_name
    | _ -> None

  let probe_spec (Probe (_, spec, _)) = spec

  let to_meta probe =
    begin match probe_focus probe with
    | None -> []
    | Some l -> ["focus", l]
    end @
    begin match probe_spec probe with
    | None -> []
    | Some spec -> [
      "spec", problem_spec_to_string spec;
      "spec_sexp", sexp_of_problem_spec spec |> Sexp.to_string_hum]
    end

end

let sexp_of_choice (Choice (ty, choice)) =
  let open Sexp in
  let ty_sexp = sexp_of_probe_type ty in
  let mk to_sexp a = List [ty_sexp; to_sexp a] in
  match ty with
  | Sample_spec _ -> mk [%sexp_of: string] choice
  | Select_var _ -> mk [%sexp_of: string] choice
  | Refine_const _ ->
      mk [%sexp_of: Term.t * const_sugg_type] choice
  | Refine_formula -> mk [%sexp_of: Formula.t] choice
  | Refine_prog -> mk [%sexp_of: Prog.t] choice

let choice_of_sexp sexp =
  let open Sexp in
  match sexp with
  | List [ty_sexp; choice] ->
    let (Some_probe_type ty) = some_probe_type_of_sexp ty_sexp in
    let mk of_sexp a = Choice (ty, of_sexp a) in
    begin match ty with
    | Sample_spec _ -> mk [%of_sexp: string] choice
    | Select_var _ -> mk [%of_sexp: string] choice
    | Refine_const _ ->
        mk [%of_sexp: Term.t * const_sugg_type] choice
    | Refine_formula -> mk [%of_sexp: Formula.t] choice
    | Refine_prog -> mk [%of_sexp: Prog.t] choice
    end
  | _ -> failwith "Invalid sexp for teacher choice."

module Choice_summary = struct

  type t = choice [@@deriving sexp]

  let to_string (Choice (ty, choice)): string =
    let with_flags ~flags s =
      s ^ if List.is_empty flags then ""
      else "  (" ^ String.concat ~sep:", " flags ^ ")" in
    match ty with
    | Sample_spec _ -> choice
    | Select_var _ -> choice
    | Refine_const _ ->
      let (t, sugg_type) = choice in
      let flags =
        (match sugg_type with Abducted -> ["abducted"] | Guessed -> []) in
      with_flags ~flags (Term.to_string t)
    | Refine_formula -> [%show: Formula.t] choice
    | Refine_prog -> [%show: Prog.t] choice

  let to_graph (Choice (ty, choice)) =
    let open Token in
    let open Token_graph in
    match ty with
    | Sample_spec _ -> singleton (tok PADDING)  (* should not be tensorized *)
    | Select_var _ -> singleton (tok ~name:choice VAR)
    | Refine_const _ ->
      let (t, sugg_type) = choice in
      let flags =
        (match sugg_type with Abducted -> [ABDUCTED] | Guessed -> []) in
      compose (tok ~flags PROBE_REFINE_CONST) [Tokenize.term t]
    | Refine_formula -> Tokenize.formula choice
    | Refine_prog -> Tokenize.program choice

  let to_meta _choice = []

end

(* ////////////////////////////////////////////////////////////////////////// *)
(* Weighted lists manipulation                                                *)
(* ////////////////////////////////////////////////////////////////////////// *)

(* The choice operator operates with weighted lists. Weighted lists are
   almost distributions, except that they can be empty. *)

type 'a wlist = ('a * float) list

let uniform = List.map ~f:(fun x -> (x, 1.))

let map_other = List.map ~f:(fun x -> `Other x)

let wnorm xs =
  let wtot = List.fold xs ~init:0. ~f:(fun acc (_, c) -> acc +. c) in
  if Float.(wtot <= 0.) then xs
  else List.map xs ~f:(fun (x, c) -> (x, c /. wtot))

let wscale c =
  List.map ~f:(fun (x, c') -> (x, c *. c'))

let wconcat xss =
  List.concat_map xss ~f:(fun (xs, c) -> wscale c (wnorm xs)) |> wnorm

let wmap xs ~f = List.map xs ~f:(fun (x, c) -> (f x, c))

(* ////////////////////////////////////////////////////////////////////////// *)
(* Teacher and monad utilities                                                *)
(* ////////////////////////////////////////////////////////////////////////// *)

module Search = Search.Make (Probe) (Choice_summary) (Event)

open Lenses
include StateT (State) (Search)

let sample ~show descr choices =
  let probe_type = Sample_spec descr in
  let choices_transformed = List.mapi choices ~f:(fun i (it, weight) ->
    let summary = Choice (probe_type, (show it)) in
    Search.{item=i; summary; weight}) in
  let* p = read problem in
  let* spec = read spec in
  let probe = Probe (probe_type, spec, problem_program p) in
  let* i = lift (Search.choose ~chance:true ~probe choices_transformed) in
  return (List.nth_exn choices i |> fst)

let choose_index:
  'a. ?failmsg:string -> 'a probe_type -> 'a wlist -> ('a * int) t =
  fun ?failmsg probe_type choices ->
  let choices = List.mapi choices ~f:(fun i (item, weight) ->
    let summary = Choice (probe_type, item) in
    Search.{item=(item, i); summary; weight}) in
  let* prob = read problem in
  let* spec = read spec in
  let probe = Probe (probe_type, spec, problem_program prob) in
  lift (Search.choose ?failmsg ~chance:false ~probe choices)

let choose ?failmsg probe_type choices =
  let* selected, _ = choose_index ?failmsg probe_type choices in
  return selected

let ensure ?failure pred msg  =
  lift (Search.ensure ?failure pred msg)

let prefer pred msg event =
  if pred then return ()
  else
    let* () = lift (Search.message msg) in
    lift (Search.event event)

let debug_msg msg =
  if not debug_mode then return ()
  else lift (Search.message msg)

(* ////////////////////////////////////////////////////////////////////////// *)
(* State manipulation utilities                                               *)
(* ////////////////////////////////////////////////////////////////////////// *)

let read_spec = map (read spec) ~f:(fun s ->
  Option.value_exn ~message:"The spec hasn't been written yet." s)

(* ////////////////////////////////////////////////////////////////////////// *)
(* Handling names and constraints                                             *)
(* ////////////////////////////////////////////////////////////////////////// *)

let set_constr name distr = modify cconstrs (Map.set ~key:name ~data:distr)

let get_constr name = map (read cconstrs) ~f:(fun d -> Map.find d name)

let used_var_names =
  let* p = read problem in
  return @@ Prog.vars_set (problem_program p)

let preferred_var_names kind =
  let mk s = Var.make kind s in
  match kind with
  | Var.Var -> [mk "x"; mk "y"; mk "z"], mk "v"
  | Param -> [mk "n"; mk "m"; mk "p"], mk "k"
  | Var_hole -> [mk "x"; mk "y"; mk "z"], mk "v"
  | Meta_var -> [], mk "c"

(* Must be different from all other names given *)
let preserved_term_universal_var_name = "k"

let fresh_var_names kind n =
  assert (n > 0);
  let* used = used_var_names in
  let* cconstrs = read cconstrs in
  let used =
    Set.union used (Map.keys cconstrs |> Set.of_list (module String)) in
  let preferred, backup_prefix = preferred_var_names kind in
  let rec aux used = function
    | 0 -> []
    | n ->
      let name = Util.Fresh.fresh_id ~preferred ~backup_prefix ~used () in
      name :: aux (Set.add used name) (n - 1) in
  return (aux used n)

let tup1_of_list = function [x] -> x | _ -> assert false
let tup2_of_list = function [x; y] -> x, y | _ -> assert false
let tup3_of_list = function [x; y; z] -> x, y, z | _ -> assert false

let fresh1 kind = map (fresh_var_names kind 1) ~f:tup1_of_list
let fresh2 kind = map (fresh_var_names kind 2) ~f:tup2_of_list
let fresh3 kind = map (fresh_var_names kind 3) ~f:tup3_of_list

(* ////////////////////////////////////////////////////////////////////////// *)
(* Instantiating constants and variables                                      *)
(* ////////////////////////////////////////////////////////////////////////// *)

let small_pos_constants_except_one = [2; 3]
let small_pos_constants = 1 :: small_pos_constants_except_one

let used_vars_set kind =
  let* all = used_var_names in
  return (Set.filter all ~f:(Var.has_kind kind))

let used_vars kind = map ~f:Set.to_list (used_vars_set kind)

let is_one_of_or_big_const cs t =
  match Term.get_const t with
  | None -> false
  | Some c -> c >= 2 || List.mem ~equal:Int.equal cs c

let instantiate_const ?(always_allow_params=false) ?(abducted=[]) const =
  let* spec = read_spec in
  let* constr = get_constr const in
  let available = spec.available_consts in
  let* used_params = used_vars Var.Param in
  let nparams = List.length used_params in
  let* fresh_param = fresh1 Var.Param in
  let abducted =
    (* The abduction suggestions should be compatible with the constraints *)
    match constr with
    | Some (One_of_or_abducted cands) ->
      (* TODO: this is not very principled *)
      List.filter abducted ~f:(is_one_of_or_big_const cands)
    | _ -> abducted in
  let choices =
    match constr with
    | Some Abduct_tight -> []
    | Some (One_of_or_abducted cands) -> uniform (List.map ~f:Term.const cands)
    | None ->
      let used_params =
        if no_param_reuse_outside_abduction then []
        else used_params in
      let param_instances =
        if not spec.use_params && not always_allow_params then []
        else if nparams >= max_num_params then used_params
        else used_params @ [fresh_param] in
      wconcat @@ uniform [
        uniform (List.map ~f:Term.const available);
        uniform (List.map ~f:Term.var param_instances)] in
  let choices =
    (* The basic suggestions should not be redundant with the abducted ones. *)
    List.filter choices ~f:(fun (t, _) ->
      not (List.mem ~equal:Term.equal abducted t)) in
  let choices =
    wconcat @@ uniform [
      wmap choices ~f:(fun c -> (c, Guessed));
      uniform (List.map abducted ~f:(fun c -> (c, Abducted)))] in
  let* substituted, _ =
    choose (Refine_const const) choices
      ~failmsg:("No valid choice for instantiating " ^ const) in
  modify problem (subst ~from:const ~substituted)

let instantiate_all_consts ?always_allow_params () =
  let* const_holes = used_vars Var.Meta_var in
  let* () =
    List.map const_holes ~f:(instantiate_const ?always_allow_params)
    |> sequence_unit in
  (* We take an opportunity to garbage collect constraints *)
  write cconstrs (Map.empty (module String))

let instantiate_dyn_var ~allow_fresh ~choices v =
  let* num_vars = map ~f:List.length (used_vars Var.Var) in
  let* max_num_vars = map ~f:max_num_vars read_spec in
  let allow_fresh = allow_fresh && num_vars < max_num_vars in
  let* fresh_dyn_var = fresh1 Var.Var in
  let choices = if allow_fresh then choices @ [fresh_dyn_var] else choices in
  let* renamed =
    choose (Select_var v) (uniform choices)
      ~failmsg:("No valid choice for instantiating variable " ^ v)in
  let* () = modify problem (rename_var ~from:v ~renamed) in
  return renamed

let instantiate_all_dyn_vars ?(blacklist=[]) ~allow_fresh ~distinct () =
  let* holes = used_vars Var.Var_hole in
  let rec aux prev = function
    | [] -> return ()
    | v::vs ->
      let* choices = used_vars Var.Var in
      let choices =
        if distinct then
          List.filter choices
            ~f:(fun c -> not (List.mem ~equal:equal_string prev c))
        else choices in
      let* selected = instantiate_dyn_var ~allow_fresh ~choices v in
      aux (selected::prev) vs
  in aux blacklist holes

(* ////////////////////////////////////////////////////////////////////////// *)
(* Filling in formula and program holes                                       *)
(* ////////////////////////////////////////////////////////////////////////// *)

(* [Unkonwn] is treated to mean "do not refine" *)
let refine_formula set_formula choices =
  let fml_choices = List.map choices ~f:(fun (i, _, w) -> (i, w)) in
  let templates = List.map choices ~f:(fun (_, t, _) -> t) in
  let* selected, index =
    with_modified problem (set_formula fml_hole) @@
    choose_index Refine_formula fml_choices in
  let template = List.nth_exn templates index in
  let* () =
    if Formula.equal selected Unknown then return ()
    else modify problem (set_formula selected) in
  return template

let append_instr prog_lens choices =
  let instr_choices = List.map choices ~f:(fun (i, _, w) -> (Prog [i], w)) in
  let templates = List.map choices ~f:(fun (_, t, _) -> t) in
  let* selected, index =
    with_modified problem (append_prog prog_lens prog_hole) @@
      choose_index Refine_prog instr_choices in
  let* () = modify problem (append_prog prog_lens selected) in
  return (List.nth_exn templates index)

(* ////////////////////////////////////////////////////////////////////////// *)
(* Generating formulas                                                        *)
(* ////////////////////////////////////////////////////////////////////////// *)

let refine_with_template' ?(allow_fresh=false) setter templates =
  let open Formula.Infix in
  let open Term in
  let* x, y = fresh2 Var.Var_hole in
  let* c = fresh1 Var.Meta_var in
  let x, y, c = var x, var y, var c in
  let instantiate = function
    | `Unknown -> Unknown
    | `Ltc -> x < c
    | `Gtc -> x > c
    | `Lec -> x <= c
    | `Gec -> x >= c
    | `Lev -> x <= y
    | `Ltv -> x < y
    | `Eqc -> x == c
    | `Eqv -> x == y
    | `Nec -> x <> c
    | `Nev -> x <> y
    | `Other f -> f in
  let choices = List.map templates ~f:(fun (t, w) -> instantiate t, t, w) in
  let* refined = refine_formula setter choices in
  let* () = instantiate_all_dyn_vars ~allow_fresh ~distinct:true () in
  return refined

let refine_with_template ?(allow_fresh=false) setter templates =
  let* _ = refine_with_template' ~allow_fresh setter templates in
  return ()

let relax_ineq_gen c = function
  | Comp (lhs, (GT | GE as op), rhs) ->
      Some (Comp (lhs, op, Term.(sub rhs c)))
  | Comp (lhs, (LT | LE as op), rhs) ->
      Some (Comp (lhs, op, Term.(add rhs c)))
  | _ -> None

let relax_ineq fml =
  let* c = fresh1 Var.Meta_var in
  let* () = set_constr c (One_of_or_abducted small_pos_constants) in
  return (relax_ineq_gen (Term.var c) fml)

(* ////////////////////////////////////////////////////////////////////////// *)
(* Generating programs                                                        *)
(* ////////////////////////////////////////////////////////////////////////// *)

let append_assignment ~templates ~allowed prog_lens =
  let* v, y = fresh2 Var.Var_hole in
  let* c, d = fresh2 Var.Meta_var in
  let* spec = read_spec in
  let* () =
    let choices = List.dedup_and_sort ~compare:Int.compare @@
      (small_pos_constants_except_one) @
      (List.filter spec.available_consts ~f:(fun c -> c > 1)) in
    set_constr d (One_of_or_abducted choices) in
  let instantiate =
    let open Term in
    let open Term.Infix in
    function
    | `Assign_const -> Assign (v, var c)
    | `Assign_var -> Assign (v, var y)
    | `Incr -> Assign (v, var v + one)
    | `Decr -> Assign (v, var v - one)
    | `Incr_const -> Assign (v, var v + var d)
    | `Decr_const -> Assign (v, var v - var d)
    | `Incr_var -> Assign (v, var v + var y)
    | `Assign_lin_neg_other -> Assign (v, var c - var y) in
  let choices =
    List.map templates ~f:(fun t -> instantiate t, t, 1.) in
  let* template = append_instr prog_lens choices in
  let* x =
    instantiate_dyn_var ~allow_fresh:false ~choices:allowed v in
  let* () = instantiate_all_dyn_vars
    ~allow_fresh:false ~distinct:true ~blacklist:[x] () in
  return (x, template)

(* Add assignments one by one until all variables are assigned or
   the user asks to stop *)
let add_assignments ~templates ~target_vars ~single prog_lens =
  let rec aux i target_vars =
    if Set.is_empty target_vars || (single && i > 0)
    then return (Set.empty (module String))
    else
      let allowed = target_vars |> Set.to_list in
      let* v, _ = append_assignment ~templates prog_lens ~allowed in
      let* rest = aux (i + 1) (Set.remove target_vars v) in
      return (Set.add rest v) in
  aux 0 target_vars

(* ////////////////////////////////////////////////////////////////////////// *)
(* Generating obligations                                                     *)
(* ////////////////////////////////////////////////////////////////////////// *)

let init_obligation _s p =
  let assums = mk_conj (p.param_assums @ p.init) in
  Implies (assums, Prog.wlp p.extra_before_loop (mk_conj (problem_invs p)))

let post_obligation ?(full=false) s p =
  let assums =
    (p.param_assums @ problem_post_invs p) @
    (if full || s.loop_guard_useful_for_post then [Not p.loop_guard] else []) in
  Implies (mk_conj assums, Prog.wlp p.extra_after_loop (mk_disj p.post))

let preserved_term_obligation _s p =
  (* We prove something stronger, which is that the LHS
     is preserved as a term. *)
  match p.preserved with
  | None -> Bconst true
  | Some (Comp (lhs, _, _)) ->
    let k = preserved_term_universal_var_name in
    let fml = Comp (lhs, Compop.EQ, Term.var k) in
    Implies (fml, Prog.wlp (problem_body p) fml)
  | Some _ -> assert false

let preserved_inv_aux_obligation _s p =
  let body = problem_body p in
  let inv = mk_conj p.inv_aux in
  Implies (mk_conj (p.param_assums @ [inv]), Prog.wlp body inv)

let preserved_inv_main_obligation ?(full=false) s p =
  match p.inv_main with
  | None -> Bconst true
  | Some inv_main_disjuncts ->
    let inv_main = mk_disj inv_main_disjuncts in
    let assums =
      (p.param_assums @ p.inv_aux) @
      (if full || s.loop_guard_useful_for_inv then [p.loop_guard] else []) in
    let body = problem_body p in
    let assumed_inv = if not s.body_implies_main_inv then [inv_main] else [] in
    Implies ((mk_conj (assums @ assumed_inv)), Prog.wlp body inv_main)

let preserved_inv_obligation s p =
  mk_conj [
    preserved_term_obligation s p; preserved_inv_aux_obligation s p;
    preserved_inv_main_obligation s p]

let all_obligations s p =
  mk_conj [
    preserved_inv_obligation s p;
    init_obligation s p;
    post_obligation s p ]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Sampling a problem specification                                           *)
(* ////////////////////////////////////////////////////////////////////////// *)

let sample_available_consts =
  let open Util.Random in
  let* rng = read rng in
  let ubound_distr = make_distr [10, 1.0; 100, 0.2; 1000, 0.05; 10_000, 0.05] in
  let ubound = sample ubound_distr rng in
  let available =
      List.init num_available_random_pos_consts ~f:(fun _ ->
        Random.State.int_incl rng 2 ubound) @
      List.init num_available_random_pos_consts ~f:(fun _ ->
        Random.State.int_incl rng (-ubound) (-2)) in
  let available = List.sort ~compare:Int.compare ([-1; 0; 1] @ available) in
  return available

let show_loop_guard_template = function
  | `Ltc -> "x < ..." | `Lec -> "x <= ..." | `Gtc -> "x > ..."
  | `Gec -> "x >= ..." | `Lev -> "x <= y"
  | `Nec -> "x != ..." | `Nev -> "x != y"

let sample_bool ~p descr =
  sample ~show:[%show: bool] descr [true, p; false, 1. -. p]

let sample_spec =
  let open Poly in
  let* num_preserved_term_vars =
    sample "num-preserved-term-vars" ~show:varint_option_to_string [
      None, 0.75;
      Some `Two, 0.22;
      Some `Three, 0.03] in
  let* num_main_inv_disjuncts =
    let p = if Option.is_some num_preserved_term_vars then 0.6 else 0.9 in
    sample "num-main-inv-disjuncts" ~show:varint_option_to_string [
      None, 1.0 -. p;
      Some `One, p *. 0.75;
      Some `Two, p *. 0.25] in
  let* num_aux_inv_conjuncts =
    let p = if Option.is_some num_main_inv_disjuncts then 0.1 else 0. in
    sample "num-aux-inv-conjuncts" ~show:varint_option_to_string [
      None, 1. -. p;
      Some `One, p /. 2.;
      Some `Two, p /. 2.] in
  let* disjunctive_post =
    sample_bool ~p:0.3 "disjunctive-post" in
  let* body_implies_main_inv =
    let p = if num_main_inv_disjuncts = Some `Two then 0.7 else 0. in
    sample_bool ~p "body-implies-inv" in
  let* body_structure =
    let p = if body_implies_main_inv then 0.5 else 0.2 in
    let* use_cond = sample_bool ~p "use-cond" in
    if not use_cond then return No_cond
    else
      let* else_branch = sample_bool ~p:0.3 "else-branch" in
      let* cond_guard =
        let p = if body_implies_main_inv then 0.8 else 0.3 in
        sample_bool ~p "cond-guard" in
      let* single_instr =
        let p = if Option.is_none num_preserved_term_vars then 0.5 else 0. in
        sample_bool ~p "single-instr" in
      return (Cond {cond_guard; else_branch; single_instr}) in
  let* loop_guard_useful_for_inv =
    let p = if Option.is_some num_main_inv_disjuncts then 0.5 else 0. in
    sample_bool ~p "loop-guard-useful-for-inv" in
  let* loop_guard_useful_for_post =
    sample_bool ~p:0.7 "loop-guard-useful-for-post" in
  let* use_params =
    sample_bool ~p:0.3 "use-params" in
  let* equalities_only_for_init =
    sample_bool ~p:0.5 "eqs-only-for-init" in
  let* require_param_assums =
    let p = if equalities_only_for_init then 0.3 else 0. in
    sample_bool ~p "require-param-assums" in
  let* loop_guard_template =
    sample "loop-guard-template" ~show:show_loop_guard_template @@
      [`Ltc, 2.; `Lec, 1.; `Gtc, 0.5; `Gec, 0.5; `Nec, 0.2; `Nev, 0.2] @
      (if num_preserved_term_vars = Some `Two then [`Lev, 2.] else []) in
  let* allow_vcomp_in_prim_inv =
    sample_bool ~p:0.2 "allow-vcomp-in-prim-inv" in
  let* assignment_templates =
    sample "assignment-templates" ~show:Probe.show_assignment_templates [
      `Only_const_incr, 1.;
      `No_var_const_assign, 1.;
      `All, if body_implies_main_inv then 5. else 1.] in
  let* available_consts =
    sample_available_consts in
  write spec @@ Some
    { num_preserved_term_vars; num_main_inv_disjuncts;
      num_aux_inv_conjuncts; disjunctive_post; body_implies_main_inv;
      body_structure; loop_guard_useful_for_inv;
      loop_guard_useful_for_post; use_params; require_param_assums;
      equalities_only_for_init; available_consts;
      loop_guard_template; allow_vcomp_in_prim_inv; assignment_templates }

let check_preserved_term_spec_difficulty =
  let* s = read_spec in
  if Option.is_some s.num_preserved_term_vars then
    let* () = write spec @@ Some {s with
      assignment_templates = `Only_const_incr;
      num_aux_inv_conjuncts = None} in
    if Option.is_some s.num_main_inv_disjuncts then
      let* s = read_spec in
      write spec @@ Some {s with
        loop_guard_useful_for_inv = true;
        loop_guard_useful_for_post = true}
    else
      let* s = read_spec in
      write spec @@ Some {s with
        loop_guard_useful_for_inv = false}
  else return ()

let check_spec_difficulty =
  let* spec = read_spec in
  let difficulty =
    List.count ~f:Fn.id [
      Option.is_some spec.num_preserved_term_vars;
      has_disjunctive_inv spec;
      spec.disjunctive_post;
      Option.is_some spec.num_aux_inv_conjuncts;
      has_if_else spec ] in
  let open Util.Random in
  let* rng = read rng in
  if enable_difficulty_control then
    ensure (difficulty <= 2 || (difficulty = 3 && bernouilli ~p:0.25 rng))
      "The problem is too difficult."
  else return ()

let control_spec_difficulty =
  let open Poly in
  let* s = read_spec in
  if s.num_aux_inv_conjuncts = Some `Two then
    write spec @@ Some {
      s with
      disjunctive_post = false;
      num_main_inv_disjuncts = Some `One;
      num_preserved_term_vars = None;
      body_implies_main_inv = false;
      loop_guard_useful_for_inv = false;
      loop_guard_useful_for_post = false;
      allow_vcomp_in_prim_inv = true;
      assignment_templates = `No_var_const_assign }
  else if s.num_preserved_term_vars = Some `Three then
    write spec @@ Some {
      s with disjunctive_post = false }
  else check_spec_difficulty

let check_spec_consistency =
  let* spec = read_spec in
  let* () = ensure
    (not (
      has_disjunctive_inv spec &&
      not spec.loop_guard_useful_for_post &&
      not spec.disjunctive_post))
    "The post obligation has shape I1|I2 -> P." in
  let* () = ensure
    (not (
      spec.loop_guard_useful_for_post &&
      spec.disjunctive_post &&
      has_single_atomic_inv_relevant_to_post spec))
    "The post obligation has shape !G -> I -> A -> B with I atomic." in
  let* () = ensure
    (has_invariant spec || spec.use_params)
    "Parameters must be used if no invariant is proposed." in
  let* () = ensure
    (Option.is_none spec.num_preserved_term_vars || not (has_cond_guard spec))
    "Conditional guards cannot be used with preserved terms." in
  let* () = ensure
    (Option.is_none spec.num_preserved_term_vars ||
      not (has_single_instr_cond spec))
    "Single instr conditionals cannot be used with preserved terms." in
  let* () = ensure
    begin match spec.num_preserved_term_vars, spec.body_structure with
    | Some `Three, Cond {else_branch=true;_}  -> false
    | _ -> true end
    "This specification would likely cause a probe overflow."
     in
  return ()

(* ////////////////////////////////////////////////////////////////////////// *)
(* Hard and soft constraints                                                  *)
(* ////////////////////////////////////////////////////////////////////////// *)

let composite_formula = function
  | Formula.Comp _ | Formula.Bconst _ -> false | _ -> true

let not_valid_unsat_or_redundant mk lens _spec problem =
  let elts = get lens problem in
  if List.length elts <= 1 then None
  else
    let fml = mk elts in
    let assums = get param_assums problem in
    let assums = mk_conj assums in
    let fml_s = [%show: Formula.t] fml in
    if Arith.surely_valid (Implies (assums, fml)) then
      Some ("Generated valid formula: " ^ fml_s)
    else if not (Arith.possibly_sat (And [assums; fml])) then
      Some ("Generated unsat formula: " ^ fml_s)
    else if is_redundant ~assuming:assums elts then
      Some ("Redundant formula: " ^ fml_s)
    else if
      composite_formula fml &&
      Option.is_some (simplify_to_atomic ~assuming:assums fml) then
      Some ("Simplifiable formula: " ^ fml_s)
    else None

let ensure_pred_msg pred =
  let* spec = read_spec in
  let* problem = read problem in
  match pred spec problem with
  | None -> return ()
  | Some msg -> ensure false msg

let ensure_not_valid_unsat_or_redundant mk lens =
  ensure_pred_msg (not_valid_unsat_or_redundant mk lens)

let ensure_invs_not_valid_unsat_or_redundant =
  let* spec = read_spec in
  let* () =
    match spec.num_main_inv_disjuncts with
    | None -> return ()
    | Some _ ->
      ensure_not_valid_unsat_or_redundant mk_disj (inv_main |-- opt) in
  let* () =
    ensure_not_valid_unsat_or_redundant mk_conj inv_aux in
  return ()

let post_not_valid_or_redundant spec problem =
  if has_invariant spec || List.is_empty problem.param_assums then
    not_valid_unsat_or_redundant mk_disj post spec problem
  else None

let ensure_post_not_valid_or_redundant =
  ensure_pred_msg post_not_valid_or_redundant

let loop_entered _spec p =
  let assums = mk_conj (p.param_assums @ p.init) in
  not (Arith.surely_valid (Implies (assums, Not p.loop_guard)))

let ensure_loop_entered =
  let* p = read problem in
  let* s = read_spec in
  ensure (loop_entered s p)
    "The loop guard is always false initially."
    ~failure:LOOP_NEVER_ENTERED

let loop_terminates _spec p =
  let assums = mk_conj (p.param_assums @ problem_invs p @ [p.loop_guard]) in
  let concl = Prog.wlp (problem_body p) p.loop_guard in
  not (Arith.surely_valid (Implies (assums, concl)))

let ensure_loop_terminates =
  let* spec = read_spec in
  let* p = read problem in
  ensure (loop_terminates spec p)
    "The loop never terminates."
    ~failure:LOOP_DOES_NOT_TERMINATE

let ensure_invariant_possibly_sat =
  let* p = read problem in
  ensure (Arith.possibly_sat (mk_conj (p.param_assums @ problem_invs p)))
    "The invariant is not satisfiable."
    ~failure:INVARIANT_UNSAT

let ensure_inv_useful =
  let* spec = read_spec in
  let* problem = read problem in
  if has_invariant spec then
    let problem = problem |> set inv_main None |> set preserved None in
    ensure (not (Arith.surely_valid (post_obligation spec problem)))
      "The invariant is useless in proving the postcondition."
      ~failure:INVARIANT_USELESS
  else return ()

let check_params_use =
  let* spec = read_spec in
  match spec.use_params with
  | false -> return ()
  | true ->
    let* used_params = used_vars Var.Param in
    prefer (List.length used_params > 0)
      "No parameter variable was used in the loop and body." NO_PARAM_USED

let check_preference pred msg failure_code =
  let* spec = read_spec in
  let* prob = read problem in
  prefer (pred spec prob) msg failure_code

let main_inv_useful spec problem =
  if Option.is_some spec.num_main_inv_disjuncts then
    let problem = problem |> set inv_main None in
    not (Arith.surely_valid (post_obligation spec problem))
  else true

let check_main_inv_useful =
  check_preference main_inv_useful
    "The main invariant is useless in proving the postcondition."
    MAIN_INV_NOT_USEFUL

let relevant_formula ~when_ lens obligation spec problem =
  if not (when_ spec) then true
  else
    let problem = set lens Unknown problem in
    not (Arith.surely_valid (obligation spec problem))

let cond_guard_relevant =
  relevant_formula
    (body_cond |-- opt |-- cond_guard) preserved_inv_obligation
    ~when_:has_cond_guard

let check_cond_guard_relevant =
  check_preference cond_guard_relevant
    "The cond guard does not appear to be necessary."
    COND_GUARD_IRRELEVANT

let loop_guard_useful_for_post =
  relevant_formula loop_guard post_obligation
    ~when_:(fun spec -> spec.loop_guard_useful_for_post)

let check_loop_guard_useful_for_post =
  check_preference loop_guard_useful_for_post
    "The loop guard is not necessary for post."
    LOOP_GUARD_IRRELEVANT_IN_PROVING_POST

let loop_guard_useful_for_inv =
  relevant_formula loop_guard preserved_inv_main_obligation
    ~when_:(fun spec -> spec.loop_guard_useful_for_inv)

let check_loop_guard_useful_for_inv =
  check_preference loop_guard_useful_for_inv
    "The loop guard is not necessary for inv."
    LOOP_GUARD_IRRELEVANT_IN_PROVING_INV

let preserved_term_useful =
  relevant_formula (preserved |-- opt) (post_obligation ~full:true)
    ~when_:(fun spec -> Option.is_some spec.num_preserved_term_vars)

let check_preserved_term_useful =
  check_preference preserved_term_useful
    "The preserved-term invariant is not useful for post."
    PRESERVED_TERM_NOT_USEFUL

let relevant_composite_formula ~when_ lens obligation spec problem =
  if not (when_ spec) then true
  else
    let subs = get lens problem in
    List.for_all (Util.Combinatorics.remove_one subs) ~f:(fun smaller ->
      let problem = set lens smaller problem in
      let obligation = obligation spec problem in
      not (Arith.surely_valid obligation))

let post_assums_useful =
  relevant_composite_formula post (post_obligation ~full:true)
    ~when_:(fun _ -> true)

let check_post_assums_useful =
  check_preference post_assums_useful
    "Some post assumptions are useless."
    USELESS_POST_DISJUNCTS

let init_conjuncts_useful =
  relevant_composite_formula init init_obligation
    ~when_:(fun _ -> true)

let check_init_conjuncts_useful =
  check_preference init_conjuncts_useful
    "Some of the initial assumptions are not useful."
    USELESS_INIT_CONJUNCTS

let main_inv_disjuncts_useful =
  relevant_composite_formula (inv_main |-- opt) all_obligations
    (* We do not want to make this check when there is only one disjunct
       because then the smaller invariant would become [false] *)
    ~when_:(fun spec -> has_disjunctive_inv spec)

let check_main_inv_disjuncts_useful =
  check_preference main_inv_disjuncts_useful
    "Some parts of the main invariant are not useful."
    USELESS_INVARIANT_DISJUNCTS

let aux_inv_useful =
  relevant_composite_formula inv_aux preserved_inv_obligation
    ~when_:(fun spec -> Option.is_some spec.num_aux_inv_conjuncts)

let check_aux_inv_useful =
  check_preference aux_inv_useful
    "Some parts of the auxilliary invariant are not useful."
    AUX_INV_IRRELEVANT

let check_param_assums_preference =
  let* spec = read_spec in
  let* p = read problem in
  if spec.require_param_assums then
    prefer (not (List.is_empty p.param_assums))
    "No param assums." NO_PARAM_ASSUMS
  else return ()

(* ////////////////////////////////////////////////////////////////////////// *)
(* Gluing all checks together                                                 *)
(* ////////////////////////////////////////////////////////////////////////// *)

let stage_checks = function
  | After_guard ->
      return ()
  | After_inv ->
      let* () = ensure_invs_not_valid_unsat_or_redundant in
      let* () = ensure_invariant_possibly_sat in
      return ()
  | After_body ->
      let* () = ensure_loop_terminates in
      let* () = check_cond_guard_relevant in
      let* () = check_aux_inv_useful in
      let* () = check_loop_guard_useful_for_inv in
      return ()
  | After_post ->
      let* () = ensure_post_not_valid_or_redundant in
      let* () = ensure_inv_useful in
      let* () = check_params_use in
      let* () = check_loop_guard_useful_for_post in
      let* () = check_main_inv_useful in
      let* () = check_preserved_term_useful in
      let* () = check_post_assums_useful in
      return ()
  | After_init ->
      let* () = ensure_not_valid_unsat_or_redundant mk_conj init in
      let* () = ensure_loop_entered in
      let* () = check_main_inv_disjuncts_useful in
      let* () = check_init_conjuncts_useful in
      let* () = check_param_assums_preference in
      return ()

let stage_to_enum = function
  | After_guard -> 1
  | After_inv -> 2
  | After_body -> 3
  | After_post -> 4
  | After_init -> 5

let perform_checks stage = stage_checks stage

let set_stage stage =
  let* () = write Lenses.stage (Some stage) in
  perform_checks stage

let redo_all_checks stage =
  sequence_unit @@ List.filter_map all_of_stage ~f:(fun stage' ->
    if stage_to_enum stage >= stage_to_enum stage'
    then Some (stage_checks stage') else None)

let redo_all_checks_for_current_stage =
  let* stage = read Lenses.stage in
  match stage with
  | None -> return ()
  | Some stage -> redo_all_checks stage

(* ////////////////////////////////////////////////////////////////////////// *)
(* Abduction                                                                  *)
(* ////////////////////////////////////////////////////////////////////////// *)

let is_param_constr c =
  Formula.vars_set c |> Set.for_all ~f:Var.(has_kind Param)

let is_meta_constr c =
  let vars = Formula.vars_set c in
  Set.exists vars ~f:Var.(has_kind Meta_var) &&
  Set.for_all vars ~f:Var.(fun v -> has_kind Meta_var v || has_kind Param v)

let simple_comp = function
  | Comp (lhs, _, rhs) ->
    (Term.sub lhs rhs |> Term.to_alist |> List.length) <= 3
  | _ -> false

let final_refinement (_, t) =
  Term.vars_set t |> Set.for_all ~f:Var.(has_kind Param)

let smallest_candidates ?(max=max_abduction_candidates) ~show fmls =
  if List.length fmls <= max then fmls
  else
    let metric f = String.length (show f) in
    let compare f f' = Int.compare (metric f) (metric f') in
    List.take (List.sort ~compare fmls) max

let sort_by ~priority xs =
  List.sort xs ~compare:(fun r r' ->
    Int.compare (- priority r) (- priority r'))

(* Instantiate all constants and propose parameter constraints in such a way to
   make some obligation hold. Constant refinements are proposed first, starting
   with those for which a suggestion that only involves existing constants and
   parameters exist. When no constant remains to be refined, parameter
   constraints are proposed.*)
let rec abduct_consts
  ?always_allow_params ?(max_num_assums=max_num_assums) ~failure mk_obl =
  let settings = Arith.{
    default_abduction_settings with
    abduct_var_diff = None } in
  let* cconstrs = read cconstrs in
  let* obl =
    let* spec = read_spec in
    let* problem = read problem in
    return (mk_obl spec problem) in
  let* () = debug_msg ("Abduct: " ^ [%show: Formula.t] obl) in
  match Arith.abduct ~settings obl with
  | [] -> instantiate_all_consts ?always_allow_params ()
  | suggs ->
    let suggs = List.concat suggs in
    let refinements =
      List.filter suggs ~f:is_meta_constr
      |> List.concat_map ~f:(fun c ->
          Arith.elim_constraint c |> List.concat)
      |> List.dedup_and_sort ~compare:[%compare: string * Term.t] in
    let final_refinements =
      List.filter refinements ~f:final_refinement in
    let final_refinements =
      (* We want to refine constants that are explicitly labeled with
         [Abduct_tight] last. *)
      let priority (c, _) =
        match Map.find cconstrs c with
        | Some Abduct_tight -> 0
        | _ -> 1 in
      sort_by ~priority final_refinements in
    let meta_vars =
      (* We put metavars appearing with unit coeffs last because
         these are the last ones to be instantiated. *)
      let unit_coeff = vars_appearing_with_unit_coeff obl in
      let priority v =
        if Set.mem unit_coeff v then 0 else 1 in
      Formula.vars_set obl |> Set.to_list
      |> List.filter ~f:Var.(has_kind Meta_var)
      |> sort_by ~priority in
    begin match final_refinements, meta_vars with
      | (c, _)::_, _ ->
        (* Case where metavar ?c has at least one final refinement. *)
        let abducted =
          List.filter_map final_refinements ~f:(fun (c', t) ->
            if equal_string c c' then Some t else None)
          |> smallest_candidates ~max:max_abducted_terms_candidates
              ~show:[%show: Term.t] in
        let* () = instantiate_const ?always_allow_params ~abducted c in
        abduct_consts ~max_num_assums ~failure mk_obl
      | [], c::_ ->
        (* Case where no final refinement is available but metavars remain. *)
        let* () = instantiate_const ?always_allow_params c in
        abduct_consts ~max_num_assums ~failure mk_obl
      | [], [] ->
        (* Cases where there is no metavar left. *)
        if max_num_assums <= 0
        then lift (Search.fail "No more parameter assumptions are allowed.")
        else
          let* cur_prob = read problem in
          let assums =
            List.filter suggs ~f:is_param_constr
            |> List.filter ~f:(fun a ->
                Arith.possibly_sat (And (a :: cur_prob.param_assums)))
            |> List.filter ~f:simple_comp
            |> List.dedup_and_sort ~compare:[%compare: Formula.t]
            |> smallest_candidates ~show:[%show: Formula.t]
            |> List.map ~f:(fun a -> (a, `Assum, 1.)) in
          let* () = ensure (not (List.is_empty assums))
              ("Abduction error: " ^ [%show: Formula.t] obl) ~failure in
          let* _ = refine_formula (append param_assums) assums in
          let* () = modify (problem |-- param_assums) remove_redundant in
          let* () = redo_all_checks_for_current_stage in
          abduct_consts ~max_num_assums:(max_num_assums-1) ~failure mk_obl
  end

(* ////////////////////////////////////////////////////////////////////////// *)
(* Sampling a problem                                                         *)
(* ////////////////////////////////////////////////////////////////////////// *)

let term_of_coeffs coeffs = coeffs
  |> List.map ~f:(fun (c, v) -> Term.mulc c (Term.var v))
  |> List.fold ~init:Term.zero ~f:Term.add

let acceptable_coeffs coeffs =
  let coeffs = List.map coeffs ~f:fst in
  (List.fold coeffs ~init:0 ~f:Util.Math.gcd = 1) &&
  (List.length coeffs <= 2 ||
    List.count coeffs ~f:(fun c -> Int.abs c = 1) >= 2)

let generate_preserved_term =
  let open Compop in
  let open Util.Random in
  let* spec = read_spec in
  let* rng = read rng in
  match spec.num_preserved_term_vars with
  | None -> return ()
  | Some n ->
    let* x, y, z = fresh3 Var.Var in
    let* c = fresh1 Var.Meta_var in
    let* () = set_constr c Abduct_tight in
    let vars = match n with `Two -> [x; y] | `Three -> [x; y; z] in
    let eq_only = bernouilli rng ~p:0.7 in
    let available_consts = spec.available_consts in
    let init_scheme =
      List.map vars ~f:(fun v ->
        let op =
          if eq_only then EQ else pick rng [EQ; GE; LE] in
        let init_val = pick rng available_consts in
        (v, op, init_val)) in
    let* () =
      sequence_unit @@ List.map init_scheme ~f:(fun (v, op, lhs) ->
        modify problem (append init
          (Comp (Term.var v, op, Term.const lhs)))) in
    let eq_only =
      List.for_all init_scheme ~f:(fun (_, op, _) -> Compop.equal op EQ) in
    let op = if eq_only then EQ else GE in
    let rec make_inv_lhs () =
      let coeffs =
        List.map init_scheme ~f:(fun (v, op, _) ->
          let coeff_sign =
            match op with
            | LE -> -1 | GE -> 1
            | EQ -> pick rng [-1; 1]
            | _ -> assert false in
          (coeff_sign * pick rng small_pos_constants, v)) in
      if acceptable_coeffs coeffs then term_of_coeffs coeffs
      else make_inv_lhs () in
    let inv_lhs = make_inv_lhs () in
    let inv = Comp (inv_lhs, op, Term.var c) in
    let* () = modify problem (fun p -> {p with preserved = Some inv}) in
    abduct_consts init_obligation ~failure:FAILURE

let generate_guard =
  let* spec = read_spec in
  if not (has_loop_guard spec) then return ()
  else
    let* () = refine_with_template
      ~allow_fresh:true (set loop_guard)
      [spec.loop_guard_template, 1.] in
    instantiate_all_consts ()

let generate_main_inv =
  let* spec = read_spec in
  match spec.num_main_inv_disjuncts with
  | None -> return ()
  | Some num_disjuncts ->
    let* cur_prob = read problem in
    let add_disjunct = append (inv_main |-- opt) in
    begin
      let* () = write (problem |-- inv_main) (Some []) in
      let* _ =
        let* extra = relax_ineq cur_prob.loop_guard in
        let templates = uniform @@
          [`Lec; `Gec; `Eqv] @
          (if spec.allow_vcomp_in_prim_inv then [`Lev; `Ltv] else []) @
          (extra |> Option.to_list |> map_other) in
        refine_with_template ~allow_fresh:true add_disjunct templates in
      let* _ =
        begin match num_disjuncts with
        | `One -> return ()
        | `Two ->
          let guard = cur_prob.loop_guard in
          let templates = uniform @@
            [`Lec; `Gec; `Eqc] @
            (if Formula.equal guard Unknown then [] else [`Other guard]) in
          refine_with_template ~allow_fresh:true add_disjunct templates
        end in
      let* () = instantiate_all_consts () in
      return ()
    end

let generate_aux_inv =
  let* spec = read_spec in
  let* () =
    match spec.num_aux_inv_conjuncts with
    | None -> return ()
    | Some `One ->
        refine_with_template (append inv_aux) (uniform [`Lec; `Gec])
    | Some `Two ->
      let* () = refine_with_template (append inv_aux) [`Gec, 1.] in
      let* () = refine_with_template (append inv_aux) [`Gec, 1.] in
      return () in
  instantiate_all_consts ()

let vars_to_use_in_body =
  (* To be called once the guard and invariants are generated *)
  let* all = used_vars_set Var.Var in
  if Set.is_empty all then
    (* If the invariant is empty, we allow one variable *)
    let* v = fresh1 Var.Var in
    return (Set.singleton (module String) v)
  else return all

let vars_not_modified_in_body =
  let* p = read problem in
  let* all = vars_to_use_in_body in
  let in_body = problem_body p |> Prog.modified_vars in
  return (Set.diff all in_body)

let body_templates =
  let t1 = [`Incr; `Decr; `Incr_const; `Decr_const] in
  let t2 = [`Incr_var; `Assign_lin_neg_other] in
  let t3 = [`Assign_const; `Assign_var] in
  function
  | `Only_const_incr -> t1
  | `No_var_const_assign -> t1 @ t2
  | `All -> t1 @ t2 @ t3

let generate_body_cond =
  let* spec = read_spec in
  let templates = body_templates spec.assignment_templates in
  match spec.body_structure with
  | No_cond -> return ()
  | Cond {cond_guard=use_cond_guard; else_branch; single_instr} ->
    let* () = write (problem |-- body_cond) @@ Some {
      cond_guard = Unknown;
      cond_at_start = true;
      tbranch = Prog [];
      fbranch = Prog []} in
    let* () =
      if use_cond_guard then
        refine_with_template
          (set (body_cond |-- opt |-- cond_guard))
          (uniform [`Lec; `Gec; `Lev; `Ltv])
      else return () in
    let* tbranch_vars =
      let* target_vars = vars_not_modified_in_body in
      add_assignments ~templates ~single:single_instr ~target_vars
        (body_cond |-- opt |-- tbranch) in
    let* () =
      if not else_branch then return ()
      else
        let* _ =
          add_assignments
            ~templates
            ~single:single_instr
            ~target_vars:tbranch_vars
            (body_cond |-- opt |-- fbranch) in
          return () in
    return ()

let generate_body =
  let* () = generate_body_cond in
  let* spec = read_spec in
  let templates = body_templates spec.assignment_templates in
  let* target_vars = vars_not_modified_in_body in
  let* _ = add_assignments ~templates ~single:false ~target_vars body_common in
  abduct_consts preserved_inv_obligation ~failure:FAILED_TO_PROVE_INV_PRESERVED

let post_candidates problem =
  let assums = mk_conj @@
    problem_post_invs problem @ [Not problem.loop_guard] @
    List.map problem.post ~f:(fun f -> Not f) in
  Formula_util.possible_consequences assums
  |> List.filter ~f:(fun f ->
    (Formula.vars_set f) |> Set.exists ~f:Var.(has_kind Var))
  |> smallest_candidates ~show:[%show: Formula.t]

let second_disjunct problem =
  match problem.inv_main with
  | Some [_; d] -> Some d
  | _ -> None

let generate_post =
  let* spec = read_spec in
  let* () =
    if spec.disjunctive_post then
      let* prob = read problem in
      let templates = uniform @@
        [`Lec; `Gec; `Nec] @
        (second_disjunct prob |> Option.to_list |> map_other) in
      let* () = refine_with_template (append post) templates in
      instantiate_all_consts ()
    else return () in
  let* () =
    let* prob = read problem in
    let templates = wconcat [
      uniform [`Ltc; `Gtc; `Nec], 0.2;  (* TODO: why is it necessary? *)
      uniform (map_other (post_candidates prob)), 1.0] in
      refine_with_template (append post) templates in
  abduct_consts post_obligation ~failure:FAILED_TO_PROVE_POST

let generate_init =
  let* spec = read_spec in
  let rec aux i =
    if i >= max_num_additional_init_statements then return ()
    else
      let* candidates =
        if spec.equalities_only_for_init then return [`Eqc, 1.]
        else
          let settings = Arith.default_abduction_settings in
          let* problem = read problem in
          Arith.abduct ~settings (init_obligation spec problem)
          |> List.concat
          |> List.dedup_and_sort ~compare:[%compare: Formula.t]
          |> smallest_candidates ~show:[%show: Formula.t]
          |> List.map ~f:(fun i -> `Other i, 1.)
          |> return in
      let candidates = (`Unknown, 1.) :: candidates in
      let* template = refine_with_template' (append init) candidates in
      if Poly.equal template `Unknown then return ()
      else aux (i + 1)
    in
  let* () = aux 0 in
  let* () =
    abduct_consts init_obligation
      ~failure:FAILED_TO_PROVE_INIT
      ~always_allow_params:spec.require_param_assums in
  return ()

(* ////////////////////////////////////////////////////////////////////////// *)
(* Post processing                                                            *)
(* ////////////////////////////////////////////////////////////////////////// *)

let shuffle_prog rng (Prog instrs) = Prog (Util.Random.shuffle rng instrs)

let neg fml = Arith.simplify (Not fml)

let as_var t =
  match Term.to_alist t with
  | [Term.Var x, 1] -> Some x
  | _ -> None

let as_var_equality = function
  | Comp (lhs, Compop.EQ, rhs) ->
    Option.map (as_var lhs) ~f:(fun v -> (v, rhs))
  | _ -> None

let rearrange_conjunctions_and_disjunctions rng _spec problem =
  List.fold [init; post] ~init:problem ~f:(fun problem lens ->
    Lens.modify lens (Util.Random.shuffle rng) problem)

let move_conditional rng _spec problem =
  if Option.is_some problem.body_cond then
    let b = Util.Random.bernouilli rng ~p:0.5 in
    set (body_cond |-- opt |-- cond_at_start) b problem
  else problem

let rearrange_instructions rng _spec =
  map_children_progs ~f:(fun p ->
    Prog (prog_instrs p |> Util.Random.shuffle rng))

let shuffle_comparisons rng _spec problem =
  map_children_formula problem
    ~f:(Formula.apply_recursively ~f:(randomize_comparisons rng))

let move_param_assum rng _spec problem =
  match problem.param_assums with
  | first_assum::other_assums when Util.Random.bernouilli rng ~p:0.2 ->
    problem |> append post (neg first_assum) |> set param_assums other_assums
  | _ -> problem

let make_post_assums _rng _spec problem =
  match problem.post with
  | first::other ->
    let assums = List.map other ~f:(fun f -> Assume (neg f)) in
    problem |> set post [first] |> set extra_after_loop (Prog assums)
  | _ -> assert false

let make_init_assignments rng _spec problem =
  let extra =
    List.map (problem.param_assums @ problem.init) ~f:(fun f ->
      match as_var_equality f with
      | Some (x, t) when Util.Random.bernouilli rng ~p:0.8 -> Assign (x, t)
      | _ -> Assume f) in
  problem
    |> set param_assums [] |> set init []
    |> set extra_before_loop (Prog extra)

let shuffle_extra_instrs rng _spec problem =
  Lens.modify extra_before_loop (shuffle_prog rng) problem |>
  Lens.modify extra_after_loop (shuffle_prog rng)

let weaken_post rng spec problem =
  let weaken_formula = function
    | Formula.Comp (lhs, Compop.(LT | GT), rhs) ->
      Formula.Comp (lhs, NE, rhs)
    | Formula.Comp (lhs, EQ, rhs) when not (has_loop_guard spec) ->
      (* We only perform this transformation when there is no
         loop guard because weakening a post equality may make it less
         interesting. *)
      (* Sometimes this will be invalid but then
         the transformation will be rejected *)
      let c = Util.Random.pick rng spec.available_consts in
      let op = Util.Random.pick rng Compop.[NE] in
      Arith.prettify (Formula.Comp (lhs, op, Term.(add rhs (const c))))
    | f -> f in
  Lens.modify post (List.map ~f:weaken_formula) problem

(* Utilities *)

(* We omit [check_init_conjuncts_useful] and [check_post_assums_useful] *)
let num_constraint_violations spec problem =
  let init_not_valid_unsat_or_redundant s p =
    Option.is_none (not_valid_unsat_or_redundant mk_conj init s p) in
  let post_not_valid_or_redundant s p =
    Option.is_none (post_not_valid_or_redundant s p) in
  List.count ~f:(fun f -> not (f spec problem)) [
    loop_terminates;
    cond_guard_relevant;
    aux_inv_useful;
    loop_guard_useful_for_inv;
    post_not_valid_or_redundant;
    loop_guard_useful_for_post;
    main_inv_useful;
    preserved_term_useful;
    init_not_valid_unsat_or_redundant;
    loop_entered;
    main_inv_disjuncts_useful]

let with_prob p trans rng spec problem =
  if Util.Random.bernouilli ~p rng then trans rng spec problem
  else problem

let checked trans rng spec problem =
  let transformed = trans rng spec problem in
  if Arith.surely_valid (all_obligations spec transformed)
  then transformed else problem

let rec checked_rep ?(attempts=10) trans rng spec problem =
  if attempts <= 0 then problem
  else
    let transformed = trans rng spec problem in
    if Arith.surely_valid (all_obligations spec transformed) &&
       (num_constraint_violations spec transformed <=
        num_constraint_violations spec problem)
    then transformed
    else checked_rep ~attempts:(attempts-1) trans rng spec problem

let perform_transformations transformations =
  let* spec = read_spec in
  let* rng = read rng in
  monadic_iter transformations ~f:(fun trans ->
    modify problem (trans rng spec))

(* Transformations on the final extracted program *)

let turn_params_into_vars p =
  let open Var in
  Prog.subst_multi p ~f:(fun v ->
    if has_kind Param v then Term.var (make Var (base_name v))
    else Term.var v)

(* ////////////////////////////////////////////////////////////////////////// *)
(* Adding useless formulas                                                    *)
(* ////////////////////////////////////////////////////////////////////////// *)

let random_formula vars params consts rng =
  let open Util.Random in
  let op = pick rng Compop.[EQ; NE; GE; GT; LE; LT] in
  let lhs_var = pick rng vars in
  let rhs_vars = List.filter vars ~f:(fun v -> not (equal_string v lhs_var)) in
  let rhs_choices =
    wconcat @@ uniform [
      uniform (List.map consts ~f:Term.const);
      uniform (List.map params ~f:Term.var);
      uniform (List.map rhs_vars ~f:Term.var)] in
  let rhs = sample (make_distr rhs_choices) rng in
  (Comp (Term.var lhs_var, op, rhs))

let add_useless_loop_guard vars params consts rng _spec problem =
  let guard = random_formula vars params consts rng in
  match problem.loop_guard with
  | Unknown -> problem |> set loop_guard guard
  | _ -> problem

let add_useless_init vars params consts rng _spec problem =
  let fml = random_formula vars params consts rng in
  problem |> append init fml

let add_useless_post vars params consts rng _spec problem =
  let fml = random_formula vars params consts rng in
  problem |> append post fml

let add_useless_conditional vars params consts rng _spec problem =
  let guard = random_formula vars params consts rng in
  let body = problem_body problem in
  let body = Prog [If (guard, body, Prog [])] in
  problem |> set body_common body |> set body_cond None

let add_useless_choice _rng _spec problem =
  let body = problem_body problem in
  let body = Prog [If (Unknown, body, body)] in
  problem |> set body_common body |> set body_cond None

let add_useless_elements =
  let* vars = used_vars Var.Var in
  let* params = used_vars Var.Param in
  let* spec = read_spec in
  let consts = spec.available_consts in
  perform_transformations [
    with_prob 0.3 (checked_rep (add_useless_loop_guard vars params consts));
    with_prob 0.2 (checked_rep (add_useless_init vars params consts));
    with_prob 0.2 (checked_rep (add_useless_post vars params consts));
    with_prob 0.1 (checked (add_useless_conditional vars params consts));
    with_prob 0.02 add_useless_choice]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Main strategy                                                              *)
(* ////////////////////////////////////////////////////////////////////////// *)

type result = {
  problem: Prog.t;
  nonprocessed: Prog.t }

let generate_problem =
  let* () = generate_preserved_term in
  let* () = generate_guard in
  let* () = set_stage After_guard in
  let* () = generate_main_inv in
  let* () = generate_aux_inv in
  let* () = set_stage After_inv in
  let* () = generate_body in
  let* () = set_stage After_body in
  let* () = generate_post in
  let* () = set_stage After_post in
  let* () = generate_init in
  let* () = set_stage After_init in
  return ()

let post_processing =
  let* () = add_useless_elements in
  let* () = perform_transformations [
    rearrange_conjunctions_and_disjunctions;
    checked move_conditional;
    checked rearrange_instructions;
    shuffle_comparisons;
    move_param_assum;
    make_post_assums;
    checked make_init_assignments;
    checked shuffle_extra_instrs;
    with_prob 0.4 (checked weaken_post)] in
  return ()

let main_with_spec =
  let* () = generate_problem in
  let* nonprocessed =
    let* prob = read problem in
    return (problem_program prob |> turn_params_into_vars) in
  let* processed =
    let* () = post_processing in
    let* prob = read problem in
    return (problem_program prob
      |> turn_params_into_vars
      |> (first_loop |-- invariants) ^= []) in
  return {problem=processed; nonprocessed}

let main =
  let* () = sample_spec in
  let* () = control_spec_difficulty in
  let* () = check_preserved_term_spec_difficulty in
  let* () = check_spec_consistency in
  main_with_spec

(* ////////////////////////////////////////////////////////////////////////// *)
(* Drivers                                                                    *)
(* ////////////////////////////////////////////////////////////////////////// *)

let init_problem = {
  param_assums = []; init = []; post = [];
  loop_guard = Unknown; preserved = None;
  inv_main = None; inv_aux = [];
  body_common = Prog []; body_cond = None;
  extra_before_loop = Prog [];
  extra_after_loop = Prog [] }

let init_state ?spec rng = State.({
  problem=init_problem; spec; rng;
  stage = None;
  cconstrs = Map.empty (module String) })

let init_teacher rng =
  let state = init_state rng in
  Search.map ~f:fst (run_state main state)
  |> Search.search_tree

let init_teacher_with_spec rng spec_sexp =
  let spec =
    Parsexp.Conv_single.parse_string_exn spec_sexp problem_spec_of_sexp in
  let state = init_state ~spec rng in
  Search.map ~f:fst (run_state main_with_spec state)
  |> Search.search_tree