open Base

(* ////////////////////////////////////////////////////////////////////////// *)
(* Types and definitions                                                      *)
(* ////////////////////////////////////////////////////////////////////////// *)

module Compop = struct
  type t = EQ | NE | LE | LT | GE | GT
    [@@deriving eq, compare, hash, sexp]

  let not = function
    | EQ -> NE
    | NE -> EQ
    | LE -> GT
    | LT -> GE
    | GE -> LT
    | GT -> LE

  let reverse = function
    | EQ -> EQ
    | NE -> NE
    | LE -> GE
    | LT -> GT
    | GE -> LE
    | GT -> LT

  let interpret = function
    | EQ -> ( = )
    | NE -> ( <> )
    | LE -> ( <= )
    | LT -> ( < )
    | GE -> ( >= )
    | GT -> ( > )

  let to_string = function
    | EQ -> "=="
    | NE -> "!="
    | LE -> "<="
    | LT -> "<"
    | GE -> ">="
    | GT -> ">"
end

type t =
  | Unknown
  | Labeled of string * t option
  | Bconst of bool
  | And of t list
  | Or of  t list
  | Not of t
  | Implies of t * t
  | Comp of Term.t * Compop.t * Term.t
  [@@deriving eq, compare, hash, sexp]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Smart constructors                                                         *)
(* ////////////////////////////////////////////////////////////////////////// *)

module Infix = struct

  let (==) lhs rhs = Comp (lhs, Compop.EQ, rhs)
  let (<>) lhs rhs = Comp (lhs, Compop.NE, rhs)
  let (<=) lhs rhs = Comp (lhs, Compop.LE, rhs)
  let (<)  lhs rhs = Comp (lhs, Compop.LT, rhs)
  let (>=) lhs rhs = Comp (lhs, Compop.GE, rhs)
  let (>)  lhs rhs = Comp (lhs, Compop.GT, rhs)

end

(* ////////////////////////////////////////////////////////////////////////// *)
(* Utilities                                                                  *)
(* ////////////////////////////////////////////////////////////////////////// *)

let is_false = function Bconst false -> true | _ -> false
let is_true = function Bconst true -> true | _ -> false

let mk_conj = function
  | [] -> Bconst true
  | [x] -> x
  | xs -> And xs

let mk_disj = function
  | [] -> Bconst false
  | [x] -> x
  | xs -> Or xs

let rec mk_implies assumptions conclusion =
  match assumptions with
  | [] -> conclusion
  | a::assumptions -> Implies (a, mk_implies assumptions conclusion)

let conjuncts = function
  | Bconst true -> []
  | And args -> args
  | e -> [e]

let disjuncts = function
  | Bconst false -> []
  | Or args -> args
  | e -> [e]

let comp_eval lhs op rhs =
  let open Option.Let_syntax in
  let%bind lhs = Term.eval lhs in
  let%bind rhs = Term.eval rhs in
  return (Compop.interpret op lhs rhs)

let rec vars_set = function
  | Bconst _  | Labeled (_, None) | Unknown -> Set.empty (module String)
  | Labeled (_, Some arg) -> vars_set arg
  | And args | Or args ->
      Set.union_list (module String) (List.map ~f:vars_set args)
  | Not a -> vars_set a
  | Implies (lhs, rhs) -> Set.union (vars_set lhs) (vars_set rhs)
  | Comp (lhs, _, rhs) -> Set.union (Term.vars_set lhs) (Term.vars_set rhs)

let rec pred_symbols = function
  | Labeled (s, None) -> Set.singleton (module String) s
  | Labeled (s, Some arg) -> Set.add (pred_symbols arg) s
  | Bconst _  | Unknown | Comp _ -> Set.empty (module String)
  | And args | Or args ->
      Set.union_list (module String) (List.map ~f:pred_symbols args)
  | Not a -> pred_symbols a
  | Implies (lhs, rhs) -> Set.union (pred_symbols lhs) (pred_symbols rhs)

let map_children ~f e =
  match e with
  | And args -> And (List.map ~f args)
  | Or args -> Or (List.map ~f args)
  | Not arg -> Not (f arg)
  | Implies (lhs, rhs) -> Implies (f lhs, f rhs)
  | Labeled (s, Some arg) -> Labeled (s, Some (f arg))
  | Comp _ | Bconst _ | Labeled (_, None) | Unknown -> e

let rec iter_sub ~f e =
  f e;
  match e with
  | Comp _ | Bconst _  | Labeled (_, None) | Unknown -> ()
  | And args | Or args -> List.iter args ~f:(iter_sub ~f)
  | Not arg | Labeled (_, Some arg) -> iter_sub ~f arg
  | Implies (lhs, rhs) -> iter_sub ~f lhs; iter_sub ~f rhs

let rec apply_recursively ~f e = f (map_children ~f:(apply_recursively ~f) e)

let subst ~from ~substituted = apply_recursively ~f:(function
  | Comp (lhs, op, rhs) -> Comp (
      Term.subst ~from ~substituted lhs, op,
      Term.subst ~from ~substituted rhs)
  | e -> e)

let subst_multi ~f = apply_recursively ~f:(function
  | Comp (lhs, op, rhs) -> Comp (
      Term.subst_multi ~f lhs, op,
      Term.subst_multi ~f rhs)
  | e -> e)

let subst_pred_symbol ~from ~substituted = apply_recursively ~f:(function
  | Labeled (s, _) when equal_string s from -> substituted
  | e -> e)

(* ////////////////////////////////////////////////////////////////////////// *)
(* Pretty printing                                                            *)
(* ////////////////////////////////////////////////////////////////////////// *)

let parens str = "(" ^ str ^ ")"

let rec implies_chain = function
  | Implies (lhs, rhs) -> lhs :: implies_chain rhs
  | e -> [e]

let rec to_string fml =
  match fml with
  | Unknown -> "??"
  | Labeled (s, None) -> s
  | Labeled (s, Some arg) -> parens (s ^ ": " ^ to_string arg)
  | Bconst b -> Bool.to_string b
  | And args -> ac_op ~parent:fml "&&" args
  | Or args -> ac_op ~parent:fml "||" args
  | Not arg -> funcall "!" [arg]
  | Implies _ -> ac_op ~parent:fml "->" (implies_chain fml)
  | Comp (lhs, op, rhs) ->
    Term.to_string lhs ^ " " ^ Compop.to_string op ^ " " ^ Term.to_string rhs

and funcall f args =
  f ^ parens (String.concat ~sep:", " (List.map ~f:to_string args))

and infix ~parent op children =
  let pprec = precedence parent in
  children
  |> List.map ~f:(fun c ->
      let s = to_string c in
      let p = precedence c in
      if p <= pprec then parens s else s)
  |> String.concat ~sep:op

and ac_op ~parent op children =
  let n = List.length children in
  if n >= 2 then infix ~parent (" " ^ op ^ " ") children
  else funcall op children

and precedence e =
  match e with
  | Bconst _ | Comp _ | Not _ | Labeled _ | Unknown -> 4
  | And _ -> 3
  | Or _ -> 2
  | Implies _ -> 1

let pp f x = Fmt.string f (to_string x)