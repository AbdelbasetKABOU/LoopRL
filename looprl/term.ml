open Base
open Base_quickcheck

type atom =
  | Var of string
  | FunApp of string * t list
  | One
  [@@deriving eq, compare, hash, sexp]

and t = (atom * int) list [@@deriving eq, compare, hash, sexp]

let to_alist t = t

let coeff atom t =
  let rec aux = function
    | [] -> None
    | (a, c)::acs ->
      if equal_atom a atom then Some c
      else aux acs in
  aux t

let rec add_atom ts ((a, c) as ac) =
  match ts with
  | [] -> [ac]
  | ((a', c') as ac')::rest ->
    if equal_atom a a' then
      let cnew = c + c' in
      if cnew = 0 then rest
      else (a, cnew) :: rest
    else ac' :: add_atom rest ac

(* Arithmetic perations *)

let zero = []

let const c = if c = 0 then zero else [(One, c)]

let atom a = [(a, 1)]

let one = atom One

let var x = atom (Var x)

let funapp f args = atom (FunApp (f, args))

let add t t' = List.fold t' ~init:t ~f:add_atom

let mulc k t =
  if k = 0 then zero
  else List.map t ~f:(fun (a, c) -> (a, c * k))

let neg t = mulc (-1) t

let divc t k =
  assert (k <> 0);
  List.map t ~f:(fun (a, c) -> (a, c / k))

let sub t t' = add t (mulc (-1) t')

let sum = List.fold ~init:zero ~f:add

module Infix = struct
  let ( + ) = add
  let ( - ) = sub
  let ( * ) = mulc
  let ( / ) = divc
  let (~-) = neg
end

(* Queries *)

let coeffs_gcd = List.fold ~init:0 ~f:(fun acc (_, c) -> Util.Math.gcd acc c)

let get_const = function
  | [] -> Some 0
  | [One, c] -> Some c
  | _ -> None

(* Substitution *)

let rec atom_subst_multi ~f =
  function
  | One -> atom One
  | Var x -> f x
  | FunApp (s, args) -> atom (FunApp (s, List.map args ~f:(subst_multi ~f)))

and subst_multi ~f t =
    List.map t ~f:(fun (a, c) -> mulc c (atom_subst_multi ~f a)) |> sum

let replace_only x t =
  fun x' ->
  if equal_string x x' then t
  else var x'

let subst ~from ~substituted = subst_multi ~f:(replace_only from substituted)

(* Other queries *)

let rec atom_vars_set = function
  | One -> Set.empty (module String)
  | FunApp (_, ts) -> Set.union_list (module String) (List.map ts ~f:(vars_set))
  | Var x -> Set.singleton (module String) x

and vars_set t =
  Set.union_list (module String)
    (List.map t ~f:(fun (a, _) -> atom_vars_set a))

let partition ~f t = List.partition_tf t ~f:(fun (x, _) -> f x)

(* Pretty printing *)

let parens s = "(" ^ s ^ ")"

let rec atom_to_string = function
  | One -> "1"
  | Var x -> x
  | FunApp (s, args) ->
      s ^ parens (String.concat ~sep:", " (List.map args ~f:to_string))

and to_string t =
  if List.is_empty t then "0"
  else
    let pp_sub (a, c) =
      if equal_atom a One then Int.to_string c
      else if c = 1 then atom_to_string a
      else if c = -1 then "-" ^ atom_to_string a
      else Int.to_string c ^ "*" ^ atom_to_string a in
    List.fold t ~init:"" ~f:(fun acc st ->
      let s = pp_sub st in
      if String.is_empty acc then s
      else if Char.equal (String.get s 0) '-'
      then acc ^ " - " ^ String.suffix s (String.length s - 1)
      else acc ^ " + " ^ s)

let pp = Fmt.using to_string Fmt.string

(* Evaluating terms *)

module type DOMAIN = sig
  type t
  val zero: t
  val one: t
  val add: t -> t -> t
  val cmul: int -> t -> t
end

let eval_atom_domain
    (type num) (module D: DOMAIN with type t = num) ~valuation =
  function
    | One -> Some D.one
    | Var x -> valuation x
    | FunApp _ -> None

let eval_domain
    (type num)
    (module D: DOMAIN with type t = num)
    ?(valuation=fun _ -> None)
    e =
  List.fold e ~init:(Some D.zero) ~f:(fun acc (a, c) ->
    let open Option.Let_syntax in
    let%bind acc = acc in
    let%bind av = eval_atom_domain (module D) ~valuation a in
    return (D.add acc (D.cmul c av)))

let eval = eval_domain (
  module struct
    type t = int
    let zero = 0
    let one = 1
    let add = ( + )
    let cmul = ( * )
  end)

(* Quickcheck *)

let quickcheck_generator =
  let open Generator in
  recursive_union
    [ return zero
    ; map ~f:const (int_inclusive (-2) 2)
    ; return (var "x")
    ; return (var "y") ]
  ~f:(fun t ->
    [ map2 t t ~f:add
    ; map2 t t ~f:sub
    ; map t ~f:(mulc 2)])

(* Unit testing *)

let%expect_test "test_subst" =
  Infix.(var "x" + var "x" + const 1)
    |> [%show: t] |> Stdio.print_endline;
  Infix.(var "x" + var "y" + const 1) |> subst ~from:"y" ~substituted:(var "x")
    |> [%show: t] |> Stdio.print_endline;
  [%expect {|
    2*x + 1
    2*x + 1 |}]

let%test_unit "test_invariants" =
  Test.with_sample_exn quickcheck_generator ~f:(fun seq ->
    Sequence.take seq 1000 |> Sequence.iter ~f:(fun t ->
      assert (List.for_all t ~f:(fun (_, c) -> c <> 0))))