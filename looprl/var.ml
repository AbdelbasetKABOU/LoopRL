open Base

type kind = Var | Var_hole | Meta_var | Param [@@deriving equal]

let sigil = function
  | Var -> ""
  | Var_hole -> "_"
  | Meta_var -> "?"
  | Param -> "#"

let make kind name = sigil kind ^ name

let decompose name =
  assert (not (String.is_empty name));
  match name.[0] with
  | '_' -> Var_hole, String.drop_prefix name 1
  | '?' -> Meta_var, String.drop_prefix name 1
  | '#' -> Param, String.drop_prefix name 1
  | _ -> Var, name

let kind name = fst (decompose name)

let base_name name = snd (decompose name)

let has_kind k var = equal_kind k (kind var)

let%test_unit "var_kinds" =
  let exs = ["x"; "y"; "?z"; "#c1"; "_num"] in
  List.iter exs ~f:(fun ex ->
    let (k, s) = decompose ex in
    assert (equal_string ex (make k s)))