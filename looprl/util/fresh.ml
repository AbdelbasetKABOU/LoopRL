open Base

let fresh_id ?(preferred=[]) ~backup_prefix ~used () =
  let rec try_preferred = function
    | [] -> try_backup 1
    | cand::preferred ->
      if not (Set.mem used cand) then cand
      else try_preferred preferred
  and try_backup i =
    let cand = backup_prefix ^ Int.to_string i in
    if not (Set.mem used cand) then cand
    else try_backup (i + 1)
  in try_preferred preferred

let%expect_test "fresh_id" =
  let preferred = ["x"; "y"; "z"] in
  let backup_prefix = "v" in
  let ex used =
    let used = Set.of_list (module String) used in
    fresh_id ~preferred ~backup_prefix ~used () in
  List.map ~f:ex [
    ["a, b"]; ["x"; "z"]; ["y"]; [""];
    preferred; (preferred @ ["v1"]);
  ] |> [%show: string list] |> Stdio.print_endline;
  [%expect {|
    ["x"; "y"; "x"; "x"; "v1"; "v2"] |}]