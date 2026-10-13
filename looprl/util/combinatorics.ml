open Base

let rec subsets = function
  | [] -> [[]]
  | x::xs ->
    let ss = subsets xs in
    ss @ List.map ss ~f:(fun s -> x::s)

let strict_subsets xs = List.rev (List.tl_exn (List.rev (subsets xs)))

let%expect_test "strict_subsets" =
  strict_subsets [1; 2; 3]
  |> List.iter ~f:(fun s -> Stdio.print_endline ([%show: int list] s));
  [%expect {|
    []
    [3]
    [2]
    [2; 3]
    [1]
    [1; 3]
    [1; 2] |}]

let remove_one xs =
  let n = List.length xs in
  List.init n ~f:(fun i -> List.filteri xs ~f:(fun j _ -> i <> j))

let%expect_test "remove_one" =
  let ex l =
    remove_one l |> [%show: int list list] |> Stdio.print_endline in
  ex [];
  ex [1];
  ex [1; 2; 3];
  [%expect {|
    []
    [[]]
    [[2; 3]; [1; 3]; [1; 2]] |}]