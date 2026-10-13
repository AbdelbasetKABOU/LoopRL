open Base

type rng = Base.Random.State.t

(* Using the Fischer Yates algorithm to shuffle an array *)
let shuffle_array rng t =
  let n = Array.length t in
  for i = 0 to n - 2 do
    let j = Base.Random.State.int_incl rng i (n - 1) in
    let tmp = t.(i) in
    t.(i) <- t.(j);
    t.(j) <- tmp
  done

let shuffle rng l =
  let t = Array.of_list l in
  shuffle_array rng t;
  Array.to_list t

let pick rng l =
  let n = List.length l in
  List.nth_exn l (Base.Random.State.int_incl rng 0 (n - 1))

let%expect_test "shuffle" =
  let ex lst =
    let res = shuffle (Base.Random.State.make [||]) lst in
    Stdio.print_endline
      ([%show: int list] lst ^ " -> " ^ [%show: int list] res) in
  ex []; ex [1]; ex [1; 2; 3; 4; 5];
  [%expect {|
    [] -> []
    [1] -> [1]
    [1; 2; 3; 4; 5] -> [3; 4; 5; 2; 1] |}]

type 'a distr = ('a * float) list

let make_distr xs =
  let open Float in
  let tot = List.fold xs ~init:0. ~f:(fun acc (_, p) -> acc + p) in
  assert (tot > 0.);
  List.map xs ~f:(fun (x, p) -> (x, p/tot))

let sample vs rng =
  let u = Random.State.float rng 1.0 in
  let rec aux cum = function
  | [] -> assert false
  | (x, p)::xs ->
    let open Float in
    let cum = cum + p in
    if u <= cum then x else aux cum xs in
  aux 0. vs

let%expect_test "sample" =
  let rng = Base.Random.State.make [||] in
  let distr = make_distr [(1, 2.); (2, 1.); (3, 1.)] in
  let samples = List.init 1000 ~f:(fun _ -> sample distr rng) in
  let count i = List.count samples ~f:(equal_int i) in
  Stdio.print_endline ([%show: int list] [count 1; count 2; count 3]);
  [%expect {| [500; 260; 240] |}]

let filtered f distr rng =
  let rec aux i =
    assert (i < 10000);
    let v = distr rng in
    if f v then v else aux (i + 1) in
  aux 0

let bernouilli ?(p=0.5) rng =
  let open Float in
  Base.Random.State.float rng 1. <= p

let sample_subset rng xs =
  List.filter xs ~f:(fun _ -> bernouilli rng)
