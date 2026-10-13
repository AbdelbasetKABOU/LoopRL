open Base
open Formula
open Term
open Prog
open Token
open Token_graph

(* ////////////////////////////////////////////////////////////////////////// *)
(* Turning programs into trees                                                *)
(* ////////////////////////////////////////////////////////////////////////// *)

let leaf ?flags ?name ?cval token =
  Node (tok ?flags ?name ?cval token, [])

let node ?flags ?name ?cval token children =
  Node (tok ?flags ?name ?cval token, children)

let var_kind = function
  | Var.Var -> VAR
  | Meta_var -> META_VAR
  | Var_hole -> VAR_HOLE
  | Param -> PARAM

let rec atom = function
  | One -> assert false
  | Var name -> leaf ~name (var_kind (Var.kind name))
  | FunApp (s, args) ->
    node ~name:s FUN_APP (List.map ~f:term args)

and const cval =
  match cval with
  | -3 -> leaf ~cval MINUS_THREE
  | -2 -> leaf ~cval MINUS_TWO
  | -1 -> leaf ~cval MINUS_ONE
  |  0 -> leaf ~cval ZERO
  |  1 -> leaf ~cval ONE
  |  2 -> leaf ~cval TWO
  |  3 -> leaf ~cval THREE
  | c when c > 0 -> leaf ~cval POS_CONST
  | c when c < 0 -> leaf ~cval NEG_CONST
  | _ -> assert false

and atom_coeff = function
  | (_, 0)  -> assert false
  | (Term.One, c) -> const c
  | (a, 1) -> atom a
  | (a, -1) -> node NEG [atom a]
  | (a, c) -> node CMUL [const c; atom a]

and term t =
  match Term.to_alist t with
  | [] -> const 0
  | [ac] -> atom_coeff ac
  | acs -> node ADD (List.map acs ~f:atom_coeff)

let comp_op = function
  | Compop.EQ -> EQ
  | Compop.NE -> NE
  | Compop.GE -> GE
  | Compop.GT -> GT
  | Compop.LE -> LE
  | Compop.LT -> LT

let rec formula = function
  | Unknown -> leaf UNKNOWN
  | Labeled (name, None) -> leaf ~name LABELED
  | Labeled (name, Some arg) -> node ~name LABELED [formula arg]
  | Bconst true -> leaf TRUE
  | Bconst false -> leaf FALSE
  | Not arg -> node NOT [formula arg]
  | And args -> node AND (List.map args ~f:formula)
  | Or args -> node OR (List.map args ~f:formula)
  | Implies (lhs, rhs) -> node IMPLIES [formula lhs; formula rhs]
  | Comp (lhs, op, rhs) -> node (comp_op op) [term lhs; term rhs]

let proof_status = function
  | None -> []
  | Some To_prove -> [TO_PROVE]
  | Some To_prove_later -> [TO_PROVE_LATER]
  | Some Proved -> [PROVED]
  | Some Proved_conditionally -> [PROVED_CONDITIONALLY]

let rec prog (Prog instrs) =
  node PROG (List.map instrs ~f:instr)
and instr = function
  | LabeledProg (name, None) -> leaf ~name LABELED_PROG
  | LabeledProg (name, Some p) -> node ~name LABELED_PROG [prog p]
  | Assume f -> node ASSUME [formula f]
  | Assert (f, pst) -> node ~flags:(proof_status pst) ASSERT [formula f]
  | Assign (x, e) -> node ASSIGN [leaf ~name:x VAR; term e]
  | If (c, tb, fb) -> node IF_ELSE [formula c; prog tb; prog fb]
  | While (g, invs, body) ->
    node WHILE [formula g; node INVARIANTS (List.map invs ~f:inv); prog body]
and inv (f, pst) =
  node ~flags:(proof_status pst) INVARIANT [formula f]

let%expect_test "tokenize_prog" =
  {|
    assume x>=0;
    y = 0;
    while ((guard: 2*x + y <= ?c)) {
      invariant x>0;
      x = _x + 1;
    }
    assert false;
  |}
  |> Parse.program
  |> prog
  |> [%show: Token.augmented Token_graph.tree]
  |> Stdio.print_endline;
  [%expect{|
    (PROG
      (ASSUME (GE (VAR.x) (ZERO.0)))
      (ASSIGN (VAR.y) (ZERO.0))
      (WHILE
        (LABELED.guard
          (LE (ADD (CMUL (TWO.2) (VAR.x)) (VAR.y)) (META_VAR.?c)))
        (INVARIANTS (INVARIANT (GT (VAR.x) (ZERO.0))))
        (PROG (ASSIGN (VAR.x) (ADD (VAR_HOLE._x) (ONE.1)))))
      (ASSERT (FALSE))) |}]

(* ////////////////////////////////////////////////////////////////////////// *)
(* Testing static analysis                                                    *)
(* ////////////////////////////////////////////////////////////////////////// *)

let program p =
  Token_graph.create (prog p)
  |> Static_analysis.add_prog_semantic_edges

let instr i =
  Token_graph.create (instr i)

let formula f =
  Token_graph.create (formula f)

let term f =
  Token_graph.create (term f)

let var name =
  Token_graph.singleton (tok ~name (var_kind (Var.kind name)))

let test_prog p =
  let g =
    Token_graph.{tree=prog p; edges=[]}
    |> Static_analysis.add_prog_semantic_edges in
  Fmt.pr "%a" Token_graph.pp g

let simple_example_prog = {|
  x = 0;
  while (x < n) {
    invariant x <= n; 'to-prove'
    x = x + 3;
  }
  assert x == n; 'proved...' |}
  |> Parse.program

let%expect_test "static_analysis_simple" =
  test_prog simple_example_prog;
  [%expect {|
    (0:PROG
      (1:ASSIGN (2:VAR.x) (3:ZERO.0))
      (4:WHILE
        (5:LT (6:VAR.x) (7:VAR.n))
        (8:INVARIANTS
          (9:INVARIANT.TO_PROVE (10:LE (11:VAR.x) (12:VAR.n))))
        (13:PROG
          (14:ASSIGN (15:VAR.x) (16:ADD (17:VAR.x) (18:THREE.3)))))
      (19:ASSERT.PROVED_CONDITIONALLY (20:EQ (21:VAR.x) (22:VAR.n))))
    (LAST_READ 7<-4 11<-4 12<-4 17<-4 22<-4)
    (LAST_WRITE 2<-1 6<-1 11<-1 17<-1 21<-1 6<-14 11<-14 15<-14 17<-14 21<-14)
    (GUARDED_BY 7<-5 11<-5 12<-5 17<-5 7<-10 12<-10 17<-10 22<-10)
    (GUARDED_BY_NEG 21<-5 22<-5)
    (COMPUTED_FROM 15<-17) |}]

let example_prog = {|
  x = 13;
  assume x > y;
  while (x >= 0) {
    invariant x >= ?c;
    x = x - 1;
    y = y - 2;
    if (z < 0) {
        z = 1;
    } else {
        z = z + 1;
    }
  }
  assert y < 0; |}
  |> Parse.program

let%expect_test "static_analysis" =
  test_prog example_prog;
  [%expect {|
    (0:PROG
      (1:ASSIGN (2:VAR.x) (3:POS_CONST.13))
      (4:ASSUME (5:GT (6:VAR.x) (7:VAR.y)))
      (8:WHILE
        (9:GE (10:VAR.x) (11:ZERO.0))
        (12:INVARIANTS (13:INVARIANT (14:GE (15:VAR.x) (16:META_VAR.?c))))
        (17:PROG
          (18:ASSIGN (19:VAR.x) (20:ADD (21:VAR.x) (22:MINUS_ONE.-1)))
          (23:ASSIGN (24:VAR.y) (25:ADD (26:VAR.y) (27:MINUS_TWO.-2)))
          (28:IF_ELSE
            (29:LT (30:VAR.z) (31:ZERO.0))
            (32:PROG (33:ASSIGN (34:VAR.z) (35:ONE.1)))
            (36:PROG
              (37:ASSIGN (38:VAR.z) (39:ADD (40:VAR.z) (41:ONE.1)))))))
      (42:ASSERT (43:LT (44:VAR.y) (45:ZERO.0))))
    (LAST_READ 15<-8 21<-8 40<-28)
    (LAST_WRITE 2<-1 6<-1 10<-1 15<-1 21<-1 10<-18 15<-18 19<-18 21<-18 24<-23
      26<-23 44<-23 30<-33 34<-33 40<-33 30<-37 38<-37 40<-37)
    (GUARDED_BY 10<-5 15<-5 21<-5 26<-5 44<-5 15<-9 21<-9 16<-14 21<-14)
    (GUARDED_BY_NEG 40<-29)
    (COMPUTED_FROM 19<-21 24<-26 38<-40) |}]