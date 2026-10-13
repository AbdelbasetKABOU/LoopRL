open Base

let print_sset label ss =
  Set.to_list ss
  |> [%show: string list]
  |> fun s -> label ^ ": " ^ s
  |> Stdio.print_endline

let analyze_program p =
  print_sset "Variables" (Prog.vars_set p);
  print_sset "Predicate symbols" (Prog.pred_symbols p);
  print_sset "Program symbols" (Prog.prog_symbols p)

let%expect_test "query functions" =
  {|
    assume S;
    x = 0;
    while (x >= 0 || C) {
      invariant x >= y && I;
      body_common;
      if (cond) {
        z = x;
        body_cond;
      }
    }
    assert F && x == z;
  |}
    |> Parse.program
    |> analyze_program;
  [%expect {|
    Variables: ["x"; "y"; "z"]
    Predicate symbols: ["C"; "F"; "I"; "S"; "cond"]
    Program symbols: ["body_common"; "body_cond"] |}]


let%expect_test "substitutions" =
  {| x = 0; while (cond) { body; x = x + 1; } |}
    |> Parse.program
    |> Prog.subst_prog_symbol
      ~from:"body"
      ~substituted:(Parse.program "z = f(c + 1); y = x + y;")
    |> Prog.subst_pred_symbol
      ~from:"cond"
      ~substituted: (Parse.formula "x <= c")
    |> Prog.subst
      ~from:"c" ~substituted:(Parse.term "10")
    |> [%show: Prog.t]
    |> Stdio.print_endline;
  [%expect{|
    x = 0;
    while (x <= 10) {
        z = f(11);
        y = x + y;
        x = x + 1;
    } |}]

let%expect_test "nested_substs" =
  {| x = 0; I: { y = 0; J: {} } z = 0; |}
  |> Parse.program
  |> Prog.subst_prog_symbol ~from:"J" ~substituted:(Parse.program "y = 1;")
  |> [%show: Prog.t]
  |> Stdio.print_endline;
  [%expect{|
    x = 0;
    I: {
        y = 0;
        y = 1;
    }
    z = 0; |}]

let%expect_test "wlp" =
  let ex prog post =
    let prog = Parse.program prog in
    let post = Parse.formula post in
    let res = Prog.wlp prog post in
    Stdio.print_endline ([%show: Formula.t] res) in
  ex "x = x + y; y = y + 1;" "x >= y";
  [%expect {| x + y >= y + 1 |}]