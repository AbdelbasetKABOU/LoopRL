(* Wrapper to use Menhir with Sedlex *)
let parse ~parser ~lexer lexbuf =
  let provider () =
    let tok = lexer lexbuf in
    let start, stop =  Sedlexing.lexing_positions lexbuf in
    tok, start, stop in
  MenhirLib.Convert.Simplified.traditional2revised parser provider

let parse_string ~parser ~lexer str =
  let lexbuf = Sedlexing.Utf8.from_string str in
  parse ~parser ~lexer lexbuf

let formula =
  parse_string ~lexer:Lexer.token ~parser:Parser.single_formula

let program =
  parse_string ~lexer:Lexer.token ~parser:Parser.single_program

let term =
  parse_string ~lexer:Lexer.token ~parser:Parser.single_term

let instr s =
  match program s with
  | Prog [instr] -> instr
  | _ -> assert false

let%expect_test "parsing_terms" =
  let ex s =
    term s |> [%show: Term.t] |> Stdio.print_endline in
  ex "x + 1 + 3";
  ex "x - y + y - x";
  ex "x + y + 2*x - 2*y";
  ex "y + x + 2*x - 2*y";
  ex "4 + f(g(x + y), ?c + 4)";
  [%expect {|
    x + 4
    0
    3*x - y
    -y + 3*x
    4 + f(g(x + y), ?c + 4) |}]

let%expect_test "parsing_formulas" =
  let ex s =
    formula s |> [%show: Formula.t] |> Stdio.print_endline in
  ex "x >= ?c";
  ex "j>=i-3";
  ex "x>0 -> x>=0";
  ex "x>0 -> y>0 -> x+y>0";
  ex "y>=0 && (binary: x == 0 || x == 1)";
  [%expect{|
    x >= ?c
    j >= i - 3
    x > 0 -> x >= 0
    x > 0 -> y > 0 -> x + y > 0
    y >= 0 && (binary: x == 0 || x == 1) |}]

let%expect_test "parsing_programs" =
  let ex p =
    program p
    |> [%show: Prog.t]
    |> fun s -> (Stdio.print_endline s; Stdio.print_endline "") in
  ex "assume x >=0;";
  ex "assume x >= 0; y = 1; while (??) {x = x + y; y = y + 1;} assert x >= y;";
  ex "if (x >= 0) { x = x + 1; } else { x = x - 1; }";
  ex "if (x >= 0) { if (y >= 0) { assume x + y >= 0; } }";
  ex "while (??) { invariant x >= 0; 'to-prove'} assert x >= 0; 'prove-later'";
  ex "init: { } while (true) { }";
  ex "init: { x = 1; } while (true) { }";
  ex "init: { x = 1; y = 0; } while (true) { }";
  ex "init; while (??) {x = x+1;} mid; while (??) {x = x-1;}";
  [%expect{|
    assume x >= 0;

    assume x >= 0;
    y = 1;
    while (??) {
        x = x + y;
        y = y + 1;
    }
    assert x >= y;

    if (x >= 0) {
        x = x + 1;
    } else {
        x = x - 1;
    }

    if (x >= 0) {
        if (y >= 0) {
            assume x + y >= 0;
        }
    }

    while (??) {
        invariant x >= 0 'to-prove';
    }
    assert x >= 0 'prove-later';

    init: { }
    while (true) {

    }

    init: x = 1;
    while (true) {

    }

    init: {
        x = 1;
        y = 0;
    }
    while (true) {

    }

    init;
    while (??) {
        x = x + 1;
    }
    mid;
    while (??) {
        x = x - 1;
    } |}]