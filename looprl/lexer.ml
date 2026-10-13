open Parser

module L = Sedlexing.Utf8

let lowercase_letter = [%sedlex.regexp? 'a'..'z']
let uppercase_letter = [%sedlex.regexp? 'A'..'Z']
let letter = [%sedlex.regexp? lowercase_letter | uppercase_letter]
let digit = [%sedlex.regexp? '0'..'9']
let identifier = [%sedlex.regexp? letter, Star (letter | digit | "_")]
let nonneg_integer = [%sedlex.regexp? Plus digit]
let string_literal = [%sedlex.regexp? "'", Plus (Compl (Chars "'")), "'"]

let reserved_identifiers =
  [ "true",   TRUE
  ; "false",  FALSE
  ; "while",  WHILE
  ; "if",     IF
  ; "else",   ELSE
  ; "assume", ASSUME
  ; "assert", ASSERT
  ; "invariant", INVARIANT ]

let trim_string_literal str = String.sub str 1 (String.length str - 2)

let rec token buf =
  match%sedlex buf with
  | eof -> EOF
  | string_literal -> STRING_LIT (trim_string_literal (L.lexeme buf))
  | white_space -> token buf
  | nonneg_integer -> INT_CONST (int_of_string (L.lexeme buf))
  | Opt ("?" | "_" | "#"), letter, Star (letter | digit | "_") ->
      let s = L.lexeme buf in
      begin match List.assoc_opt s reserved_identifiers with
      | Some x -> x
      | None -> IDENTIFIER s
      end
  | "," -> COMMA
  | ":" -> COLON
  | ";" -> SEMICOLON
  | "+" -> PLUS
  | "-" -> MINUS
  | "*" | "⋅" -> TIMES
  | "(" -> LEFT_PAREN
  | ")" -> RIGHT_PAREN
  | "{" -> LEFT_CURLY
  | "}" -> RIGHT_CURLY
  | "=" -> ASSIGN
  | "==" -> EQ
  | "!=" -> NE
  | "<" -> LT
  | "<=" -> LE
  | ">" -> GT
  | ">=" -> GE
  | "->" -> IMPLIES
  | "&" -> AND
  | "&&" -> AND
  | "|" -> OR
  | "||" -> OR
  | "!" -> NOT
  | "??" -> UNKNOWN
  | _ -> failwith "Lexer error"