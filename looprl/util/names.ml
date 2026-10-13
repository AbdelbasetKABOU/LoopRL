open Base

(* Convert a string such as 'Tokenizer.MY_TOKEN' into 'my-token' *)
(* This is useful for changing the default behavior of ppx_show *)
let transform_ppx_enum_name s =
  String.lowercase s
  |> String.substr_replace_all ~pattern:"_" ~with_:"-"
  |> String.split_on_chars ~on:['.']
  |> List.last_exn