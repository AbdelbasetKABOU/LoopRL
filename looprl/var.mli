type kind = Var | Var_hole | Meta_var | Param
val make : kind -> string -> string
val has_kind : kind -> string -> bool
val kind: string -> kind
val base_name: string -> string
val decompose : string -> kind * string