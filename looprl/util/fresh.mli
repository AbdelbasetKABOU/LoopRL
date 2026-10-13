open Base

val fresh_id:
  ?preferred:string list ->
  backup_prefix:string -> used:Set.M(String).t -> unit -> string