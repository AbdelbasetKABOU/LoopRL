open Base

module type SEARCH_MONAD = sig

  type probe
  type summary
  type outcome
  type event

  include Util.Monads.MONAD

  type +'a tree =
    | Pure: 'a -> 'a tree
    | Fail: string * outcome -> 'a tree
    | Msg: string * (unit -> 'a tree) -> 'a tree
    | Event: event * (unit -> 'a tree) -> 'a tree
    | Choice: {
        chance: bool;
        probe: probe;
        choices: 'b choice list;
        cont: 'b -> 'a tree } -> 'a tree

  and 'b choice = {
    item: 'b;
    summary: summary;
    weight: float }

  val search_tree: 'a t -> 'a tree
  val fail: ?outcome:outcome -> string -> 'a t
  val event: event -> unit t
  val message: string -> unit t
  val ensure: ?failure:outcome -> bool -> string -> unit t
  val choose:
    ?failmsg:string -> ?chance:bool ->
    probe:probe -> 'a choice list -> 'a t

end