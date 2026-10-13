open Base
open Util.Monads

(* Disabling this may be useful for debugging purposes. *)
let fail_on_empty_choices = false

(* This optimization may be a bad idea as it may create infinite loops. *)
let optimize_singleton_choices = false

(* Enable adding debug messages in the tree (Msg constructor) *)
let enable_messages = true


module Make
  (Probe: sig type t end) (Summary: sig type t end)
  (Event: sig type event type outcome val default_failure: outcome end) =
struct

  module T =
  struct
    type +'a tree =
      | Pure: 'a -> 'a tree
      | Fail: string * Event.outcome -> 'a tree
      | Msg: string * (unit -> 'a tree) -> 'a tree
      | Event: Event.event * (unit -> 'a tree) -> 'a tree
      | Choice: {
          chance: bool;
          probe: Probe.t;
          choices: 'b choice list;
          cont: 'b -> 'a tree } -> 'a tree

    and 'b choice = {
      item: 'b;
      summary: Summary.t;
      weight: float }

    type 'a t = {run_cont: 'r. (('a -> 'r tree) -> 'r tree)}

    let return x = {run_cont = fun k -> k x}

    let bind {run_cont} ~f = {run_cont =
      fun k -> run_cont (fun x -> (f x).run_cont k)}
  end

  include T
  include MakeMonad (T)

  let search_tree {run_cont} = run_cont (fun x -> Pure x)

  let choose ~chance ~probe choices = {run_cont = fun k ->
    Choice {chance; probe; choices; cont = fun b -> k b}}

  let fail ?(outcome=Event.default_failure) msg =
    {run_cont = fun _ -> Fail (msg, outcome)}

  let message s =
    if enable_messages then {run_cont = fun k -> Msg (s, k)}
    else return ()

  let event e = {run_cont = fun k -> Event (e, k)}

  let ensure ?failure pred msg =
    if not pred then fail ?outcome:failure msg else return ()

  (* Optimization: no need to branch *)
  let choose ?failmsg ?(chance=false) ~probe = function
    | [] when fail_on_empty_choices ->
      fail (Option.value failmsg ~default:"Empty choice.")
    | [c] when optimize_singleton_choices -> return c.item
    | cs -> choose ~chance ~probe cs

end