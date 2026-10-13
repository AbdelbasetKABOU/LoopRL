open Base
open Python_lib
open Python_lib.Let_syntax
open Event_intf

(* We put everything in a generative functor because some
   of the code below depends on [Py.init] *)

module Make () = struct

(* Random seeds *)

let rng_class =
  let init = Class_wrapper.Init.defunc begin
    let%map_open seed = keyword_opt "seed" (list int) ~docstring:"" in
    fun _ ->
    match seed with
    | None -> Random.State.make_self_init ()
    | Some seed -> Random.State.make (Array.of_list seed)
  end in
  Class_wrapper.make "CamlRng" ~init ~methods:[]

let rng_of_python p = Class_wrapper.unwrap rng_class p |> Option.value_exn
let rng_param = Defunc.Of_python.create ~type_name:"CamlRng" ~conv:rng_of_python

(* Program wrapper *)


let prog_class =
  let init = Class_wrapper.Init.defunc begin
    let%map_open s = positional "prog_str" string ~docstring:"" in
    fun _ -> Parse.program s end in
  let normalize_task = Class_wrapper.Method.no_arg "normalize_task"
    (fun cls ~self:(p, _) ->
      Class_wrapper.wrap cls (Prog_util.normalize_task p)) in
  let to_string _ = [%show: Prog.t] in
  Class_wrapper.make "Prog"
    ~init ~to_string ~to_string_repr:to_string ~methods:[normalize_task]

type prog = Prog.t [@@deriving sexp]  (* useful for ppx_python *)
let python_of_prog = Class_wrapper.wrap prog_class
let prog_of_python p = Class_wrapper.unwrap prog_class p |> Option.value_exn
let prog_param = Defunc.Of_python.create ~type_name:"Prog" ~conv:prog_of_python

(* Tensorizer configuration *)

let tensorizer_config_param =
  Defunc.Of_python.create
    ~type_name:"TensorizerConfig" ~conv:[%of_python: Tensorize.config]

(* Tokenizer configuration *)

let tokenizer_config_param =
  Defunc.Of_python.create
    ~type_name:"TokenizerConfig"
    ~conv:[%of_python: Graphable.tokenizer_config]

(* Unique identifier maps *)

let uid_map_class =
  let open Tensorize in
  let init =
    Class_wrapper.Init.no_arg (fun _ -> Uid_map.empty) in
  let to_string_repr _ m =
    let vars = Uid_map.vars m in
    Fmt.(str "UidMap(%a)" (list ~sep:comma string) vars) in
  Class_wrapper.make "UidMap" ~init ~to_string_repr ~methods:[]

let uid_map_param =
  Defunc.Of_python.create ~type_name:"UidMap" ~conv:(fun py ->
    Class_wrapper.unwrap uid_map_class py |> Option.value_exn)

let python_of_uid_map = Class_wrapper.wrap uid_map_class

(* Graphable wrapper *)

let graphable_class =
  let to_string _ (Graphable.Pack (x, (module G))) = G.to_string x in
  let meta = Class_wrapper.Method.no_arg "meta"
    (fun _ ~self:(Graphable.Pack (g, (module G)), _) ->
      G.to_meta g
      |> List.map ~f:(fun (k, v) -> (k, python_of_string v))
      |> Py.Dict.of_bindings_string) in
  let graph = Class_wrapper.Method.no_arg "graph"
    (fun _ ~self:(Graphable.Pack (g, (module G)), _) ->
      G.to_graph g
      |> Token_graph.add_canonical_edges
      |> [%show: Token_graph.t] |> python_of_string) in
  let tensorize = Class_wrapper.Method.defunc "tensorize" begin
    let%map_open config =
      positional "tensorizer_config" tensorizer_config_param ~docstring:""
    and tokenizer_config =
      positional "tokenizer_config" tokenizer_config_param ~docstring:""
    and uids =
      positional "uids" uid_map_param ~docstring:"" in
    fun _ ~self:(Graphable.Pack (g, (module G)), _) ->
      Graphable.tensorize (module G)
        ~tokenizer_config ~config uids g
      |> [%python_of: Tensorize.GraphTensors.t * uid_map]
    end in
  let serialize = Class_wrapper.Method.no_arg "serialize"
    (fun _ ~self:(Graphable.Pack (g, (module G)), _) ->
      G.sexp_of_t g |> Sexp.to_string_mach |> python_of_string) in
  Class_wrapper.make "Graphable"
    ~to_string ~to_string_repr:to_string
    ~methods:[meta; graph; tensorize; serialize]

let wrap_graphable (type a) (module G: Graphable.S with type t = a) (g: a) =
  Class_wrapper.wrap graphable_class (Graphable.Pack (g, (module G)))

let python_unserialize_graphable
  (type a) (module G: Graphable.S with type t = a) =
  let%map_open sexp = positional "sexp" string ~docstring:"" in
  fun () ->
    Parsexp.Conv_single.parse_string_exn sexp G.t_of_sexp
    |> wrap_graphable (module G)

type graphable = Graphable.t

let python_of_graphable = Class_wrapper.wrap graphable_class

(* Teacher result *)

type teacher_result = Teacher.result = {
  problem: prog;
  nonprocessed: prog } [@@deriving python, sexp]

(* Agent specification *)

type agent_spec = {
  event_names: string list;
  outcome_names: string list;
  event_rewards: float list;
  outcome_rewards: float list;
  event_max_occurences: int list;
  success_code: int;
  default_failure_code: int;
  size_limit_exceeded_code: int;
  min_success_reward: float
} [@@deriving python]

(* Search wrapper *)

module type WITH_SEARCH = sig
  module Probe: Graphable.S
  module Choice_summary: Graphable.S
  module Event: EVENT
  module Search: Search_intf.SEARCH_MONAD with
    type summary := Choice_summary.t and
    type probe := Probe.t and
    type event := Event.event and
    type outcome := Event.outcome
end

module Search_wrapper (M: WITH_SEARCH) = struct

  open M
  open Search

  let not_a_choice_point () = failwith "Not a choice point."

  let unserialize_error () =
    failwith "It is impossible to resume an unserialized search tree."

  let sexp_of_choice {summary; weight; item=_} =
    Sexp.List [
      [%sexp_of: M.Choice_summary.t] summary;
      [%sexp_of: float] weight]

  let choice_of_sexp sexp =
    let open Sexp in
    match sexp with
    | List [summary; weight] ->
      let summary = [%of_sexp: M.Choice_summary.t] summary in
      let weight = [%of_sexp: float] weight in
      {item=(); summary; weight}
    | _ -> failwith "Invalid sexp for choice point"

  let sexp_of_search_tree sexp_of_res tree =
    let open Sexp in
    let mk name args = List (Atom name::args) in
    match tree with
    | Pure res -> mk "Pure" [sexp_of_res res]
    | Fail (msg, outcome) ->
        let msg = [%sexp_of: string] msg in
        let outcome = [%sexp_of: M.Event.outcome] outcome in
        mk "Fail" [msg; outcome]
    | Msg (msg, _) -> mk "Msg" [[%sexp_of: string] msg]
    | Event (e, _) -> mk "Event" [[%sexp_of: M.Event.event] e]
    | Choice {chance; probe; choices; _} ->
      let chance = [%sexp_of: bool] chance in
      let probe = [%sexp_of: M.Probe.t] probe in
      let choices = List.sexp_of_t sexp_of_choice choices in
      mk "Choice" [chance; probe; choices]

  let search_tree_of_sexp res_of_sexp sexp =
    let open Sexp in
    match sexp with
    | List [Atom "Fail"; msg; outcome] ->
        Fail ([%of_sexp: string] msg, [%of_sexp: M.Event.outcome] outcome)
    | List [Atom "Msg"; msg] ->
        Msg ([%of_sexp: string] msg, fun () -> unserialize_error ())
    | List [Atom "Event"; e] ->
        Event ([%of_sexp: M.Event.event] e, fun () -> unserialize_error ())
    | List [Atom "Pure"; v] -> Pure (res_of_sexp v)
    | List [Atom "Choice"; chance; probe; choices] ->
        let chance = [%of_sexp: bool] chance in
        let probe = [%of_sexp: M.Probe.t] probe in
        let choices = List.t_of_sexp choice_of_sexp choices in
        let cont () = unserialize_error () in
        Choice {chance; probe; choices; cont}
    | _ -> assert false

  let make_class class_name python_of_success sexp_of_success success_of_sexp =
    let to_string_repr _ = function
      | Pure _ -> "<pure>"
      | Fail _ -> "<fail>"
      | Msg _ -> "<msg>"
      | Event _ -> "<event>"
      | Choice {chance=false; _} -> "<choice>"
      | Choice {chance=true; _} -> "<chance>" in
    let is_choice = Class_wrapper.Method.no_arg "is_choice"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Choice _ -> Py.Bool.t
        | _ -> Py.Bool.f) in
    let is_chance = Class_wrapper.Method.no_arg "is_chance"
    (fun _ ~self:(tree,_) ->
      match tree with
      | Choice {chance=true; _} -> Py.Bool.t
      | _ -> Py.Bool.f) in
    let is_failure = Class_wrapper.Method.no_arg "is_failure"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Fail _ -> Py.Bool.t
        | _ -> Py.Bool.f) in
    let is_success = Class_wrapper.Method.no_arg "is_success"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Pure _ -> Py.Bool.t
        | _ -> Py.Bool.f) in
    let is_message = Class_wrapper.Method.no_arg "is_message"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Msg _ -> Py.Bool.t
        | _ -> Py.Bool.f) in
    let is_event = Class_wrapper.Method.no_arg "is_event"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Event _ -> Py.Bool.t
        | _ -> Py.Bool.f) in
    let success_value = Class_wrapper.Method.no_arg "success_value"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Pure x -> python_of_success x
        | _ -> assert false) in
    let failure_message = Class_wrapper.Method.no_arg "failure_message"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Fail (err, _) -> python_of_string err
        | _ -> assert false) in
    let failure_code = Class_wrapper.Method.no_arg "failure_code"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Fail (_, outcome) -> python_of_int (M.Event.outcome_to_enum outcome)
        | _ -> assert false) in
    let event_code = Class_wrapper.Method.no_arg "event_code"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Event (e, _) -> python_of_int (M.Event.event_to_enum e)
        | _ -> assert false) in
    let message = Class_wrapper.Method.no_arg "message"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Msg (s, _) -> python_of_string s
        | _ -> assert false) in
    let next = Class_wrapper.Method.no_arg "next"
      (fun cls ~self:(tree,_) ->
        match tree with
        | Msg (_, next) | Event (_, next) ->
            Class_wrapper.wrap cls (next ())
        | _ -> assert false) in
    let probe = Class_wrapper.Method.no_arg "probe"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Choice {probe; _} -> wrap_graphable (module Probe) probe
        | _ -> not_a_choice_point ()) in
    let choices = Class_wrapper.Method.no_arg "choices"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Choice {choices; _} ->
          List.map choices ~f:(fun c ->
            Graphable.Pack (c.summary, (module Choice_summary)))
          |> [%python_of: graphable list]
        | _ -> not_a_choice_point ()) in
    let weights = Class_wrapper.Method.no_arg "weights"
      (fun _ ~self:(tree,_) ->
        match tree with
        | Choice {choices; _} ->
          List.map choices ~f:(fun c -> c.weight)
          |> [%python_of: float list]
        | _ -> not_a_choice_point ()) in
    let select = Class_wrapper.Method.defunc "select"
      begin
        let open Python_lib.Let_syntax in
        let%map_open i =
          positional "i" int ~docstring:"index of the selected choice" in
        fun cls ~self:(tree,_) ->
          match tree with
          | Choice {choices; cont; _} ->
            let sel = (List.nth_exn choices i).item in
            Class_wrapper.wrap cls (cont sel)
          | _ -> assert false
      end in
    let serialize = Class_wrapper.Method.no_arg "serialize"
      (fun _ ~self:(tree,_) ->
        sexp_of_search_tree sexp_of_success tree
        |> Sexp.to_string_hum
        |> [%python_of: string]) in
    let cls =
      Class_wrapper.make class_name ~to_string_repr
        ~methods:[
          serialize;
          is_choice; is_failure; is_success; is_event; is_chance;
          is_message; message; next;
          failure_code; event_code;
          success_value; failure_message;
          probe; choices; weights; select] in
    let unserialize =
      let open Python_lib.Let_syntax in
      let%map_open sexp = positional "sexp" string ~docstring:"" in
      fun () ->
      Parsexp.Conv_single.parse_string_exn sexp
        (search_tree_of_sexp success_of_sexp)
      |> Class_wrapper.wrap cls in
    cls, unserialize

  let all_events, all_outcomes =
    let open M.Event in
    List.init (max_event + 1) ~f:(
      fun i -> event_of_enum i |> Option.value_exn),
    List.init (max_outcome + 1) ~f:(
      fun i -> outcome_of_enum i |> Option.value_exn)

  let agent_spec =
    let open M.Event in {
    event_names = List.map all_events ~f:(fun e ->
      show_event e |> Util.Names.transform_ppx_enum_name);
    outcome_names = List.map all_outcomes ~f:(fun e ->
      show_outcome e |> Util.Names.transform_ppx_enum_name);
    event_rewards = List.map all_events ~f:event_reward;
    outcome_rewards = List.map all_outcomes ~f:outcome_reward;
    event_max_occurences = List.map all_events ~f:max_event_occurences;
    success_code = outcome_to_enum success;
    default_failure_code = outcome_to_enum default_failure;
    size_limit_exceeded_code = outcome_to_enum size_limit_exceeded;
    min_success_reward = min_success_reward
  } |> [%python_of: agent_spec]

end

end  (* closing the main generative functor *)