module type MONAD_DEF =
sig
  type +'a t
  val return: 'a -> 'a t
  val bind: 'a t -> f:('a -> 'b t) -> 'b t
end

module type MONAD =
sig
  include MONAD_DEF
  val (>>=): 'a t -> ('a -> 'b t) -> 'b t
  val (let*): 'a t -> ('a -> 'b t) -> 'b t
  val map: 'a t -> f:('a -> 'b) -> 'b t
  val map2: 'a t -> 'b t -> f:('a -> 'b -> 'c) -> 'c t
  val sequence: 'a t list -> 'a list t
  val sequence_unit: unit t list -> unit t
  val monadic_iter: 'a list -> f:('a -> unit t) -> unit t
  val monadic_list_filter: 'a list -> f:('a -> bool t) -> 'a list t
  val monadic_concat_map: 'a list -> f:('a -> 'b list t) -> 'b list t
  val iterm: 'a list -> f:('a -> unit t) -> unit t
end

module MakeMonad (M: MONAD_DEF): (MONAD with type 'a t := 'a M.t) =
struct
  include M
  let (>>=) mx f = M.bind mx ~f
  let (let*) mx f = M.bind mx ~f
  let map mx ~f = mx >>= (fun x -> return (f x))
  let map2 mx my ~f = mx >>= (fun x -> my >>= fun y -> return (f x y))
  let rec sequence = function
    | [] -> return []
    | mx::mxs ->
      let* x = mx in
      let* xs = sequence mxs in
      return (x::xs)
  let sequence_unit mxs =
    let* _: unit list = sequence mxs in
    return ()
  let iterm xs ~f = List.map f xs |> sequence_unit
  let rec monadic_list_filter xs ~f =
    match xs with
    | [] -> return []
    | x::xs ->
      let* b = f x in
      let* rest = monadic_list_filter xs ~f in
      if b then return (x::rest) else return rest
  let rec monadic_concat_map xs ~f =
    match xs with
    | [] -> return []
    | x::xs ->
      let* ys = f x in
      let* yss = monadic_concat_map xs ~f in
      return (ys @ yss)
  let monadic_iter xs ~f = List.map f xs |> sequence_unit
end

module Identity: (MONAD with type 'a t = 'a) =
struct
  module T =
  struct
    type 'a t = 'a
    let return x = x
    let bind x ~f = f x
  end
  include T
  include MakeMonad (T)
end

module StateT (S: sig type t end) (M: MONAD):
sig
  include MONAD
  val read: (S.t, 'a) Lens.t -> 'a t
  val write: (S.t, 'a) Lens.t -> 'a -> unit t
  val modify: (S.t, 'a) Lens.t -> ('a -> 'a) -> unit t
  val with_local: (S.t, 'a) Lens.t -> 'b t -> 'b t
  val with_value: (S.t, 'a) Lens.t -> 'a -> 'b t -> 'b t
  val with_modified: (S.t, 'a) Lens.t -> ('a -> 'a) -> 'b t -> 'b t
  val lift: 'a M.t -> 'a t
  val run_state: 'a t -> S.t -> ('a * S.t) M.t
end =
struct
  module T =
  struct
    type 'a t = S.t -> ('a * S.t) M.t
    let return x = fun st -> M.return (x, st)
    let bind trans ~f =
       fun st -> M.bind (trans st) ~f:(fun (x, st) -> (f x) st)
  end
  include T
  include MakeMonad (T)
  open Lens.Infix
  let read lens = fun st -> M.return (st |. lens, st)
  let write lens v = fun st -> M.return ((), (lens ^= v) @@ st)
  let modify lens f = fun st -> M.return ((), (lens ^%= f) @@ st)
  let lift mx = fun st -> M.map mx ~f:(fun x -> (x, st))
  let run_state trans = trans
  let with_local lens comp =
    let* s = read lens in
    let* r = comp in
    let* _ = write lens s in
    return r
  let with_modified lens f comp =
    with_local lens @@
    let* _ = modify lens f in
    comp
  let with_value lens v = with_modified lens (fun _ -> v)
end
