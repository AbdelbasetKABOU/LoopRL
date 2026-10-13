open Base
open Python_lib
open Python_lib.Let_syntax
open Looprl

let () = if not (Py.is_initialized ()) then Py.initialize ()

open Looprl.Python.Make ()
module PyTeacher = Search_wrapper (Teacher)
module PySolver = Search_wrapper (Solver)

let py_teacher, unserialize_teacher =
  PyTeacher.make_class "TeacherState"
    python_of_teacher_result sexp_of_teacher_result teacher_result_of_sexp

let py_solver, unserialize_solver =
  PySolver.make_class "SolverState"
    python_of_prog Prog.sexp_of_t Prog.t_of_sexp

let init_teacher =
  let%map_open rng = positional "rng" rng_param  ~docstring:"" in
  fun () ->
  let tree = Teacher.init_teacher rng in
  Class_wrapper.wrap py_teacher tree

let init_teacher_with_spec =
  let%map_open rng = positional "rng" rng_param  ~docstring:""
  and spec = positional "spec_sexp" string  ~docstring:"" in
  fun () ->
  let tree = Teacher.init_teacher_with_spec rng spec in
  Class_wrapper.wrap py_teacher tree

let init_solver =
  let%map_open p = positional "prog" prog_param ~docstring:"" in
  let solver = Solver.init_solver p in
  fun () -> Class_wrapper.wrap py_solver solver

let pretraining_tasks_sampler =
  let%map_open rng = positional "rng" rng_param ~docstring:""
  and true_false = keyword "true_false" bool ~docstring:"" in
  fun () ->
    let py_sample _ =
      let (assums, concl, non_concl) =
        Pretraining.sample_task ~true_false ~random_sigils:true rng in
      let mk_probe f = Graphable.Pack (f, (module Pretraining.Probe)) in
      let mk f = Graphable.Pack (f, (module Pretraining.Choice_summary)) in
      [%python_of: graphable * graphable * graphable]
        (mk_probe assums, mk concl, mk non_concl) in
    Py.Callable.of_function py_sample

let token_encoding_size =
  let%map_open config =
    positional "config" tensorizer_config_param ~docstring:"" in
  fun () -> [%python_of: int] (Tensorize.token_encoding_size config)

let uid_encoding_offset =
  let%map_open config =
    positional "config" tensorizer_config_param ~docstring:"" in
  fun () -> [%python_of: int] (Tensorize.uid_encoding_offset config)

let num_edge_types =
  Defunc.no_arg (fun () () -> [%python_of: int] Token.num_edges)

let () =
  let mod_ = Py_module.create "looprl_ocaml_lib" in
  Py_module.set mod_ "init_teacher" init_teacher;
  Py_module.set mod_ "init_teacher_with_spec" init_teacher_with_spec;
  Py_module.set mod_ "init_solver" init_solver;
  Py_module.set mod_ "unserialize_teacher" unserialize_teacher;
  Py_module.set mod_ "unserialize_solver" unserialize_solver;
  Py_module.set mod_ "pretraining_tasks_sampler" pretraining_tasks_sampler;
  Py_module.set mod_ "token_encoding_size" token_encoding_size;
  Py_module.set mod_ "uid_encoding_offset" uid_encoding_offset;
  Py_module.set mod_ "num_edge_types" num_edge_types;
  Py_module.set mod_ "unserialize_teacher_probe"
    (python_unserialize_graphable (module Teacher.Probe));
  Py_module.set mod_ "unserialize_teacher_action"
    (python_unserialize_graphable (module Teacher.Choice_summary));
  Py_module.set mod_ "unserialize_solver_probe"
    (python_unserialize_graphable (module Solver.Probe));
  Py_module.set mod_ "unserialize_solver_action"
    (python_unserialize_graphable (module Solver.Choice_summary));
  Py_module.set mod_ "unserialize_formula"
    (python_unserialize_graphable (module Pretraining.Choice_summary));
  Py_module.set_value mod_ "solver_spec" PySolver.agent_spec;
  Py_module.set_value mod_ "teacher_spec" PyTeacher.agent_spec;
  Class_wrapper.register_in_module rng_class mod_;
  Class_wrapper.register_in_module prog_class mod_;
  Class_wrapper.register_in_module uid_map_class mod_;
  Class_wrapper.register_in_module graphable_class mod_