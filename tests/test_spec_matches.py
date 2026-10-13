from looprl_lib.training.analyze import spec_matches

spec_example = """
    ((num_preserved_term_vars ()) (num_main_inv_disjuncts (One))
    (num_aux_inv_conjuncts ()) (disjunctive_post false)
    (body_structure
    (Cond (cond_guard false) (else_branch false) (single_instr true)))
    (body_implies_main_inv true) (loop_guard_useful_for_inv false)
    (loop_guard_useful_for_post true) (use_params true)
    (require_param_assums false) (equalities_only_for_init true)
    (loop_guard_template Lec) (allow_vcomp_in_prim_inv false)
    (assignment_templates All))
    """

spec_pat_1 = "((num_preserved_term_vars ()) (loop_guard_useful_for_post true))"
spec_pat_2 = "((loop_guard_useful_for_post true) (num_preserved_term_vars ()))"
spec_pat_3 = "((loop_guard_useful_for_post false) (num_preserved_term_vars ()))"

assert spec_matches(spec_pat=spec_pat_1, spec=spec_example)
assert spec_matches(spec_pat=spec_pat_2, spec=spec_example)
assert not spec_matches(spec_pat=spec_pat_3, spec=spec_example)

