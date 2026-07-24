from camera_operator_sr.pipeline.stages import DEPENDENCIES, STAGES, downstream, topological_order


def test_graph_is_acyclic_and_topological():
    order = topological_order(); assert len(order) == len(set(order))
    for stage, dependencies in DEPENDENCIES.items(): assert all(order.index(dep) < order.index(stage) for dep in dependencies)
    assert "P12_summary" in downstream("P06_train_teacher_correct")
