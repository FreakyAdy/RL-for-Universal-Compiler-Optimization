"""Dataset schema tests."""

from rl_uco.data.schema import DatasetRow, rows_to_dataframe


def test_row_to_dataframe():
    row = DatasetRow(
        function_id="fn_1",
        ir_kind="llvm",
        graph_path="g.pt",
        isa="x86_64_v3",
        pass_sequence=[{"pass_id": 1, "name": "instcombine", "pipeline": "instcombine", "kind": "transform"}],
        wall_time_ns=100.0,
        energy_j=1.0,
        baseline_wall_time_ns=200.0,
        baseline_energy_j=2.0,
        reward=-0.75,
        correct=True,
    )
    df = rows_to_dataframe([row])
    assert len(df) == 1
    assert df.iloc[0]["function_id"] == "fn_1"
