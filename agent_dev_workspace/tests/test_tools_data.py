import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_tokenize_tool():
    # 1. Initialize session and load premade dataset tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load the fixture dataset
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # 2. Instantiate tools_data tools with the same session
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # 3. Invoke tokenize_tool
    res = tokenize_tool.invoke({})

    # Assert returned summary structure
    assert res["n_documents"] == 8
    assert res["total_tokens"] == 27

    # 4. Assert against Known Answer #10 in TEST_FIXTURE.md
    unigrams = session.artifacts["unigrams"]
    assert len(unigrams) == 8

    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    assert unigrams == expected_tokens


def test_list_files_tool():
    # 1. Initialize session
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # 2. Invoke list_files_tool on "newdataset" subdirectory
    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    # 3. Assert against Known Answer #11 in TEST_FIXTURE.md
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_inspect_data_tool():
    # 1. Initialize session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Populate unigrams column to test its exclusion in preview
    tokenize_tool.invoke({})

    # 2. Invoke inspect_data_tool with custom parameters
    res = inspect_data_tool.invoke({"n": 3, "max_text_length": 20})

    assert res["total_rows"] == 8
    assert len(res["head"]) == 3

    # Check row 0 details (load_dataset_tool stores label in category_name)
    row0 = res["head"][0]
    assert "unigrams" not in row0
    assert row0["text_length"] == 29
    assert row0["text"] == "always alpha alpha a..."
    assert row0["category_name"] == "catA"

    # 3. Test bounds checking (invalid n)
    err_res = inspect_data_tool.invoke({"n": -1})
    assert "error" in err_res
    assert "Parameter 'n' must be a positive integer" in err_res["error"]


def test_check_missing_tool():
    # 1. Initialize session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # 2. Invoke check_missing_tool
    res = check_missing_tool.invoke({})

    # 3. Assert against Known Answer #2 in TEST_FIXTURE.md
    assert res["total_documents"] == 8
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]
    assert res["has_missing"] is True


def test_check_duplicates_tool():
    # Fresh session for drop=False check
    session1 = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool1 = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool1.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools1 = make_tools(session1)
    check_duplicates_tool1 = next(t for t in data_tools1 if t.name == "check_duplicates_tool")

    # 1. Check duplicates with drop=False
    res_nodrop = check_duplicates_tool1.invoke({"drop": False})
    assert res_nodrop["initial_documents"] == 8
    assert res_nodrop["duplicate_count"] == 1
    assert res_nodrop["remaining_documents"] == 8

    # Fresh session for drop=True check to isolate dataframe state
    session2 = SessionState()
    premade_tools2, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool2 = next(t for t in premade_tools2 if t.name == "load_dataset_tool")

    load_dataset_tool2.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools2 = make_tools(session2)
    check_duplicates_tool2 = next(t for t in data_tools2 if t.name == "check_duplicates_tool")

    # 2. Check duplicates with drop=True (assert keep=False removes both copies, leaving 6)
    res_drop = check_duplicates_tool2.invoke({"drop": True})
    assert res_drop["initial_documents"] == 8
    assert res_drop["duplicate_count"] == 1
    assert res_drop["remaining_documents"] == 6
    assert len(session2.dataframe) == 6


def test_sample_data_tool():
    # Initialize session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # Invoke sample_data_tool with n=4, random_state=42
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    # Assert against Known Answer #13 in TEST_FIXTURE.md
    assert res["initial_documents"] == 8
    assert res["sampled_documents"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4


def test_describe_data_tool():
    # Initialize session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    # Invoke describe_data_tool
    res = describe_data_tool.invoke({})

    # Assert overall stats against Known Answer #18 in TEST_FIXTURE.md
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30

    # Assert catA stats
    catA = res["per_category_stats"]["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == 0
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == 29

    # Assert catB stats
    catB = res["per_category_stats"]["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == 18
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == 30

    # Assert plotting requirement
    assert session.pending_figure is not None
