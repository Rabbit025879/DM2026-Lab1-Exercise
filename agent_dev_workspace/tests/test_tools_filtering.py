import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    # 1. Initialize session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # 2. Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Instantiate filtering tools and run variance_filter_tool with threshold=0.15
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    res = variance_filter_tool.invoke({"threshold": 0.15})

    # 4. Assert against Known Answer #5 in TEST_FIXTURE.md
    assert res["total_features"] == 5
    assert res["kept_features_count"] == 4
    assert res["removed_features_count"] == 1

    # Check terms kept and removed
    assert set(res["kept_terms"]) == {"alpha", "gamma", "delta", "beta"}
    assert res["removed_terms"] == ["always"]

    # Verify full_report stored in session.results_store
    stored_result = session.results_store[res["result_id"]]
    report_df = stored_result["full_report"]

    assert list(report_df.columns)[:2] == ["term", "variance"]
    
    variances_dict = dict(zip(report_df["term"], report_df["variance"]))
    assert variances_dict["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert variances_dict["gamma"] == pytest.approx(1.109375, abs=1e-3)
    assert variances_dict["delta"] == pytest.approx(0.25, abs=1e-3)
    assert variances_dict["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert variances_dict["always"] == pytest.approx(0.109375, abs=1e-3)


def test_pearson_filter_tool():
    # 1. Initialize session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # 2. Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Instantiate filtering tools and run pearson_filter_tool for target_class="catB"
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    # 4. Assert against Known Answer #6 in TEST_FIXTURE.md
    assert res["target_class"] == "catB"
    assert res["total_features"] == 5

    correlations = res["correlations"]
    assert correlations["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert correlations["gamma"] == pytest.approx(0.8307, abs=1e-3)
    assert correlations["always"] == pytest.approx(0.3780, abs=1e-3)
    assert correlations["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert correlations["beta"] == pytest.approx(-0.7746, abs=1e-3)

    # Verify full_report column structure
    stored_result = session.results_store[res["result_id"]]
    report_df = stored_result["full_report"]
    assert list(report_df.columns)[:2] == ["term", "pearson_r"]


def test_spearman_filter_tool():
    # 1. Initialize session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # 2. Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Instantiate filtering tools and run spearman_filter_tool for target_class="catB"
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    # 4. Assert against Known Answer #6 in TEST_FIXTURE.md
    assert res["target_class"] == "catB"
    assert res["total_features"] == 5

    correlations = res["correlations"]
    assert correlations["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert correlations["gamma"] == pytest.approx(0.9363, abs=1e-3)
    assert correlations["always"] == pytest.approx(0.3780, abs=1e-3)
    assert correlations["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert correlations["beta"] == pytest.approx(-0.7746, abs=1e-3)

    # Verify full_report column structure
    stored_result = session.results_store[res["result_id"]]
    report_df = stored_result["full_report"]
    assert list(report_df.columns)[:2] == ["term", "spearman_r"]
