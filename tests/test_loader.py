from loader import DATA_DIR, load_all_datasets


def test_data_dir_exists():
    assert DATA_DIR.is_dir(), f"Missing data dir: {DATA_DIR}"


def test_load_all_has_expected_count():
    rows = load_all_datasets()
    assert len(rows) == 30, f"expected 30 prompts (3×10), got {len(rows)}"


def test_each_row_has_required_keys():
    for row in load_all_datasets():
        for key in ("id", "type", "category", "prompt", "expected"):
            assert key in row
        assert row["expected"] in ("refuse", "answer")
