from tasks.optimizer_tasks import add_recommendation_metadata


def test_add_recommendation_metadata_sets_fields():
    records = [{"player_id": 1}, {"player_id": 2}]
    result = add_recommendation_metadata(records, recommended_at="2026-01-19T00:00:00")

    assert result[0]["recommended_at"] == "2026-01-19T00:00:00"
    assert result[0]["recommendation_key"] == "1_2026-01-19T00:00:00"
    assert result[1]["recommendation_key"] == "2_2026-01-19T00:00:00"
