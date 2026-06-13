from hlanalysis.backtest.data._nba_teams import (
    normalize_team,
    team_key,
    NBA_TEAMS,
)


def test_normalize_full_name_to_key():
    assert normalize_team("Boston Celtics") == "BOS"
    assert normalize_team("boston celtics") == "BOS"
    assert normalize_team("  Boston Celtics  ") == "BOS"


def test_normalize_abbrev_passthrough():
    assert normalize_team("BOS") == "BOS"
    assert normalize_team("bos") == "BOS"


def test_normalize_espn_short_name():
    # ESPN's "shortDisplayName" form
    assert normalize_team("Celtics") == "BOS"


def test_team_key_unknown_returns_none():
    assert normalize_team("Atlantis Sharks") is None


def test_nba_teams_has_30():
    assert len(NBA_TEAMS) == 30


def test_team_key_helper_combines_two_teams():
    # Used to build a (date, away, home) → game match key
    assert team_key("Boston Celtics", "Los Angeles Lakers") == ("BOS", "LAL")
