"""
Transformation tasks for parsing FPL API responses into typed data.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime



def parse_players_from_bootstrap(
    bootstrap_data: Dict[str, Any],
    gameweek_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Parse player data from bootstrap-static API response.
    
    Maps API 'elements' to our PLAYER_SCHEMA.
    
    Args:
        bootstrap_data: Response from fetch_fpl_endpoint (bootstrap-static)
        gameweek_id: Current gameweek ID for metadata
        
    Returns:
        List of player dictionaries matching PLAYER_SCHEMA
    """
    
    # Extract the 'elements' array (players) from API response
    api_players = bootstrap_data["data"]["elements"]
    
    parsed_players = []
    
    for api_player in api_players:
        # Map API fields to our schema fields
        player = {
            # Identity
            "player_id": api_player.get("id"),
            "web_name": api_player.get("web_name"),
            "first_name": api_player.get("first_name"),
            "second_name": api_player.get("second_name"),
            "team_id": api_player.get("team"),
            "position_id": api_player.get("element_type"),
            
            # Cost & Value
            "now_cost": api_player.get("now_cost"),
            "selected_by_percent": _safe_float(api_player.get("selected_by_percent")),
            "form": _safe_float(api_player.get("form")),
            "value_season": _safe_float(api_player.get("value_season")),
            "value_form": _safe_float(api_player.get("value_form")),
            
            # Core Stats
            "total_points": api_player.get("total_points"),
            "minutes": api_player.get("minutes"),
            "goals_scored": api_player.get("goals_scored"),
            "assists": api_player.get("assists"),
            "clean_sheets": api_player.get("clean_sheets"),
            "goals_conceded": api_player.get("goals_conceded"),
            "bonus": api_player.get("bonus"),
            "bps": api_player.get("bps"),
            
            # Status
            "status": api_player.get("status"),
            "chance_of_playing_next_round": api_player.get("chance_of_playing_next_round"),
            
            # Metadata
            "gameweek_fetched": gameweek_id,
        }
        
        parsed_players.append(player)
    
    print(f"Parsed {len(parsed_players)} players")
    return parsed_players


def parse_teams_from_bootstrap(
    bootstrap_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Parse team data from bootstrap-static API response.
    
    Maps API 'teams' to our TEAM_SCHEMA.
    
    Args:
        bootstrap_data: Response from fetch_fpl_endpoint (bootstrap-static)
        
    Returns:
        List of team dictionaries matching TEAM_SCHEMA
    """
    
    api_teams = bootstrap_data["data"]["teams"]
    
    parsed_teams = []
    
    for api_team in api_teams:
        team = {
            "team_id": api_team.get("id"),
            "name": api_team.get("name"),
            "short_name": api_team.get("short_name"),
            "strength": api_team.get("strength"),
            "form": _safe_float(api_team.get("form")),
            "points": api_team.get("points"),
            "position": api_team.get("position"),
            "strength_overall_home": api_team.get("strength_overall_home"),
            "strength_overall_away": api_team.get("strength_overall_away"),
            "strength_attack_home": api_team.get("strength_attack_home"),
            "strength_attack_away": api_team.get("strength_attack_away"),
            "strength_defence_home": api_team.get("strength_defence_home"),
            "strength_defence_away": api_team.get("strength_defence_away"),
        }
        
        parsed_teams.append(team)
    
    print(f"Parsed {len(parsed_teams)} teams")
    return parsed_teams



def parse_gameweeks_from_bootstrap(
    bootstrap_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Parse gameweek data from bootstrap-static API response.
    
    Maps API 'events' to our GAMEWEEK_SCHEMA.
    
    Args:
        bootstrap_data: Response from fetch_fpl_endpoint (bootstrap-static)
        
    Returns:
        List of gameweek dictionaries matching GAMEWEEK_SCHEMA
    """
    
    api_events = bootstrap_data["data"]["events"]
    
    parsed_gameweeks = []
    
    for api_event in api_events:
        gameweek = {
            "gameweek_id": api_event.get("id"),
            "name": api_event.get("name"),
            "deadline_time": _parse_timestamp(api_event.get("deadline_time")),
            "finished": api_event.get("finished"),
            "is_current": api_event.get("is_current"),
            "is_next": api_event.get("is_next"),
            "is_previous": api_event.get("is_previous"),
            "average_entry_score": api_event.get("average_entry_score"),
            "highest_score": api_event.get("highest_score"),
            "most_selected": api_event.get("most_selected"),
            "most_transferred_in": api_event.get("most_transferred_in"),
            "most_captained": api_event.get("most_captained"),
            "most_vice_captained": api_event.get("most_vice_captained"),
        }
        
        parsed_gameweeks.append(gameweek)
    
    print(f"Parsed {len(parsed_gameweeks)} gameweeks")
    return parsed_gameweeks


def parse_fixtures(
    fixtures_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Parse fixture data from fixtures API response.
    
    Maps API fixtures to our FIXTURE_SCHEMA.
    
    Args:
        fixtures_data: Response from fetch_fpl_endpoint (fixtures)
        
    Returns:
        List of fixture dictionaries matching FIXTURE_SCHEMA
    """
    
    api_fixtures = fixtures_data["data"]
    
    parsed_fixtures = []
    
    for api_fixture in api_fixtures:
        fixture = {
            "fixture_id": api_fixture.get("id"),
            "gameweek_id": api_fixture.get("event"),
            "kickoff_time": _parse_timestamp(api_fixture.get("kickoff_time")),
            "team_h": api_fixture.get("team_h"),
            "team_a": api_fixture.get("team_a"),
            "team_h_score": api_fixture.get("team_h_score"),
            "team_a_score": api_fixture.get("team_a_score"),
            "finished": api_fixture.get("finished"),
            "started": api_fixture.get("started"),
        }
        
        parsed_fixtures.append(fixture)
    
    print(f"Parsed {len(parsed_fixtures)} fixtures")
    return parsed_fixtures


# Helper functions for type conversion

def _safe_float(value: Any) -> Optional[float]:
    """
    Safely convert value to float, handling strings and None.
    
    Args:
        value: Value to convert (could be string, number, or None)
        
    Returns:
        Float value or None if conversion fails
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_timestamp(value: Any) -> Optional[str]:
    """
    Parse timestamp string to format suitable for Snowflake TIMESTAMP_NTZ.
    
    Args:
        value: Timestamp string from API
        
    Returns:
        Formatted timestamp string or None
    """
    if value is None or value == "":
        return None
    
    # FPL API returns ISO format timestamps - Snowflake handles these directly
    # Just return as-is, Snowflake will parse
    return value