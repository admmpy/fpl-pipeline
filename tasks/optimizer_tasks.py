"""
Tasks for squad optimization using cvxpy.
"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import logging
import cvxpy as cp
import numpy as np
import pandas as pd
from prefect import task, get_run_logger
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPTIMIZATION_CONFIG

logger = logging.getLogger(__name__)

class SquadOptimizer:
    """Optimizes FPL squad selection using integer programming."""
    
    def __init__(self, config: dict):
        """Initialize with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.squad_constraints = config["optimization"]["squad_constraints"]
        self.starting_constraints = config["optimization"]["starting_constraints"]
        self.solver = config["optimization"]["solver"]
        self.budget = config["optimization"]["budget"]
        
    def _create_variables(self, n_players: int) -> Tuple[cp.Variable, cp.Variable,
                                                        cp.Variable, cp.Variable]:
        """Create optimization variables.
        
        Args:
            n_players: Number of players to consider
            
        Returns:
            Tuple of (squad, starting, captain, vice_captain) binary variables
        """
        # Binary variables for squad selection
        squad = cp.Variable(n_players, boolean=True)
        
        # Binary variables for starting XI
        starting = cp.Variable(n_players, boolean=True)
        
        # Binary variables for captain and vice
        captain = cp.Variable(n_players, boolean=True)
        vice_captain = cp.Variable(n_players, boolean=True)
        
        return squad, starting, captain, vice_captain
    
    def _get_squad_constraints(self, squad: cp.Variable,
                             players_df: pd.DataFrame) -> List[cp.Constraint]:
        """Get constraints for valid 15-player squad.
        
        Args:
            squad: Binary variables for squad selection
            players_df: DataFrame with player information
        """
        constraints = []
        
        # Total players constraint
        constraints.append(cp.sum(squad) == self.squad_constraints["total_players"])
        
        # Position constraints
        for pos, count in [
            (1, self.squad_constraints["goalkeeper_count"]),
            (2, self.squad_constraints["defender_count"]),
            (3, self.squad_constraints["midfielder_count"]),
            (4, self.squad_constraints["forward_count"])
        ]:
            pos_players = (players_df["position_id"] == pos)
            constraints.append(cp.sum(squad[pos_players]) == count)
            
        # Team constraint (max 3 per team)
        for team in players_df["team_id"].unique():
            team_players = (players_df["team_id"] == team)
            constraints.append(
                cp.sum(squad[team_players]) <= self.squad_constraints["max_per_team"]
            )
            
        # Budget constraint
        prices = players_df["now_cost"].values # now_cost is already £m in our dbt mart
        constraints.append(cp.sum(cp.multiply(prices, squad)) <= self.budget)
        
        return constraints
    
    def _get_starting_constraints(self, squad: cp.Variable,
                                starting: cp.Variable,
                                captain: cp.Variable,
                                vice_captain: cp.Variable,
                                players_df: pd.DataFrame) -> List[cp.Constraint]:
        """Get constraints for valid starting XI selection.
        
        Args:
            squad: Binary variables for squad selection
            starting: Binary variables for starting XI
            captain: Binary variables for captain
            vice_captain: Binary variables for vice-captain
            players_df: DataFrame with player information
        """
        constraints = []
        
        # Starting XI size constraint
        constraints.append(cp.sum(starting) == self.starting_constraints["total_players"])
        
        # Can only start players in squad
        constraints.append(starting <= squad)
        
        # Formation constraints
        for pos, min_count in [
            (1, self.starting_constraints["min_goalkeeper"]),
            (2, self.starting_constraints["min_defender"]),
            (3, self.starting_constraints["min_midfielder"]),
            (4, self.starting_constraints["min_forward"])
        ]:
            pos_players = (players_df["position_id"] == pos)
            constraints.append(cp.sum(starting[pos_players]) >= min_count)
            
        # Captain constraints
        constraints.append(cp.sum(captain) == 1)  # Exactly one captain
        constraints.append(cp.sum(vice_captain) == 1)  # Exactly one vice
        constraints.append(captain + vice_captain <= 1)  # Can't be both
        constraints.append(captain <= starting)  # Captain must start
        constraints.append(vice_captain <= starting)  # Vice must start
        
        return constraints
    
    def _get_transfer_penalty(self, squad: cp.Variable,
                                current_squad: List[int],
                                players_df: pd.DataFrame,
                                constraints: List[cp.Constraint]) -> cp.Expression:
        """Get transfer penalty and add limit constraint.
        
        Args:
            squad: Binary variables for squad selection
            current_squad: List of current squad player IDs
            players_df: DataFrame with player information
            constraints: List to append limit constraint to
        """
        current_squad_mask = players_df["player_id"].isin(current_squad)
        current_squad_vector = current_squad_mask.astype(int).values
        
        # Calculate number of transfers
        transfers = cp.sum(cp.abs(squad - current_squad_vector)) / 2
        
        # Add transfer limit constraint
        constraints.append(transfers <= self.config["optimization"]["max_transfers"])
        
        # Return transfer cost expression
        return transfers * self.config["optimization"]["transfer_penalty"]
    
    def optimize(self, players_df: pd.DataFrame,
                predicted_points: np.ndarray,
                current_squad: Optional[List[int]] = None) -> Dict:
        """Optimize squad selection.
        
        Args:
            players_df: DataFrame with player information
            predicted_points: Array of predicted points for each player
            current_squad: Optional list of current squad player IDs for transfer mode
            
        Returns:
            Dictionary with optimization results
        """
        n_players = len(players_df)
        
        # Create variables
        squad, starting, captain, vice_captain = self._create_variables(n_players)
        
        # Initialize constraints list
        constraints = []
        
        # Add squad constraints
        constraints.extend(self._get_squad_constraints(squad, players_df))
        
        # Add starting XI constraints
        constraints.extend(self._get_starting_constraints(
            squad, starting, captain, vice_captain, players_df))
        
        # Calculate objective components
        expected_points = (cp.sum(cp.multiply(predicted_points, starting)) +
                         cp.sum(cp.multiply(predicted_points, captain)))  # Captain doubles points
        
        # Add transfer penalty if in transfer mode
        transfer_penalty_expr = (
            self._get_transfer_penalty(squad, current_squad, players_df, constraints)
            if current_squad is not None else 0
        )
        
        # Define problem
        objective = cp.Maximize(expected_points - transfer_penalty_expr)
        prob = cp.Problem(objective, constraints)
        
        # Solve
        logger.info(f"Solving optimization problem with {self.solver}")
        prob.solve(solver=self.solver)
        
        if prob.status != "optimal":
            raise ValueError(f"Problem could not be solved: {prob.status}")
            
        # Extract results
        selected_squad_mask = (squad.value > 0.5)
        starting_xi_mask = (starting.value > 0.5)
        captain_mask = (captain.value > 0.5)
        vice_captain_mask = (vice_captain.value > 0.5)
        
        players_df['is_in_squad'] = selected_squad_mask
        players_df['is_starter'] = starting_xi_mask
        players_df['is_captain'] = captain_mask
        players_df['is_vice_captain'] = vice_captain_mask
        
        selected_squad = players_df[selected_squad_mask].copy()
        starting_xi = players_df[starting_xi_mask].copy()
        captain_player = players_df[captain_mask].iloc[0]
        vice_captain_player = players_df[vice_captain_mask].iloc[0]
        
        # Calculate bench order by predicted points
        bench = selected_squad[~selected_squad["player_id"].isin(starting_xi["player_id"])]
        bench = bench.sort_values("expected_points_next_gw", ascending=False)
        
        result = {
            "squad": selected_squad.to_dict(orient="records"),
            "starting_xi": starting_xi.to_dict(orient="records"),
            "captain": captain_player.to_dict(),
            "vice_captain": vice_captain_player.to_dict(),
            "bench": bench.to_dict(orient="records"),
            "expected_points": float(prob.value),
            "solver_status": prob.status,
        }
        
        if current_squad is not None:
            # We need to evaluate the expression to get the value
            # but transfer_penalty_expr is only available during solve
            # and transfers variable is not explicitly saved.
            # We can re-calculate or just omit if complex.
            # For now, let's just use the prob.value which includes the penalty.
            pass
            
        return result

@task
def optimize_squad_task(predictions: List[Dict[str, Any]], current_squad: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Prefect task wrapper for the SquadOptimizer.
    """
    logger = get_run_logger()
    
    df = pd.DataFrame(predictions)
    
    optimizer = SquadOptimizer(OPTIMIZATION_CONFIG)
    
    results = optimizer.optimize(
        players_df=df,
        predicted_points=df['expected_points_next_gw'].values,
        current_squad=current_squad
    )
    
    # Format the results into a flat list of records for Snowflake insertion
    # We only want the 15 players in the recommended squad
    squad_records = results['squad']
    
    # Add metadata
    for record in squad_records:
        record['expected_points_5_gw'] = record['expected_points_next_gw'] * 5 # Simplified
        record['recommendation_at'] = pd.Timestamp.now().isoformat()
        
    logger.info(f"Optimization complete. Total Expected Points: {results['expected_points']:.2f}")
    
    return squad_records
