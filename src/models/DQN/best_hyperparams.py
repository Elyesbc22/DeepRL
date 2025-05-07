import os
import json
from typing import Dict, Any, Optional, Union, List
import numpy as np

# Path to the file storing the best hyperparameters
BEST_HYPERPARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_hyperparams.json")

def get_best_hyperparams(env_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the best hyperparameters for a given environment.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        Dictionary of best hyperparameters if available, None otherwise
    """
    if not os.path.exists(BEST_HYPERPARAMS_FILE):
        return None
    
    try:
        with open(BEST_HYPERPARAMS_FILE, 'r') as f:
            all_hyperparams = json.load(f)
        
        return all_hyperparams.get(env_name)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def update_best_hyperparams(env_name: str, hyperparams: Dict[str, Any], 
                           eval_reward: float) -> bool:
    """
    Update the best hyperparameters for a given environment if the current
    evaluation reward is better than the previous best.
    
    Args:
        env_name: Name of the environment
        hyperparams: Current hyperparameters
        eval_reward: Current evaluation reward
        
    Returns:
        True if hyperparameters were updated, False otherwise
    """
    # Load existing hyperparameters
    all_hyperparams = {}
    if os.path.exists(BEST_HYPERPARAMS_FILE):
        try:
            with open(BEST_HYPERPARAMS_FILE, 'r') as f:
                all_hyperparams = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # Check if we should update
    should_update = False
    if env_name not in all_hyperparams:
        should_update = True
    elif eval_reward > all_hyperparams[env_name].get("eval_reward", float('-inf')):
        should_update = True
    
    # Update if needed
    if should_update:
        # Create a copy of hyperparams to avoid modifying the original
        hyperparams_copy = hyperparams.copy()
        # Add the evaluation reward
        hyperparams_copy["eval_reward"] = eval_reward
        # Update the dictionary
        all_hyperparams[env_name] = hyperparams_copy
        
        # Save to file
        os.makedirs(os.path.dirname(BEST_HYPERPARAMS_FILE), exist_ok=True)
        with open(BEST_HYPERPARAMS_FILE, 'w') as f:
            json.dump(all_hyperparams, f, indent=2)
        
        return True
    
    return False