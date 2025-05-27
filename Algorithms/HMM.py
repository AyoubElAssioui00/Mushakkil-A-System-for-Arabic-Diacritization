import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set
import pickle
import os
from Data.HMM import load_pickle



class DiacriticHMM:
    def __init__(self):
        arabic_letters = load_pickle('constants/ARABIC_LETTERS_LIST.pickle')
        self.diacritics = {
            "",        # No diacritic (required for spaces and some letters)
            arabic_letters[0],        # Fatha
            arabic_letters[1],        # Kasra  
            arabic_letters[2],        # Damma
            arabic_letters[3],        # Sukun
            arabic_letters[4],        # Tanween Fath
            arabic_letters[5],        # Tanween Kasr
            arabic_letters[6],        # Tanween Damm
            arabic_letters[7],        # Shadda
            arabic_letters[8],
            arabic_letters[9],
            arabic_letters[10],
            arabic_letters[11],
            arabic_letters[12],
            arabic_letters[13],
            # Add more diacritics as needed
        }
        
        # Constraint: Space character must always have empty diacritic
        self.space_constraint = (" ", "")
        
        # Model parameters
        self.states = set()           # Set of all diacritics (hidden states)
        self.observations = set()     # Set of all letters/characters
        self.initial_probs = {}       # π: Initial state probabilities
        self.transition_probs = {}    # A: Transition probabilities 
        self.emission_probs = {}      # B: Emission probabilities
        
        # Smoothing parameter for unseen observations
        self.smoothing = 1e-10
        
    def train(self, training_data: List[List[Tuple[str, str]]]):
        """
        Train HMM parameters from supervised data using MLE
        
        Args:
            training_data: List of sequences, each sequence is [(observation, hidden_state), ...]
        """
        print("Training HMM model...")
        
        # Extract all states and observations
        for sequence in training_data:
            for obs, state in sequence:
                self.states.add(state)
                self.observations.add(obs)
        
        # Ensure we have the constraint diacritics
        self.states.update(self.diacritics)
        
        # Initialize counters
        initial_counts = Counter()
        transition_counts = defaultdict(Counter)
        emission_counts = defaultdict(Counter)
        state_counts = Counter()
        
        # Count occurrences
        for sequence in training_data:
            if not sequence:
                continue
                
            # Count initial state
            initial_counts[sequence[0][1]] += 1
            
            # Count transitions and emissions
            for i, (obs, state) in enumerate(sequence):
                # Count emission
                emission_counts[state][obs] += 1
                state_counts[state] += 1
                
                # Count transition (if not last position)
                if i < len(sequence) - 1:
                    next_state = sequence[i + 1][1]
                    transition_counts[state][next_state] += 1
        
        # Calculate probabilities with smoothing
        total_sequences = len(training_data)
        
        # Initial probabilities
        self.initial_probs = {}
        for state in self.states:
            self.initial_probs[state] = (initial_counts[state] + self.smoothing) / (total_sequences + len(self.states) * self.smoothing)
        
        # Transition probabilities
        self.transition_probs = {}
        for state in self.states:
            self.transition_probs[state] = {}
            total_transitions = sum(transition_counts[state].values()) + len(self.states) * self.smoothing
            for next_state in self.states:
                self.transition_probs[state][next_state] = (transition_counts[state][next_state] + self.smoothing) / total_transitions
        
        # Emission probabilities with constraints
        self.emission_probs = {}
        for state in self.states:
            self.emission_probs[state] = {}
            
            # Handle space constraint
            if state == "":  # No diacritic state
                for obs in self.observations:
                    if obs == " ":
                        self.emission_probs[state][obs] = 1.0  # Space must have no diacritic
                    else:
                        # Normal probability for other observations
                        total_emissions = state_counts[state] + len(self.observations) * self.smoothing
                        self.emission_probs[state][obs] = (emission_counts[state][obs] + self.smoothing) / total_emissions
            else:  # Other diacritic states
                for obs in self.observations:
                    if obs == " ":
                        self.emission_probs[state][obs] = self.smoothing  # Very small probability for space
                    else:
                        # Normal probability for other observations
                        total_emissions = state_counts[state] + len(self.observations) * self.smoothing
                        self.emission_probs[state][obs] = (emission_counts[state][obs] + self.smoothing) / total_emissions
        
        print(f"Training completed. States: {len(self.states)}, Observations: {len(self.observations)}")
    
    def save_model(self, filepath: str):
        """
        Save the trained HMM model to disk
        
        Args:
            filepath: Path where to save the model (e.g., 'models/hmm_diacritization.pkl')
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save all model parameters
        model_data = {
            'diacritics': self.diacritics,
            'space_constraint': self.space_constraint,
            'states': self.states,
            'observations': self.observations,
            'initial_probs': self.initial_probs,
            'transition_probs': self.transition_probs,
            'emission_probs': self.emission_probs,
            'smoothing': self.smoothing
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained HMM model from disk
        
        Args:
            filepath: Path to the saved model file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore all model parameters
        self.diacritics = model_data['diacritics']
        self.space_constraint = model_data['space_constraint']
        self.states = model_data['states']
        self.observations = model_data['observations']
        self.initial_probs = model_data['initial_probs']
        self.transition_probs = model_data['transition_probs']
        self.emission_probs = model_data['emission_probs']
        self.smoothing = model_data['smoothing']
        
        print(f"Model loaded from: {filepath}")
        print(f"Model info - States: {len(self.states)}, Observations: {len(self.observations)}")
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """
        Class method to create and load HMM model from file in one step
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            DiacriticHMM: Loaded model instance
        """
        hmm = cls()
        hmm.load_model(filepath)
        return hmm
    
    def viterbi_decode(self, observations: List[str]) -> List[Tuple[str, str]]:
        """
        Use Viterbi algorithm to find most likely sequence of hidden states
        
        Args:
            observations: List of characters/letters to predict diacritics for
            
        Returns:
            List of (observation, predicted_state) tuples
        """
        if not observations:
            return []
        
        n_obs = len(observations)
        states_list = list(self.states)
        n_states = len(states_list)
        
        # Initialize Viterbi tables
        viterbi = np.zeros((n_states, n_obs))
        path = np.zeros((n_states, n_obs), dtype=int)
        
        # Handle unseen observations
        def get_emission_prob(state, obs):
            if obs in self.emission_probs[state]:
                return self.emission_probs[state][obs]
            else:
                # Unseen observation - use uniform smoothing
                return self.smoothing
        
        # Initialize first column
        for i, state in enumerate(states_list):
            obs = observations[0]
            
            # Apply space constraint
            if obs == " " and state != "":
                viterbi[i, 0] = -np.inf  # Impossible
            elif obs == " " and state == "":
                viterbi[i, 0] = np.log(self.initial_probs[state]) + np.log(1.0)  # Certain
            else:
                emission_prob = get_emission_prob(state, obs)
                if emission_prob > 0:
                    viterbi[i, 0] = np.log(self.initial_probs[state]) + np.log(emission_prob)
                else:
                    viterbi[i, 0] = -np.inf
        
        # Fill the rest of the table
        for t in range(1, n_obs):
            obs = observations[t]
            
            for j, curr_state in enumerate(states_list):
                # Apply space constraint
                if obs == " " and curr_state != "":
                    viterbi[j, t] = -np.inf
                    path[j, t] = 0
                    continue
                elif obs == " " and curr_state == "":
                    # For space with no diacritic, find best previous state
                    max_prob = -np.inf
                    max_prev = 0
                    for i, prev_state in enumerate(states_list):
                        if viterbi[i, t-1] == -np.inf:
                            continue
                        prob = viterbi[i, t-1] + np.log(self.transition_probs[prev_state][curr_state]) + np.log(1.0)
                        if prob > max_prob:
                            max_prob = prob
                            max_prev = i
                    viterbi[j, t] = max_prob
                    path[j, t] = max_prev
                    continue
                
                # Normal case
                emission_prob = get_emission_prob(curr_state, obs)
                if emission_prob <= 0:
                    viterbi[j, t] = -np.inf
                    path[j, t] = 0
                    continue
                
                log_emission = np.log(emission_prob)
                max_prob = -np.inf
                max_prev = 0
                
                for i, prev_state in enumerate(states_list):
                    if viterbi[i, t-1] == -np.inf:
                        continue
                    prob = viterbi[i, t-1] + np.log(self.transition_probs[prev_state][curr_state]) + log_emission
                    if prob > max_prob:
                        max_prob = prob
                        max_prev = i
                
                viterbi[j, t] = max_prob
                path[j, t] = max_prev
        
        # Backtrack to find best path
        best_path = []
        
        # Find best final state
        last_col = viterbi[:, -1]
        best_last_state = np.argmax(last_col)
        
        # If all probabilities are -inf, choose the empty diacritic state as fallback
        if np.all(last_col == -np.inf):
            best_last_state = states_list.index("") if "" in states_list else 0
        
        # Reconstruct path
        states_path = [best_last_state]
        for t in range(n_obs - 1, 0, -1):
            best_last_state = path[best_last_state, t]
            states_path.append(best_last_state)
        
        states_path.reverse()
        
        # Convert to output format
        result = []
        for i, obs in enumerate(observations):
            state_idx = states_path[i]
            predicted_state = states_list[state_idx]
            result.append((obs, predicted_state))
        
        return result
    
    def predict(self, test_sequences: List[List[str]]) -> List[List[Tuple[str, str]]]:
        """
        Predict diacritics for multiple test sequences
        
        Args:
            test_sequences: List of sequences, each sequence is a list of characters
            
        Returns:
            List of predicted sequences in format [(observation, predicted_state), ...]
        """
        predictions = []
        
        for sequence in test_sequences:
            prediction = self.viterbi_decode(sequence)
            predictions.append(prediction)
        
        return predictions
    
    def calculate_diacritic_error_rate(self, predictions: List[List[Tuple[str, str]]], 
                                     ground_truth: List[List[Tuple[str, str]]]) -> float:
        """
        Calculate Diacritic Error Rate (DER) - percentage of incorrectly predicted diacritics
        
        Args:
            predictions: Predicted sequences [(obs, predicted_state), ...]
            ground_truth: True sequences [(obs, true_state), ...]
            
        Returns:
            DER as a float between 0 and 1
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Number of predicted and ground truth sequences must match")
        
        total_characters = 0
        total_errors = 0
        
        for pred_seq, true_seq in zip(predictions, ground_truth):
            if len(pred_seq) != len(true_seq):
                raise ValueError("Predicted and ground truth sequences must have same length")
            
            for (pred_obs, pred_state), (true_obs, true_state) in zip(pred_seq, true_seq):
                if pred_obs != true_obs:
                    raise ValueError("Observations don't match between prediction and ground truth")
                
                total_characters += 1
                if pred_state != true_state:
                    total_errors += 1
        
        if total_characters == 0:
            return 0.0
        
        der = total_errors / total_characters
        return der

