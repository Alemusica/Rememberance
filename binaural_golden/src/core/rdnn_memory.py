"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RDNN MEMORY - Recurrent Deep Neural Network for Evolution          ║
║                                                                              ║
║   PyTorch-based short-term memory with hidden state persistence:             ║
║   • LSTM/GRU cells maintain context across generations                       ║
║   • Hidden state persists across optimization runs (warm start)              ║
║   • Learns fitness landscape patterns, not just trajectories                 ║
║                                                                              ║
║   RESEARCH BASIS:                                                            ║
║   • Yu et al. 2023: Experience-Based Surrogate-Assisted EA (SAEA)            ║
║   • Jiwatode 2024: RHEA Continual Learning with hidden state                 ║
║   • Hochreiter & Schmidhuber 1997: LSTM for long-term dependencies           ║
║                                                                              ║
║   KEY INSIGHT (from gap analysis):                                           ║
║   Our ShortTermMemory uses ring buffers (stateless between runs).            ║
║   This module adds RECURRENT state that persists across runs,                ║
║   allowing the optimizer to "remember" successful search patterns.           ║
║                                                                              ║
║   INTEGRATION:                                                               ║
║   - Inputs from PokayokeObserver (anomaly features)                          ║
║   - Inputs from PhysicsRulesEngine (rule satisfaction scores)                ║
║   - Outputs: Search direction hints, mutation rate suggestions               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
import json
import pickle
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class RDNNArchitecture(Enum):
    """Available recurrent architectures."""
    LSTM = auto()      # Long Short-Term Memory (better for long sequences)
    GRU = auto()       # Gated Recurrent Unit (faster, often comparable)
    VANILLA = auto()   # Simple RNN (baseline, not recommended)


@dataclass
class RDNNConfig:
    """
    Configuration for RDNN Memory module.
    
    PAPER REFERENCES:
    - hidden_size=64: Yu et al. 2023 used 64-128 for fitness prediction
    - num_layers=2: Jiwatode 2024 found 2 layers sufficient
    - dropout=0.1: Prevents overfitting on small optimization datasets
    """
    # Architecture
    architecture: RDNNArchitecture = RDNNArchitecture.GRU
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False  # Causal prediction doesn't benefit from bidirectional
    
    # Input features (computed from observation)
    input_features: int = 32  # Will be computed dynamically
    
    # Output heads
    output_mutation_rate: bool = True      # Suggest adaptive mutation
    output_search_direction: bool = True   # Suggest parameter changes
    output_fitness_prediction: bool = True  # Predict next fitness
    output_anomaly_probability: bool = True  # Predict anomaly likelihood
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # State persistence
    persist_hidden_state: bool = True  # KEY FEATURE: warm start
    state_file: Optional[str] = None
    
    # Device
    device: str = "cpu"  # "cuda" if available


@dataclass
class RDNNObservation:
    """
    Single observation fed to RDNN at each generation.
    
    Combines signals from:
    - Fitness landscape (current population stats)
    - PokayokeObserver (anomaly detection)
    - PhysicsRulesEngine (rule satisfaction)
    - Operator stats (mutation/crossover success)
    """
    generation: int
    
    # Fitness features
    best_fitness: float
    mean_fitness: float
    fitness_std: float
    fitness_velocity: float  # Improvement since last gen
    
    # Diversity features
    population_diversity: float
    phenotype_diversity: float
    
    # Objective-specific (multi-objective)
    spine_flatness: float = 0.0
    ear_uniformity: float = 0.0
    total_energy: float = 0.0
    
    # Physics rule satisfaction (from PhysicsRulesEngine)
    antinode_score: float = 0.0
    node_avoidance_score: float = 0.0
    edge_distance_score: float = 0.0
    phase_coherence_score: float = 0.0
    
    # Anomaly signals (from PokayokeObserver)
    stagnation_signal: float = 0.0
    diversity_collapse_signal: float = 0.0
    regression_signal: float = 0.0
    
    # Operator effectiveness
    mutation_success_rate: float = 0.0
    crossover_success_rate: float = 0.0
    
    # Gene activation state
    emission_genes_active: float = 0.0  # 0=SEED, 0.5=BLOOM, 1=FREEZE
    
    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert observation to feature tensor."""
        features = [
            # Normalized fitness features
            self.best_fitness,
            self.mean_fitness,
            min(self.fitness_std, 1.0),  # Clipped
            np.clip(self.fitness_velocity, -1, 1),
            
            # Diversity
            self.population_diversity,
            self.phenotype_diversity,
            
            # Objectives
            self.spine_flatness,
            self.ear_uniformity,
            self.total_energy,
            
            # Physics
            self.antinode_score,
            self.node_avoidance_score,
            self.edge_distance_score,
            self.phase_coherence_score,
            
            # Anomalies
            self.stagnation_signal,
            self.diversity_collapse_signal,
            self.regression_signal,
            
            # Operators
            self.mutation_success_rate,
            self.crossover_success_rate,
            
            # Gene state
            self.emission_genes_active,
            
            # Generation context (normalized)
            min(self.generation / 100, 1.0),
        ]
        
        return torch.tensor(features, dtype=torch.float32, device=device)


@dataclass
class RDNNPrediction:
    """
    RDNN output predictions for next generation.
    
    Used by evolutionary optimizer to adapt parameters.
    """
    # Suggested mutation rate [0, 1]
    suggested_mutation_rate: float = 0.2
    
    # Search direction hints (where to focus)
    focus_spine: float = 0.5      # 0=ignore, 1=prioritize
    focus_ears: float = 0.5
    focus_energy: float = 0.5
    
    # Predicted next fitness (for surrogate assist)
    predicted_fitness: float = 0.0
    prediction_confidence: float = 0.5
    
    # Anomaly warning
    anomaly_probability: float = 0.0
    likely_anomaly_type: str = "none"
    
    # Hidden state info (for debugging)
    hidden_state_norm: float = 0.0
    
    # Recommended action
    recommended_action: str = "continue"


# ══════════════════════════════════════════════════════════════════════════════
# RDNN MODEL (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

class RDNNModel(nn.Module):
    """
    Recurrent neural network for evolution guidance.
    
    Architecture:
        Input (observation features)
            ↓
        [LSTM/GRU layers] ← Hidden state persists across runs
            ↓
        Multi-head outputs:
        ├── Mutation rate head
        ├── Search direction head
        ├── Fitness prediction head
        └── Anomaly probability head
    
    The KEY innovation is that hidden state is saved and loaded
    between optimization runs, providing "experience transfer".
    """
    
    def __init__(self, config: RDNNConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(
            20,  # Number of features in RDNNObservation
            config.hidden_size
        )
        
        # Recurrent layers
        if config.architecture == RDNNArchitecture.LSTM:
            self.rnn = nn.LSTM(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True,
            )
        elif config.architecture == RDNNArchitecture.GRU:
            self.rnn = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = nn.RNN(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True,
            )
        
        # Output dimension
        out_dim = config.hidden_size * (2 if config.bidirectional else 1)
        
        # Output heads
        self.mutation_head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # [spine, ears, energy]
            nn.Softmax(dim=-1),  # Focus distribution
        )
        
        self.fitness_head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # [predicted_fitness, confidence]
        )
        
        self.anomaly_head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # [stagnation, diversity, regression, none]
            nn.Softmax(dim=-1),
        )
        
        # Hidden state storage
        self._hidden_state: Optional[Tuple[torch.Tensor, ...]] = None
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """
        Forward pass through RDNN.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            hidden: Optional hidden state from previous run
        
        Returns:
            outputs: Dict of output tensors
            new_hidden: Updated hidden state
        """
        # Input projection
        x = self.input_proj(x)
        
        # RNN forward
        if hidden is not None:
            rnn_out, new_hidden = self.rnn(x, hidden)
        else:
            rnn_out, new_hidden = self.rnn(x)
        
        # Use last timestep for predictions
        last_out = rnn_out[:, -1, :]  # (batch, hidden_size)
        
        # Output heads
        outputs = {
            "mutation_rate": self.mutation_head(last_out),
            "direction": self.direction_head(last_out),
            "fitness": self.fitness_head(last_out),
            "anomaly": self.anomaly_head(last_out),
        }
        
        return outputs, new_hidden
    
    def get_hidden_state(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """Get current hidden state for persistence."""
        return self._hidden_state
    
    def set_hidden_state(self, hidden: Optional[Tuple[torch.Tensor, ...]]):
        """Set hidden state (for warm start)."""
        self._hidden_state = hidden


# ══════════════════════════════════════════════════════════════════════════════
# RDNN MEMORY (Main Interface)
# ══════════════════════════════════════════════════════════════════════════════

class RDNNMemory:
    """
    RDNN Memory system with hidden state persistence.
    
    This is the main class to use in the optimizer. It:
    1. Maintains observation history
    2. Runs RDNN inference
    3. Persists hidden state across runs
    4. Learns from successful optimizations
    
    USAGE:
        # Initialize (loads persisted state if available)
        rdnn = RDNNMemory(config, state_path="./rdnn_state")
        
        # Each generation
        obs = RDNNObservation(generation=gen, best_fitness=0.75, ...)
        prediction = rdnn.step(obs)
        
        # Use prediction
        mutation_rate = prediction.suggested_mutation_rate
        focus = prediction.focus_spine
        
        # After optimization run
        rdnn.finalize_run(success=True, final_fitness=0.92)
        rdnn.save_state()  # Persist hidden state for next run
    
    PAPER REFERENCE:
        Yu et al. 2023 - "Experience-based" means the hidden state
        encodes patterns from previous runs, enabling faster convergence
        on similar problems (warm start advantage).
    """
    
    def __init__(
        self,
        config: Optional[RDNNConfig] = None,
        state_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize RDNN Memory.
        
        Args:
            config: RDNN configuration (uses defaults if None)
            state_path: Directory to load/save hidden state
        """
        self.config = config or RDNNConfig()
        self.state_path = Path(state_path) if state_path else None
        
        # Initialize model
        self.model = RDNNModel(self.config)
        self.model.to(self.config.device)
        self.model.eval()  # Inference mode by default
        
        # Optimizer for online learning
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Observation history (for sequence input)
        self.history: List[RDNNObservation] = []
        self.max_history = 20  # Keep last N observations for context
        
        # Hidden state
        self._hidden: Optional[Tuple[torch.Tensor, ...]] = None
        
        # Run statistics
        self.total_runs = 0
        self.successful_runs = 0
        self.run_start_time = None
        
        # Load persisted state if available
        if self.state_path and self.state_path.exists():
            self._load_state()
        
        logger.info(
            f"RDNNMemory initialized: {self.config.architecture.name}, "
            f"hidden={self.config.hidden_size}, layers={self.config.num_layers}"
        )
    
    def step(self, observation: RDNNObservation) -> RDNNPrediction:
        """
        Process one generation and return predictions.
        
        This is called once per generation by the optimizer.
        
        Args:
            observation: Current generation observation
        
        Returns:
            RDNNPrediction with suggestions for next generation
        """
        # Add to history
        self.history.append(observation)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Prepare input sequence
        features = torch.stack([
            obs.to_tensor(self.config.device)
            for obs in self.history
        ]).unsqueeze(0)  # (1, seq_len, features)
        
        # Forward pass
        with torch.no_grad():
            outputs, new_hidden = self.model(features, self._hidden)
            self._hidden = new_hidden
        
        # Parse outputs
        mutation_rate = outputs["mutation_rate"].item()
        direction = outputs["direction"].squeeze().cpu().numpy()
        fitness_out = outputs["fitness"].squeeze().cpu().numpy()
        anomaly = outputs["anomaly"].squeeze().cpu().numpy()
        
        # Determine likely anomaly type
        anomaly_types = ["stagnation", "diversity_collapse", "regression", "none"]
        likely_anomaly = anomaly_types[np.argmax(anomaly)]
        anomaly_prob = 1.0 - anomaly[3]  # P(not none)
        
        # Hidden state norm (for monitoring)
        if self._hidden is not None:
            if isinstance(self._hidden, tuple):
                h = self._hidden[0]  # h for LSTM
            else:
                h = self._hidden
            hidden_norm = torch.norm(h).item()
        else:
            hidden_norm = 0.0
        
        # Determine recommended action
        action = self._determine_action(
            mutation_rate, direction, anomaly_prob, likely_anomaly
        )
        
        return RDNNPrediction(
            suggested_mutation_rate=float(mutation_rate),
            focus_spine=float(direction[0]),
            focus_ears=float(direction[1]),
            focus_energy=float(direction[2]),
            predicted_fitness=float(np.clip(fitness_out[0], 0, 1)),
            prediction_confidence=float(np.clip(fitness_out[1], 0, 1)),
            anomaly_probability=float(anomaly_prob),
            likely_anomaly_type=likely_anomaly,
            hidden_state_norm=hidden_norm,
            recommended_action=action,
        )
    
    def _determine_action(
        self,
        mutation_rate: float,
        direction: np.ndarray,
        anomaly_prob: float,
        anomaly_type: str,
    ) -> str:
        """Determine recommended action based on predictions."""
        if anomaly_prob > 0.7:
            if anomaly_type == "stagnation":
                return "inject_diversity"
            elif anomaly_type == "diversity_collapse":
                return "reset_population"
            elif anomaly_type == "regression":
                return "restore_best"
        
        if mutation_rate > 0.4:
            return "increase_exploration"
        elif mutation_rate < 0.1:
            return "fine_tune"
        
        return "continue"
    
    def finalize_run(
        self,
        success: bool,
        final_fitness: float,
        final_objectives: Optional[Dict[str, float]] = None,
        notes: str = "",
    ):
        """
        Finalize an optimization run.
        
        Called when optimization completes. Updates statistics and
        optionally performs online learning from the experience.
        
        Args:
            success: Whether the run achieved its goal
            final_fitness: Final best fitness achieved
            final_objectives: Final objective scores
            notes: Optional notes about the run
        """
        self.total_runs += 1
        if success:
            self.successful_runs += 1
        
        # Log run summary
        duration = (
            datetime.now().timestamp() - self.run_start_time
            if self.run_start_time else 0
        )
        
        logger.info(
            f"Run finalized: success={success}, fitness={final_fitness:.4f}, "
            f"duration={duration:.1f}s, total_runs={self.total_runs}"
        )
        
        # Online learning from successful runs
        if success and len(self.history) > 5:
            self._learn_from_run(final_fitness)
        
        # Clear history for next run (but keep hidden state!)
        self.history.clear()
        self.run_start_time = None
    
    def _learn_from_run(self, final_fitness: float):
        """
        Online learning from a successful run.
        
        Uses the recorded trajectory to improve predictions.
        """
        if len(self.history) < 5:
            return
        
        self.model.train()
        
        # Prepare training data (predict next fitness from current observation)
        features = torch.stack([
            obs.to_tensor(self.config.device)
            for obs in self.history[:-1]  # All but last
        ]).unsqueeze(0)
        
        # Target: actual fitnesses (shifted by 1)
        target_fitnesses = torch.tensor(
            [obs.best_fitness for obs in self.history[1:]],
            dtype=torch.float32,
            device=self.config.device,
        )
        
        # Forward
        outputs, _ = self.model(features, None)
        # outputs["fitness"] is (batch, 2) where [0] is fitness, [1] is confidence
        predicted = outputs["fitness"][:, 0]  # Just fitness scalar
        
        # Target is just the last fitness (since we predict from full sequence)
        target = target_fitnesses[-1].unsqueeze(0)
        
        # Loss
        loss = nn.MSELoss()(predicted, target)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip
        )
        self.optimizer.step()
        
        self.model.eval()
        
        logger.debug(f"Online learning: loss={loss.item():.6f}")
    
    def start_run(self):
        """Start a new optimization run."""
        self.run_start_time = datetime.now().timestamp()
        logger.info("RDNN: New optimization run started")
    
    def reset_hidden_state(self):
        """Reset hidden state (cold start)."""
        self._hidden = None
        logger.info("RDNN: Hidden state reset")
    
    def save_state(self, path: Optional[Path] = None):
        """
        Save hidden state and model for persistence.
        
        This is the KEY method for experience transfer between runs.
        """
        save_path = path or self.state_path
        if save_path is None:
            logger.warning("No state path configured, cannot save")
            return
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hidden_state": self._hidden,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "config": {
                "architecture": self.config.architecture.name,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        torch.save(state, save_path / "rdnn_state.pt")
        
        logger.info(f"RDNN state saved to {save_path}")
    
    def _load_state(self):
        """Load persisted state."""
        state_file = self.state_path / "rdnn_state.pt"
        if not state_file.exists():
            logger.info("No persisted state found, starting fresh")
            return
        
        try:
            state = torch.load(state_file, map_location=self.config.device)
            
            # Verify config compatibility
            saved_config = state.get("config", {})
            if (saved_config.get("hidden_size") != self.config.hidden_size or
                saved_config.get("num_layers") != self.config.num_layers):
                logger.warning(
                    "Config mismatch, cannot load state. Starting fresh."
                )
                return
            
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self._hidden = state.get("hidden_state")
            self.total_runs = state.get("total_runs", 0)
            self.successful_runs = state.get("successful_runs", 0)
            
            logger.info(
                f"RDNN state loaded: {self.total_runs} previous runs, "
                f"{self.successful_runs} successful"
            )
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "success_rate": (
                self.successful_runs / self.total_runs
                if self.total_runs > 0 else 0
            ),
            "history_length": len(self.history),
            "hidden_state_active": self._hidden is not None,
            "architecture": self.config.architecture.name,
            "hidden_size": self.config.hidden_size,
        }


# ══════════════════════════════════════════════════════════════════════════════
# OBSERVATION BUILDER (Integration Helper)
# ══════════════════════════════════════════════════════════════════════════════

class ObservationBuilder:
    """
    Helper to build RDNNObservation from various sources.
    
    Integrates:
    - PokayokeObserver anomaly signals
    - PhysicsRulesEngine rule scores
    - Population fitness statistics
    - ExciterGene activation state
    
    USAGE:
        builder = ObservationBuilder()
        
        # Set data from different sources
        builder.set_fitness_data(population_fitnesses, objectives)
        builder.set_anomaly_data(pokayoke_observer.get_anomaly_signals())
        builder.set_physics_data(physics_engine.get_rule_scores(genome))
        builder.set_gene_state(exciter_genes)
        
        # Build observation
        obs = builder.build(generation=gen)
    """
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._prev_best_fitness: float = 0.0
    
    def set_fitness_data(
        self,
        fitnesses: np.ndarray,
        objectives: Optional[List[Dict[str, float]]] = None,
    ) -> "ObservationBuilder":
        """Set fitness data from population."""
        self._data["best_fitness"] = float(np.max(fitnesses))
        self._data["mean_fitness"] = float(np.mean(fitnesses))
        self._data["fitness_std"] = float(np.std(fitnesses))
        self._data["fitness_velocity"] = (
            self._data["best_fitness"] - self._prev_best_fitness
        )
        
        # Diversity
        unique = len(np.unique(np.round(fitnesses, 4)))
        self._data["phenotype_diversity"] = unique / len(fitnesses)
        self._data["population_diversity"] = (
            np.std(fitnesses) / (np.mean(fitnesses) + 1e-8)
        )
        
        # Objectives
        if objectives:
            obj_means = {}
            for key in objectives[0].keys():
                values = [o.get(key, 0) for o in objectives]
                obj_means[key] = np.mean(values)
            
            self._data["spine_flatness"] = obj_means.get("spine_flatness", 0)
            self._data["ear_uniformity"] = obj_means.get("ear_lr_uniformity", 0)
            self._data["total_energy"] = obj_means.get("total_energy", 0)
        
        self._prev_best_fitness = self._data["best_fitness"]
        return self
    
    def set_anomaly_data(
        self,
        anomaly_signals: Dict[str, float],
    ) -> "ObservationBuilder":
        """Set anomaly signals from PokayokeObserver."""
        self._data["stagnation_signal"] = anomaly_signals.get("stagnation", 0)
        self._data["diversity_collapse_signal"] = anomaly_signals.get(
            "diversity_collapse", 0
        )
        self._data["regression_signal"] = anomaly_signals.get("regression", 0)
        return self
    
    def set_physics_data(
        self,
        rule_scores: Dict[str, float],
    ) -> "ObservationBuilder":
        """Set rule satisfaction scores from PhysicsRulesEngine."""
        self._data["antinode_score"] = rule_scores.get("antinode", 0)
        self._data["node_avoidance_score"] = rule_scores.get("node_avoidance", 0)
        self._data["edge_distance_score"] = rule_scores.get("edge_distance", 0)
        self._data["phase_coherence_score"] = rule_scores.get("phase_coherence", 0)
        return self
    
    def set_operator_data(
        self,
        mutation_successes: int,
        crossover_successes: int,
        total_evaluations: int,
    ) -> "ObservationBuilder":
        """Set operator effectiveness data."""
        self._data["mutation_success_rate"] = (
            mutation_successes / (total_evaluations + 1)
        )
        self._data["crossover_success_rate"] = (
            crossover_successes / (total_evaluations + 1)
        )
        return self
    
    def set_gene_state(
        self,
        emission_active_ratio: float,
    ) -> "ObservationBuilder":
        """
        Set gene activation state.
        
        Args:
            emission_active_ratio: Ratio of exciters with emission active [0,1]
        """
        self._data["emission_genes_active"] = emission_active_ratio
        return self
    
    def build(self, generation: int) -> RDNNObservation:
        """Build the RDNNObservation."""
        return RDNNObservation(
            generation=generation,
            best_fitness=self._data.get("best_fitness", 0),
            mean_fitness=self._data.get("mean_fitness", 0),
            fitness_std=self._data.get("fitness_std", 0),
            fitness_velocity=self._data.get("fitness_velocity", 0),
            population_diversity=self._data.get("population_diversity", 0),
            phenotype_diversity=self._data.get("phenotype_diversity", 0),
            spine_flatness=self._data.get("spine_flatness", 0),
            ear_uniformity=self._data.get("ear_uniformity", 0),
            total_energy=self._data.get("total_energy", 0),
            antinode_score=self._data.get("antinode_score", 0),
            node_avoidance_score=self._data.get("node_avoidance_score", 0),
            edge_distance_score=self._data.get("edge_distance_score", 0),
            phase_coherence_score=self._data.get("phase_coherence_score", 0),
            stagnation_signal=self._data.get("stagnation_signal", 0),
            diversity_collapse_signal=self._data.get("diversity_collapse_signal", 0),
            regression_signal=self._data.get("regression_signal", 0),
            mutation_success_rate=self._data.get("mutation_success_rate", 0),
            crossover_success_rate=self._data.get("crossover_success_rate", 0),
            emission_genes_active=self._data.get("emission_genes_active", 0),
        )
    
    def reset(self):
        """Reset builder for next generation."""
        # Keep prev_best_fitness for velocity calculation
        prev_best = self._prev_best_fitness
        self._data.clear()
        self._prev_best_fitness = prev_best


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def create_rdnn_memory(
    state_path: Optional[str] = None,
    architecture: RDNNArchitecture = RDNNArchitecture.GRU,
    hidden_size: int = 64,
    use_cuda: bool = False,
) -> RDNNMemory:
    """
    Factory function to create RDNN memory with sensible defaults.
    
    Args:
        state_path: Path to save/load state (enables warm start)
        architecture: LSTM or GRU
        hidden_size: Hidden state dimension
        use_cuda: Whether to use GPU if available
    
    Returns:
        Configured RDNNMemory instance
    """
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    
    config = RDNNConfig(
        architecture=architecture,
        hidden_size=hidden_size,
        device=device,
    )
    
    return RDNNMemory(config, state_path)
