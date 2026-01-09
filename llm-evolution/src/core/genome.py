"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AGNOSTIC GENOME PROTOCOL                                   ║
║                                                                              ║
║   Il genome NON SA cosa rappresenta. È un contenitore di parametri che       ║
║   può essere:                                                                ║
║   • Singing bowl tibetana (diametro, spessore, materiale...)                 ║
║   • Piatto vibroacustico (cutouts, exciters, contorno...)                   ║
║   • Funzione benchmark (vettore n-dimensionale)                              ║
║   • Qualsiasi altra cosa descrivibile con parametri                          ║
║                                                                              ║
║   L'Orchestrator LLM genera lo schema in base alla descrizione umana.        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Protocol, Dict, Any, List, TypeVar, Optional, runtime_checkable
from dataclasses import dataclass, field
import numpy as np
import json


# =============================================================================
# GENOME PROTOCOL
# =============================================================================

@runtime_checkable
class Genome(Protocol):
    """
    Protocol che ogni genome deve implementare.
    
    Questo è il contratto che permette all'evoluzione di essere agnostica:
    non importa COSA sia il genome, basta che rispetti questa interfaccia.
    """
    
    def to_vector(self) -> np.ndarray:
        """
        Converte il genome a vettore numerico normalizzato [0,1].
        Usato per operatori GA standard (crossover, mutation).
        """
        ...
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, schema: 'GenomeSchema') -> 'Genome':
        """
        Ricostruisce genome da vettore normalizzato.
        """
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializza genome per LLM/storage/logging.
        I valori sono nei loro range naturali (non normalizzati).
        """
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], schema: 'GenomeSchema') -> 'Genome':
        """Deserializza da dizionario."""
        ...
    
    def validate(self) -> List[str]:
        """
        Ritorna lista di errori di validazione.
        Lista vuota = genome valido.
        """
        ...
    
    def clone(self) -> 'Genome':
        """Crea copia indipendente."""
        ...


# =============================================================================
# GENE SPECIFICATION
# =============================================================================

@dataclass
class GeneSpec:
    """
    Specifica di un singolo gene (parametro del genome).
    
    Supporta:
    - float: numeri decimali con range
    - int: interi con range
    - categorical: scelta da lista
    - bool: vero/falso
    """
    name: str
    type: str  # "float", "int", "categorical", "bool"
    
    # Per numerici (float, int)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Per categorici
    categories: Optional[List[str]] = None
    
    # Metadata per LLM
    description: str = ""
    unit: str = ""  # "mm", "Hz", "kg", etc.
    
    # Mutation hint per LLM
    mutation_hint: str = ""  # "small_step", "large_jump", "flip", etc.
    
    def random_value(self, rng: np.random.Generator = None) -> Any:
        """Genera valore random valido per questo gene."""
        rng = rng or np.random.default_rng()
        
        if self.type == "float":
            return float(rng.uniform(self.min_value, self.max_value))
        elif self.type == "int":
            return int(rng.integers(self.min_value, self.max_value + 1))
        elif self.type == "categorical":
            return str(rng.choice(self.categories))
        elif self.type == "bool":
            return bool(rng.choice([True, False]))
        else:
            raise ValueError(f"Unknown gene type: {self.type}")
    
    def normalize(self, value: Any) -> float:
        """Normalizza valore in [0, 1]."""
        if self.type == "float":
            return (value - self.min_value) / (self.max_value - self.min_value)
        elif self.type == "int":
            return (value - self.min_value) / (self.max_value - self.min_value)
        elif self.type == "categorical":
            return self.categories.index(value) / max(1, len(self.categories) - 1)
        elif self.type == "bool":
            return 1.0 if value else 0.0
        return 0.0
    
    def denormalize(self, norm_value: float) -> Any:
        """Denormalizza da [0, 1] al valore originale."""
        norm_value = np.clip(norm_value, 0.0, 1.0)
        
        if self.type == "float":
            return self.min_value + norm_value * (self.max_value - self.min_value)
        elif self.type == "int":
            return int(round(self.min_value + norm_value * (self.max_value - self.min_value)))
        elif self.type == "categorical":
            idx = int(round(norm_value * (len(self.categories) - 1)))
            return self.categories[idx]
        elif self.type == "bool":
            return norm_value > 0.5
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializza specifica gene."""
        d = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
        }
        if self.type in ("float", "int"):
            d["min"] = self.min_value
            d["max"] = self.max_value
            if self.unit:
                d["unit"] = self.unit
        elif self.type == "categorical":
            d["categories"] = self.categories
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeneSpec':
        """Deserializza specifica gene."""
        return cls(
            name=data["name"],
            type=data["type"],
            min_value=data.get("min"),
            max_value=data.get("max"),
            categories=data.get("categories"),
            description=data.get("description", ""),
            unit=data.get("unit", ""),
        )


# =============================================================================
# GENOME SCHEMA
# =============================================================================

@dataclass
class GenomeSchema:
    """
    Schema completo del genome.
    
    Generato dall'Orchestrator LLM in base alla descrizione umana.
    Contiene tutto ciò che serve per creare, validare e mutare genomi.
    """
    name: str
    genes: List[GeneSpec] = field(default_factory=list)
    
    # Constraints (validati dopo creazione)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata per LLM context
    description: str = ""
    domain: str = ""  # "acoustic", "mechanical", "mathematical", etc.
    domain_knowledge: str = ""  # Injected from RAG
    
    # Required tools for fitness evaluation
    required_tools: List[str] = field(default_factory=list)
    
    @property
    def dimension(self) -> int:
        """Dimensione vettore numerico."""
        return len(self.genes)
    
    @property
    def gene_names(self) -> List[str]:
        """Lista nomi geni."""
        return [g.name for g in self.genes]
    
    def get_gene(self, name: str) -> Optional[GeneSpec]:
        """Ottieni specifica gene per nome."""
        for g in self.genes:
            if g.name == name:
                return g
        return None
    
    def to_prompt_context(self) -> str:
        """
        Genera context per prompt LLM.
        Usato quando un agente deve ragionare sul genome.
        """
        lines = [
            f"# Genome: {self.name}",
            f"Domain: {self.domain}",
            f"Description: {self.description}",
            "",
            "## Genes:",
        ]
        
        for g in self.genes:
            if g.type in ("float", "int"):
                unit = f" {g.unit}" if g.unit else ""
                lines.append(
                    f"- **{g.name}**: {g.type} [{g.min_value}, {g.max_value}]{unit}"
                )
            elif g.type == "categorical":
                lines.append(f"- **{g.name}**: one of {g.categories}")
            elif g.type == "bool":
                lines.append(f"- **{g.name}**: true/false")
            
            if g.description:
                lines.append(f"  {g.description}")
        
        if self.constraints:
            lines.append("")
            lines.append("## Constraints:")
            for c in self.constraints:
                lines.append(f"- {c.get('description', c)}")
        
        if self.domain_knowledge:
            lines.append("")
            lines.append("## Domain Knowledge:")
            lines.append(self.domain_knowledge[:2000])  # Truncate
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializza schema."""
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "genes": [g.to_dict() for g in self.genes],
            "constraints": self.constraints,
            "required_tools": self.required_tools,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenomeSchema':
        """Deserializza schema."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            domain=data.get("domain", ""),
            genes=[GeneSpec.from_dict(g) for g in data.get("genes", [])],
            constraints=data.get("constraints", []),
            required_tools=data.get("required_tools", []),
        )
    
    def to_json(self) -> str:
        """Serializza a JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'GenomeSchema':
        """Deserializza da JSON."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# DICT GENOME IMPLEMENTATION
# =============================================================================

class DictGenome:
    """
    Implementazione generica di Genome basata su dizionario.
    
    Questa è l'implementazione di default usata quando l'Orchestrator
    genera uno schema. Non richiede classi custom per ogni dominio.
    """
    
    def __init__(self, data: Dict[str, Any], schema: GenomeSchema):
        """
        Args:
            data: Dizionario gene_name -> value
            schema: Schema che definisce i geni
        """
        self._data = dict(data)
        self._schema = schema
    
    @property
    def schema(self) -> GenomeSchema:
        """Schema del genome."""
        return self._schema
    
    def to_vector(self) -> np.ndarray:
        """Normalizza tutto in [0,1] per GA."""
        vector = []
        for gene in self._schema.genes:
            val = self._data.get(gene.name)
            if val is not None:
                vector.append(gene.normalize(val))
            else:
                vector.append(0.5)  # Default to middle
        return np.array(vector, dtype=np.float64)
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, schema: GenomeSchema) -> 'DictGenome':
        """Denormalizza da [0,1]."""
        data = {}
        for i, gene in enumerate(schema.genes):
            if i < len(vector):
                data[gene.name] = gene.denormalize(vector[i])
            else:
                data[gene.name] = gene.random_value()
        return cls(data, schema)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializza valori."""
        return dict(self._data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], schema: GenomeSchema) -> 'DictGenome':
        """Deserializza."""
        return cls(data, schema)
    
    def validate(self) -> List[str]:
        """Valida genome contro schema e constraints."""
        errors = []
        
        # Check all genes present and in bounds
        for gene in self._schema.genes:
            if gene.name not in self._data:
                errors.append(f"Missing gene: {gene.name}")
                continue
            
            val = self._data[gene.name]
            
            if gene.type in ("float", "int"):
                if val < gene.min_value or val > gene.max_value:
                    errors.append(
                        f"{gene.name}={val} out of bounds "
                        f"[{gene.min_value}, {gene.max_value}]"
                    )
            elif gene.type == "categorical":
                if val not in gene.categories:
                    errors.append(
                        f"{gene.name}='{val}' not in {gene.categories}"
                    )
        
        # TODO: Evaluate constraints (needs constraint evaluator)
        
        return errors
    
    def clone(self) -> 'DictGenome':
        """Crea copia indipendente."""
        return DictGenome(dict(self._data), self._schema)
    
    def mutate_gene(self, gene_name: str, value: Any) -> 'DictGenome':
        """Crea nuovo genome con gene modificato."""
        new_data = dict(self._data)
        new_data[gene_name] = value
        return DictGenome(new_data, self._schema)
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any):
        self._data[key] = value
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def __repr__(self):
        gene_strs = [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                     for k, v in self._data.items()]
        return f"Genome({', '.join(gene_strs)})"
    
    def __str__(self):
        return self.__repr__()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def random_genome(
    schema: GenomeSchema, 
    rng: np.random.Generator = None
) -> DictGenome:
    """
    Genera genome random secondo schema.
    
    Args:
        schema: Schema che definisce i geni
        rng: Random generator (opzionale)
    
    Returns:
        DictGenome con valori random validi
    """
    rng = rng or np.random.default_rng()
    data = {}
    for gene in schema.genes:
        data[gene.name] = gene.random_value(rng)
    return DictGenome(data, schema)


def crossover_uniform(
    parent1: DictGenome,
    parent2: DictGenome,
    rng: np.random.Generator = None,
) -> DictGenome:
    """
    Crossover uniforme: ogni gene preso random da un genitore.
    """
    rng = rng or np.random.default_rng()
    data = {}
    for gene in parent1.schema.genes:
        if rng.random() < 0.5:
            data[gene.name] = parent1[gene.name]
        else:
            data[gene.name] = parent2[gene.name]
    return DictGenome(data, parent1.schema)


def crossover_blend(
    parent1: DictGenome,
    parent2: DictGenome,
    alpha: float = 0.5,
    rng: np.random.Generator = None,
) -> DictGenome:
    """
    BLX-alpha crossover per geni numerici, random per altri.
    """
    rng = rng or np.random.default_rng()
    data = {}
    
    for gene in parent1.schema.genes:
        v1, v2 = parent1[gene.name], parent2[gene.name]
        
        if gene.type in ("float", "int"):
            # BLX-alpha
            d = abs(v2 - v1)
            low = min(v1, v2) - alpha * d
            high = max(v1, v2) + alpha * d
            low = max(low, gene.min_value)
            high = min(high, gene.max_value)
            
            if gene.type == "float":
                data[gene.name] = float(rng.uniform(low, high))
            else:
                data[gene.name] = int(round(rng.uniform(low, high)))
        else:
            # Random pick for categorical/bool
            data[gene.name] = v1 if rng.random() < 0.5 else v2
    
    return DictGenome(data, parent1.schema)


def mutate_gaussian(
    genome: DictGenome,
    sigma: float = 0.1,
    gene_rate: float = 0.2,
    rng: np.random.Generator = None,
) -> DictGenome:
    """
    Mutazione gaussiana: perturba geni numerici, flip altri.
    
    Args:
        genome: Genome da mutare
        sigma: Deviazione standard (in scala normalizzata [0,1])
        gene_rate: Probabilità di mutare ogni gene
        rng: Random generator
    
    Returns:
        Nuovo genome mutato
    """
    rng = rng or np.random.default_rng()
    data = dict(genome.to_dict())
    
    for gene in genome.schema.genes:
        if rng.random() >= gene_rate:
            continue
        
        if gene.type in ("float", "int"):
            # Gaussian perturbation in normalized space
            norm_val = gene.normalize(data[gene.name])
            norm_val += rng.normal(0, sigma)
            norm_val = np.clip(norm_val, 0, 1)
            data[gene.name] = gene.denormalize(norm_val)
        
        elif gene.type == "categorical":
            # Random new category
            data[gene.name] = rng.choice(gene.categories)
        
        elif gene.type == "bool":
            # Flip
            data[gene.name] = not data[gene.name]
    
    return DictGenome(data, genome.schema)
