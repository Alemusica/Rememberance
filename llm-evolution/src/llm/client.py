"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         OLLAMA LLM CLIENT                                    ║
║                                                                              ║
║   Client per LLM locali via Ollama. Niente cloud, tutto in locale.           ║
║   Supporta caching, retry, e parsing JSON strutturato.                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import httpx
import json
import re
import hashlib
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configurazione modello LLM."""
    name: str                      # es. "deepseek-r1:14b"
    temperature: float = 0.7       # 0 = deterministico, 1 = creativo
    max_tokens: int = 4096         # Max token output
    top_p: float = 0.9             # Nucleus sampling
    top_k: int = 40                # Top-k sampling
    context_length: int = 32768    # Context window
    
    # Ollama specific
    num_gpu: int = -1              # -1 = auto
    repeat_penalty: float = 1.1   # Penalità ripetizioni


# Modelli predefiniti per ruoli specifici
MODELS = {
    # Orchestrator: reasoning pesante, pianificazione
    "orchestrator": ModelConfig(
        name="deepseek-r1:14b",
        temperature=0.3,        # Più deterministico per planning
        max_tokens=8192,
        context_length=65536,
    ),
    
    # Strategy: decisioni su parametri evolutivi
    "strategy": ModelConfig(
        name="qwen2.5-coder:14b",
        temperature=0.5,
        max_tokens=4096,
    ),
    
    # Analysis: interpretazione fitness, anomalie
    "analysis": ModelConfig(
        name="llama3.2:3b",
        temperature=0.2,        # Molto deterministico
        max_tokens=2048,
    ),
    
    # Fast: decisioni semplici, alta frequenza
    "fast": ModelConfig(
        name="qwen2.5:3b",
        temperature=0.1,
        max_tokens=512,
    ),
    
    # Code: generazione codice per tool mancanti
    "code": ModelConfig(
        name="qwen2.5-coder:14b",
        temperature=0.2,
        max_tokens=8192,
    ),
}


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient:
    """
    Client asincrono per Ollama API.
    
    Features:
    - Caching locale delle risposte
    - Retry automatico con backoff
    - Parsing JSON strutturato
    - Support per chat multi-turn
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        cache_dir: Optional[Path] = None,
        enable_cache: bool = True,
        timeout: float = 300.0,
    ):
        """
        Args:
            base_url: URL base di Ollama (default localhost:11434)
            cache_dir: Directory per cache risposte
            enable_cache: Abilita caching
            timeout: Timeout richieste in secondi
        """
        self.base_url = base_url
        self.enable_cache = enable_cache
        self.timeout = timeout
        
        # Setup cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "llm-evolution"
        
        if enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        
        # Stats
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "tokens_generated": 0,
            "errors": 0,
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=10),
            )
        return self._client
    
    async def generate(
        self,
        prompt: str,
        model: Union[ModelConfig, str],
        system: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> str:
        """
        Genera risposta da LLM.
        
        Args:
            prompt: Prompt utente
            model: ModelConfig o nome modello
            system: System prompt (opzionale)
            use_cache: Usa cache se disponibile
            **kwargs: Parametri extra per Ollama
        
        Returns:
            Testo generato
        """
        # Normalize model
        if isinstance(model, str):
            model = MODELS.get(model, ModelConfig(name=model))
        
        # Check cache
        if self.enable_cache and use_cache:
            cache_key = self._cache_key(prompt, model.name, system)
            cached = self._load_cache(cache_key)
            if cached is not None:
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit: {cache_key[:8]}...")
                return cached
        
        # Build request
        payload = {
            "model": model.name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": model.temperature,
                "num_predict": model.max_tokens,
                "top_p": model.top_p,
                "top_k": model.top_k,
                "repeat_penalty": model.repeat_penalty,
            }
        }
        
        if system:
            payload["system"] = system
        
        payload["options"].update(kwargs)
        
        # Make request with retry
        self.stats["requests"] += 1
        
        for attempt in range(3):
            try:
                client = await self._get_client()
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                
                text = result.get("response", "")
                
                # Update stats
                self.stats["tokens_generated"] += result.get("eval_count", 0)
                
                # Save to cache
                if self.enable_cache and use_cache and text:
                    self._save_cache(cache_key, text)
                
                return text
                
            except httpx.HTTPError as e:
                self.stats["errors"] += 1
                logger.warning(f"Ollama error (attempt {attempt + 1}): {e}")
                
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        return ""
    
    async def generate_json(
        self,
        prompt: str,
        model: Union[ModelConfig, str],
        system: Optional[str] = None,
        schema: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Genera e parsa risposta JSON.
        
        Args:
            prompt: Prompt utente
            model: Modello
            system: System prompt
            schema: JSON schema atteso (per validazione)
        
        Returns:
            Dizionario parsed
        
        Raises:
            ValueError: Se non riesce a parsare JSON
        """
        # Add JSON instruction
        json_prompt = prompt.strip()
        if not json_prompt.endswith("JSON"):
            json_prompt += "\n\nRespond with valid JSON only, no explanation."
        
        response = await self.generate(json_prompt, model, system, use_cache=False)
        
        # Try to extract JSON
        try:
            # Direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # Markdown JSON block
            r'```\s*([\s\S]*?)\s*```',       # Any code block
            r'\{[\s\S]*\}',                   # Raw JSON object
            r'\[[\s\S]*\]',                   # Raw JSON array
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue
        
        logger.error(f"Cannot parse JSON from: {response[:500]}...")
        raise ValueError(f"Failed to parse JSON from LLM response")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Union[ModelConfig, str],
    ) -> str:
        """
        Chat multi-turn.
        
        Args:
            messages: Lista di {"role": "user"|"assistant"|"system", "content": "..."}
            model: Modello
        
        Returns:
            Risposta assistant
        """
        if isinstance(model, str):
            model = MODELS.get(model, ModelConfig(name=model))
        
        payload = {
            "model": model.name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": model.temperature,
                "num_predict": model.max_tokens,
            }
        }
        
        self.stats["requests"] += 1
        
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            
            text = result.get("message", {}).get("content", "")
            self.stats["tokens_generated"] += result.get("eval_count", 0)
            
            return text
            
        except httpx.HTTPError as e:
            self.stats["errors"] += 1
            logger.error(f"Chat error: {e}")
            raise
    
    async def list_models(self) -> List[str]:
        """Lista modelli disponibili su Ollama."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except httpx.HTTPError as e:
            logger.error(f"Cannot list models: {e}")
            return []
    
    async def ensure_model(self, model_name: str) -> bool:
        """Verifica che il modello sia disponibile."""
        models = await self.list_models()
        return any(model_name in m for m in models)
    
    async def pull_model(self, model_name: str) -> bool:
        """Scarica modello (blocking)."""
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=None,  # No timeout for pulls
            )
            response.raise_for_status()
            return True
        except httpx.HTTPError as e:
            logger.error(f"Cannot pull model: {e}")
            return False
    
    def _cache_key(self, prompt: str, model: str, system: Optional[str]) -> str:
        """Genera chiave cache deterministica."""
        content = f"{model}:{system or ''}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_cache(self, key: str) -> Optional[str]:
        """Carica da cache locale."""
        cache_file = self.cache_dir / f"{key}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        return None
    
    def _save_cache(self, key: str, content: str):
        """Salva in cache locale."""
        cache_file = self.cache_dir / f"{key}.txt"
        cache_file.write_text(content, encoding="utf-8")
    
    def clear_cache(self):
        """Svuota cache locale."""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.txt"):
                f.unlink()
            logger.info("Cache cleared")
    
    async def close(self):
        """Chiudi client HTTP."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche utilizzo."""
        return {
            **self.stats,
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(1, self.stats["requests"])
            ),
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_client: Optional[OllamaClient] = None


def get_client() -> OllamaClient:
    """Get singleton client."""
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client


async def quick_generate(
    prompt: str,
    model: str = "fast",
    system: Optional[str] = None,
) -> str:
    """Quick generation with default client."""
    client = get_client()
    return await client.generate(prompt, model, system)
