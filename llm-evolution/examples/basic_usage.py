"""
Basic usage examples for LLM Evolution agents.

This demonstrates how to use the LLMClient, CrossoverAgent, and ExplainerAgent.
"""

import asyncio
import logging
from src.llm import LLMClient
from src.agents import CrossoverAgent, ExplainerAgent, AgentConfig

logging.basicConfig(level=logging.INFO)


async def example_llm_client():
    """Example: Using LLMClient directly."""
    print("\n=== LLM Client Example ===")
    
    # Initialize client (Ollama)
    client = LLMClient(provider="ollama", ollama_endpoint="http://localhost:11434")
    
    try:
        response = await client.complete(
            prompt="What is evolutionary computation?",
            model="llama2",
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response[:200]}...")
        
        # JSON completion
        json_response = await client.complete_json(
            prompt="Generate a simple genome with 3 parameters: width, height, thickness",
            model="llama2",
            schema={
                "type": "object",
                "properties": {
                    "width": {"type": "number"},
                    "height": {"type": "number"},
                    "thickness": {"type": "number"}
                }
            }
        )
        print(f"JSON Response: {json_response}")
        
    finally:
        await client.close()


async def example_crossover():
    """Example: Using CrossoverAgent."""
    print("\n=== Crossover Agent Example ===")
    
    # Initialize client and agent
    client = LLMClient(provider="ollama")
    config = AgentConfig(model_name="llama2", temperature=0.7)
    agent = CrossoverAgent(client, config)
    
    # Example parents
    parent_a = {"width": 0.4, "height": 0.6, "thickness": 0.01}
    parent_b = {"width": 0.5, "height": 0.5, "thickness": 0.015}
    
    parent_a_fitness = {
        "ear_uniformity": 0.85,
        "spine_coupling": 0.60,
        "flatness": 0.70
    }
    
    parent_b_fitness = {
        "ear_uniformity": 0.70,
        "spine_coupling": 0.90,
        "flatness": 0.75
    }
    
    try:
        result = await agent.crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            parent_a_fitness=parent_a_fitness,
            parent_b_fitness=parent_b_fitness,
            domain_constraints="Width and height must be between 0.3 and 0.8 meters. Thickness must be between 0.005 and 0.02 meters."
        )
        
        print(f"Offspring: {result['offspring_genome']}")
        print(f"Inherited from A: {result['inherited_from_a']}")
        print(f"Inherited from B: {result['inherited_from_b']}")
        print(f"Rationale: {result['rationale']}")
        
    finally:
        await client.close()


async def example_explainer():
    """Example: Using ExplainerAgent."""
    print("\n=== Explainer Agent Example ===")
    
    # Initialize client and agent
    client = LLMClient(provider="ollama")
    config = AgentConfig(model_name="llama2", temperature=0.5)
    agent = ExplainerAgent(client, config)
    
    # Example anomaly context
    fitness_history = [
        0.65, 0.67, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68
    ]
    
    current_genome = {"width": 0.4, "height": 0.6, "thickness": 0.01}
    
    try:
        result = await agent.explain(
            anomaly_type="STAGNATION",
            generation=10,
            fitness_history=fitness_history,
            genome=current_genome,
            anomaly_details={
                "stagnation_generations": 7,
                "population_diversity": 0.15
            }
        )
        
        print(f"Explanation: {result['explanation']}")
        print(f"Severity: {result['severity']}/10")
        print(f"Recommended Action: {result['action']}")
        print(f"Reasoning: {result['reasoning']}")
        
    finally:
        await client.close()


async def main():
    """Run all examples."""
    print("LLM Evolution - Basic Usage Examples")
    print("=" * 50)
    
    # Note: These examples require Ollama running locally
    # Uncomment to run:
    # await example_llm_client()
    # await example_crossover()
    # await example_explainer()


if __name__ == "__main__":
    asyncio.run(main())
