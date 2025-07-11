from enum import Enum
from textwrap import dedent
from typing import List

from pydantic import BaseModel, Field

from models import ModelProvider


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph"""

    CONCEPT = "concept"
    METHOD = "method"
    ENTITY = "entity"
    METRIC = "metric"
    APPLICATION = "application"


class RelationshipType(str, Enum):
    """Types of relationships between nodes"""

    DEFINES = "defines"
    REQUIRES = "requires"
    IMPLEMENTS = "implements"
    MEASURES = "measures"
    APPLIES_TO = "applies_to"
    CONTRASTS_WITH = "contrasts_with"
    RELATED_TO = "related_to"


class KGNode(BaseModel):
    """A node in the knowledge graph"""

    id: str = Field(..., description="Unique identifier for the node")
    label: str = Field(..., description="Human-readable name of the concept")
    type: NodeType = Field(..., description="Type of the node")
    description: str = Field(..., description="Brief description of the concept")

    class Config:
        use_enum_values = True


class KGRelationship(BaseModel):
    """A relationship between two nodes"""

    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node")
    relationship: RelationshipType = Field(..., description="Type of relationship")
    description: str = Field(..., description="Brief explanation of the relationship")

    class Config:
        use_enum_values = True


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph for a document chunk"""

    nodes: List[KGNode] = Field(..., description="List of all nodes in the graph")
    relationships: List[KGRelationship] = Field(
        ..., description="List of all relationships"
    )

    def get_json_schema(self) -> dict:
        """Get JSON schema for the knowledge graph"""
        return self.model_json_schema()


def build_knowledge_graph(chunk: str, model: ModelProvider) -> KnowledgeGraph | None:
    prompt = """
    ## Task
    Extract a knowledge graph from the provided document chunk, focusing on concepts essential for studying and understanding the material.

    ## Document Chunk
    ```
    {DOCUMENT_CHUNK}
    ```

    ## Instructions

    ### Node Types
    - **concept**: Key terms, theories, principles
    - **method**: Procedures, algorithms, techniques
    - **entity**: People, organizations, systems, tools
    - **metric**: Measurements, KPIs, evaluation criteria
    - **application**: Use cases, examples, implementations

    ### Relationship Types
    - **defines**: Source defines or explains target
    - **requires**: Source requires or depends on target
    - **causes**: Source causes or leads to target
    - **implements**: Source implements target
    - **measures**: Source measures target
    - **applies_to**: Source applies to target
    - **composed_of**: Source is composed of target
    - **contrasts_with**: Source contrasts with target
    - **related_to**: General relationship between source and target

    ## Output Requirements
    Return a valid JSON object matching this exact structure:

    ```json
    {{
      "nodes": [
        {{
          "id": "unique_snake_case_identifier",
          "label": "Human Readable Name",
          "type": "concept|method|entity|metric|application",
          "description": "Brief description of the concept",
        }}
      ],
      "relationships": [
        {{
          "source": "source_node_id",
          "target": "target_node_id",
          "relationship": "defines|requires|implements|measures|applies_to|contrasts_with|related_to",
          "description": "Brief explanation of the relationship"
        }}
      ]
    }}
    ```

    ## Guidelines
    - Focus on study-relevant concepts that help understand the material
    - Use clear, consistent terminology
    - Create meaningful relationships between concepts
    - Prioritize educational value over completeness
    - Use snake_case for node IDs (e.g., "neural_network", "gradient_descent")
    - Keep descriptions concise but informative

    Extract only the most important concepts and relationships for learning this material.
    """
    return model.generate_with_schema(
        dedent(prompt.format(DOCUMENT_CHUNK=chunk)), KnowledgeGraph
    )


if __name__ == "__main__":
    from models import OpenAIModel

    model = OpenAIModel("gpt-4.1-mini")
    chunk = """
    Goodhart's law was first introduced by Goodhart (1984), and has later been elaborated upon by works such as Manheim &amp; Garrabrant (2019). Goodhart's law has also previously been studied in the context of machine learning. In particular, Hennessy &amp; Goodhart (2023) investigate Goodhart's law analytically in the context where a machine learning model is used to evaluate an agent's actions - unlike them, we specifically consider the RL setting. Ashton (2021) shows by example that RL systems can be susceptible to Goodharting in certain situations. In contrast, we show that Goodhart's law is a robust phenomenon across a wide range of environments, explain why it occurs in RL, and use it to devise new solution methods.

    In the context of RL, Goodhart's law is closely related to reward gaming . Specifically, if reward gaming means an agent finding an unintended way to increase its reward, then Goodharting is an instance of reward gaming where optimisation of the proxy initially leads to desirable behaviour, followed by a decrease after some threshold. Krakovna et al. (2020) list illustrative examples of reward hacking, while Pan et al. (2021) manually construct proxy rewards for several environments and then demonstrate that most of them lead to reward hacking. Zhuang &amp; Hadfield-Menell (2020) consider proxy rewards that depend on a strict subset of the features which are relevant to the true reward and then show that optimising such a proxy in some cases may be arbitrarily bad, given certain assumptions. Skalse et al. (2022) introduce a theoretical framework for analysing reward hacking. They then demonstrate that, in any environment and for any true reward function, it is impossible to create a non-trivial proxy reward that is guaranteed to be unhackable. Also relevant, Everitt et al. (2017) study the related problem of reward corruption, Song et al. (2019) investigate overfitting in model-free RL due to faulty implications from correlations in the environment, and Pang et al. (2022) examine reward gaming in language models. Unlike these works, we analyse reward hacking through the lens of Goodhart's law and show that this perspective provides novel insights.

    Gao et al. (2023) consider the setting where a large language model is optimised against a reward model that has been trained on a 'gold standard' reward function, and investigate how the performance of the language model according to the gold standard reward scales in the size of the language model, the amount of training data, and the size of the reward model. They find that the performance of the policy follows a Goodhart curve, where the slope gets less prominent for larger reward models and larger amounts of training data. Unlike them, we do not only focus on language, but rather, aim to establish to what extent Goodhart dynamics occur for a wide range of RL environments. Moreover, we also aim to explain Goodhart's law, and use it as a starting point for developing new algorithms.
    """
    from textwrap import dedent

    kg = build_knowledge_graph(chunk, model)
    if kg is not None:
        print(kg.model_dump_json(indent=2))
    else:
        print("No knowledge graph generated")

