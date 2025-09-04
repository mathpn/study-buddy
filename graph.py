from enum import Enum
from textwrap import dedent
from typing import List

from pydantic import BaseModel, Field

from logger import setup_logger
from models import ModelProvider

logger = setup_logger(__name__)


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
    """Builds a knowledge graph from a text chunk using a model."""
    logger.info("Building knowledge graph from chunk")
    system_prompt = """
    ## Task
    Extract a knowledge graph from the provided document chunk, focusing on concepts essential for studying and understanding the material.

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
    - **implements**: Source implements target
    - **measures**: Source measures target
    - **applies_to**: Source applies to target
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

    user_prompt = """
    ## Document Chunks

    {DOCUMENT_CHUNK}
    """

    graph = model.chat_with_schema(
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": dedent(system_prompt)}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": dedent(user_prompt.format(DOCUMENT_CHUNK=chunk)),
                    }
                ],
            },
        ],
        KnowledgeGraph,
    )

    if graph:
        logger.info(
            f"Successfully built graph with {len(graph.nodes)} nodes and {len(graph.relationships)} relationships."
        )
    else:
        logger.warning("Failed to build knowledge graph from chunk.")

    return graph


def merge_knowledge_graphs(
    main_graph: KnowledgeGraph, new_graph: KnowledgeGraph
) -> None:
    """
    Merge a new knowledge graph into the main cumulative graph.
    Avoids duplicate nodes (by id) and relationships.

    Args:
        main_graph: The cumulative session graph to merge into
        new_graph: The new graph fragment to merge
    """
    logger.info("Merging new knowledge graph into main graph.")
    logger.debug(
        f"Main graph before merge: {len(main_graph.nodes)} nodes, {len(main_graph.relationships)} relationships."
    )
    logger.debug(
        f"New graph to merge: {len(new_graph.nodes)} nodes, {len(new_graph.relationships)} relationships."
    )

    existing_node_ids = {node.id for node in main_graph.nodes}
    existing_relationships = {
        (rel.source, rel.target, rel.relationship) for rel in main_graph.relationships
    }

    nodes_added = 0
    for node in new_graph.nodes:
        if node.id not in existing_node_ids:
            main_graph.nodes.append(node)
            existing_node_ids.add(node.id)
            nodes_added += 1

    rels_added = 0
    for relationship in new_graph.relationships:
        rel_key = (
            relationship.source,
            relationship.target,
            relationship.relationship,
        )
        if rel_key not in existing_relationships:
            main_graph.relationships.append(relationship)
            existing_relationships.add(rel_key)
            rels_added += 1

    logger.info(f"Added {nodes_added} new nodes and {rels_added} new relationships.")
    logger.debug(
        f"Main graph after merge: {len(main_graph.nodes)} nodes, {len(main_graph.relationships)} relationships."
    )
