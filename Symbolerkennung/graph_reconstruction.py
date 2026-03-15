import os
import networkx as nx

from config import OUTPUT_DIR

def reconstruct_graph(detected_components, extracted_relations):
    """Baut aus erkannten Komponenten und Relationen einen gerichteten Graphen auf.

    Args:
        detected_components: Liste von Dicts, z.B. [{'id': 1, 'type': 'start', 'label': 'Start'}]
        extracted_relations: Liste von Tupeln, z.B. [(quell_id, ziel_id, 'relationstyp')]

    Returns:
        networkx DiGraph-Objekt mit typisierten Knoten und Kanten.
    """
    G = nx.DiGraph()

    for comp in detected_components:
        G.add_node(comp['id'], type=comp['type'], label=comp.get('label', ''))

    for source, target, rel_type in extracted_relations:
        G.add_edge(source, target, type=rel_type)

    return G

def export_to_mermaid(G, filepath):
    """Exportiert den NetworkX-Graphen als Mermaid.js-Markdown- und .mermaid-Datei.

    Args:
        G:        Gerichteter NetworkX-Graph mit Knotenattributen 'type' und 'label'.
        filepath: Zielpfad für die .md-Datei (die .mermaid-Datei wird parallel erzeugt).
    """
    mermaid_content_lines = ["stateDiagram-v2"]

    # Knotentypen in Mermaid-Syntax übersetzen
    for node, data in G.nodes(data=True):
        node_type = data['type']
        if node_type == 'choice':
            mermaid_content_lines.append(f"    state Node{node} <<choice>>")
        elif node_type == 'state':
            label = data.get('label', 'State')
            mermaid_content_lines.append(f"    Node{node} : {label}")
        elif node_type == 'action':
            label = data.get('label', 'Action')
            mermaid_content_lines.append(f"    Node{node} : {label}")

    # Kanten mit optionaler Beschriftung
    for u, v, data in G.edges(data=True):
        rel_label = data.get('type', '')

        u_str = f"Node{u}"
        v_str = f"Node{v}"

        # Start/End-Knoten auf Mermaid-Syntax [*] abbilden
        if G.nodes[u]['type'] == 'start':
            u_str = "[*]"

        if G.nodes[v]['type'] == 'ending':
            v_str = "[*]"

        if rel_label:
            mermaid_content_lines.append(f"    {u_str} --> {v_str} : {rel_label}")
        else:
            mermaid_content_lines.append(f"    {u_str} --> {v_str}")

    # Als Markdown-Datei speichern
    mermaid_md_lines = ["```mermaid"] + mermaid_content_lines + ["```"]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(mermaid_md_lines))
    print(f"Mermaid graph saved to {filepath}")

    # Als reine .mermaid-Datei speichern
    mermaid_filepath = os.path.splitext(filepath)[0] + ".mermaid"
    with open(mermaid_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(mermaid_content_lines))
    print(f"Raw Mermaid file saved to {mermaid_filepath}")

if __name__ == '__main__':
    # Testdaten: simulierte Objekt-Erkennungs-Ausgabe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    mock_components = [
        {'id': 1, 'type': 'start'},
        {'id': 2, 'type': 'action', 'label': 'Initialize'},
        {'id': 3, 'type': 'choice'},
        {'id': 4, 'type': 'state', 'label': 'Processing'},
        {'id': 5, 'type': 'ending'}
    ]
    
    mock_relations = [
        (1, 2, ''),
        (2, 3, ''),
        (3, 4, 'yes'),
        (3, 5, 'no'),
        (4, 5, '')
    ]
    
    graph = reconstruct_graph(mock_components, mock_relations)
    
    output_path = os.path.join(OUTPUT_DIR, "reconstructed_diagram.md")
    export_to_mermaid(graph, output_path)
    
    print("Graph mock reconstruction completed successfully.")
