import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Diese Funktion erstellt und speichert den Konnektivitätsgraphen für eine gegebene .bxr-Datei
def plot_connectivity_graph(filepath, output_dir, connectivity_threshold, 
                           max_connections_per_node, betweenness_threshold, dpi=600):
    betweenness_threshold = betweenness_threshold / 1000
    # Speicherpfad für den Konnektivitätsgraphen
    output_path = os.path.join(output_dir, f'ConnGraph_{os.path.splitext(os.path.basename(filepath))[0]}.png')

    # Überprüfung, ob der Graph bereits existiert
    if os.path.exists(output_path):
        print(f"Überspringe '{output_path}', da der Konnektivitätsgraph bereits existiert.")
        return None

    # Pfad zur .npz-Datei basierend auf dem .bxr-Dateinamen
    npz_file_path = os.path.splitext(filepath)[0] + '_features.npz'

    # Überprüfen, ob die .npz-Datei existiert
    if not os.path.exists(npz_file_path):
        print(f"Warnung: '{npz_file_path}' nicht gefunden. Überspringe Datei {os.path.basename(filepath)}.")
        return None

    # Laden der Daten aus der .npz-Datei
    data = np.load(npz_file_path, allow_pickle=True)
    pearson_corr_matrix = data['pearson_corr_matrix']
    pearson_corr_matrix = np.where(pearson_corr_matrix > connectivity_threshold, pearson_corr_matrix, 0) #NEW
    unique_channels = data['unique_channels']
    num_channels = data['num_channels']

    # Netzwerk erstellen
    G = nx.Graph()
    for i in range(num_channels):
        neighbors = [(j, pearson_corr_matrix[i, j]) for j in range(num_channels) if i != j]
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)
        top_connections = [(unique_channels[i], unique_channels[j], weight) 
                           for j, weight in neighbors[:max_connections_per_node]]
        G.add_weighted_edges_from(top_connections)

    if len(G.nodes) == 0:
        print(f"Warnung: Kein Netzwerk für {os.path.basename(filepath)} generiert.")
        return None

    # Betweenness-Zentralität berechnen
    betweenness = nx.betweenness_centrality(G)
    filtered_nodes = [node for node, centrality in betweenness.items() if centrality >= betweenness_threshold]
    G_filtered = G.subgraph(filtered_nodes).copy()
    
    if len(G_filtered.nodes) == 0:
        print(f"Warnung: Nach dem Filtern keine Knoten für {os.path.basename(filepath)} übrig.")
        return None

    betweenness_filtered = nx.betweenness_centrality(G_filtered)
    node_sizes = [1500 * betweenness_filtered.get(int(channel), 0) + 1 for channel in G_filtered.nodes]

    # Positionen für die Knoten
    pos = {channel: (channel % 64, channel // 64) for channel in G_filtered.nodes}

    # Netzwerk plotten
    plt.figure(figsize=(12, 12))
    edges = [(u, v, d) for u, v, d in G_filtered.edges(data=True) if d['weight'] > connectivity_threshold]

    #edges = list(G_filtered.edges(data=True))
    if len(edges) == 0:
        print(f"Warnung: Keine Kanten für {os.path.basename(filepath)} vorhanden. Überspringe Datei.")
        plt.close()
        return None
    weights = [d['weight'] for _, _, d in edges]
    #weights = [e[2]['weight'] for e in edges]
    nx.draw_networkx_nodes(G_filtered, pos, node_size=node_sizes, node_color='red')
    edge_collection = nx.draw_networkx_edges(G_filtered, pos, edgelist=edges, 
                                             edge_color=weights, 
                                             edge_cmap=plt.cm.viridis, 
                                             edge_vmin=0, edge_vmax=1, width=0.1)

    plt.colorbar(edge_collection, label='Pearson-Correlation').set_label('Pearson-Correlation', fontsize=12)
    plt.title(f'{os.path.basename(filepath)}', fontsize=16)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path

def create_summary_plot(image_paths, summary_path, dpi=300):
    if not image_paths:
        print(f"Keine Bilder zum Zusammenfügen für {os.path.basename(summary_path)} gefunden.")
        return

    num_images = len(image_paths)
    fig_width = 5 * num_images
    fig_height = 5

    plt.figure(figsize=(fig_width, fig_height))

    for idx, img_path in enumerate(image_paths):
        img = plt.imread(img_path)
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(img_path).replace('_connectivity_graph.png', ''))
    
    plt.tight_layout(pad=0.5)
    plt.savefig(summary_path, format='png', dpi=dpi, bbox_inches='tight')
    plt.close()

def process_all_bxr_files_in_directory(root_dir, connectivity_threshold, 
                                       max_connections_per_node, betweenness_threshold):
    connectivity_output_dir = os.path.join(root_dir, 'ConnectivityGraphs')
    os.makedirs(connectivity_output_dir, exist_ok=True)

    for subdir, _, files in os.walk(root_dir):
        folder_name = os.path.basename(subdir)
        if folder_name.startswith('ID2024-'):
            print(f'Verarbeite Ordner: {folder_name}')
            individual_plots = []
            for file in files:
                if file.endswith('_NB.bxr'):
                    filepath = os.path.join(subdir, file)
                    print(f'  Verarbeite Datei: {file}')
                    plot_path = plot_connectivity_graph(
                        filepath, 
                        connectivity_output_dir, 
                        connectivity_threshold=connectivity_threshold,
                        max_connections_per_node=max_connections_per_node,
                        betweenness_threshold=betweenness_threshold
                    )
                    if plot_path:
                        individual_plots.append(plot_path)
            
            if individual_plots:
                summary_filename = f'{folder_name}_summary_connectivity_graphs.png'
                summary_path = os.path.join(connectivity_output_dir, summary_filename)
                print(f'  Erstelle Übersichtsplot: {summary_filename}')
                create_summary_plot(individual_plots, summary_path)
            else:
                print(f'  Keine gültigen Konnektivitätsgraphen für {folder_name} gefunden.')

def main():
    root_dir = "data"
    connectivity_threshold = 0.2
    max_connections_per_node = 20
    betweenness_threshold = 0.8 # Hauptparameter um geplottete Netzwerkdichte zu definieren. Zur Vergleichbarkeit pro Experiment den Parameter nicht ändern!

    if not os.path.isdir(root_dir):
        print('Der angegebene Pfad ist kein gültiges Verzeichnis.')
        return

    process_all_bxr_files_in_directory(
        root_dir,
        connectivity_threshold=connectivity_threshold,
        max_connections_per_node=max_connections_per_node,
        betweenness_threshold=betweenness_threshold
    )

if __name__ == "__main__":
    main()
