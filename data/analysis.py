import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load all CSV files into DataFrames"""
    algorithm = pd.read_csv('algorithm.csv')
    routes = pd.read_csv('routes.csv')
    nodes = pd.read_csv('nodes.csv')
    edges = pd.read_csv('edges.csv')
    translation = pd.read_csv('translation.csv')
    network = pd.read_csv('network.csv')
    return algorithm, routes, nodes, edges, translation, network

def clean_data(routes, edges, algorithm, translation):
    """Clean and prepare data for analysis"""
    # Remove zero-length routes
    routes = routes[routes['distance_meters'] > 0].copy()
    
    # First merge algorithm names with edges
    edges = edges.merge(
        algorithm[['id', 'name']].rename(columns={'id': 'edge_algorithm_id', 'name': 'algorithm_name'}),
        on='edge_algorithm_id',
        how='left'
    )
    
    # Then merge algorithm names with routes through edges
    routes = routes.merge(
        edges[['id', 'algorithm_name']].rename(columns={'id': 'edge_id'}),
        on='edge_id',
        how='left'
    )
    
    # Finally merge translation names
    routes = routes.merge(
        translation[['id', 'translation']].rename(columns={'id': 'translation_id'}),
        on='translation_id',
        how='left'
    )
    
    return routes, edges

def plot_algorithm_comparison(edges):
    """Visualize route length distributions by algorithm"""
    plt.figure(figsize=(12, 8))
    
    # Boxplot
    plt.subplot(2, 2, 1)
    sns.boxplot(x='algorithm_name', y='length', data=edges,
                palette='Set2')
    plt.xlabel('Algorithm')
    plt.ylabel('Route Length (m)')
    plt.title('Distribution of Route Lengths by Algorithm')
    
    # Violin plot
    plt.subplot(2, 2, 2)
    sns.violinplot(x='algorithm_name', y='length', data=edges,
                   palette='Set2', inner='quartile')
    plt.xlabel('Algorithm')
    plt.ylabel('Route Length (m)')
    plt.title('Density Distribution of Route Lengths')
    
    # Bar chart of total lengths
    plt.subplot(2, 2, 3)
    algorithm_total = edges.groupby('algorithm_name')['length'].sum().reset_index()
    sns.barplot(x='algorithm_name', y='length', data=algorithm_total,
                palette='Set2')
    plt.xlabel('Algorithm')
    plt.ylabel('Total Network Length (m)')
    plt.title('Total Network Length by Algorithm')
    
    # Scatter plot of individual routes
    plt.subplot(2, 2, 4)
    sns.stripplot(x='algorithm_name', y='length', data=edges,
                  jitter=True, palette='Set2', alpha=0.5)
    plt.xlabel('Algorithm')
    plt.ylabel('Route Length (m)')
    plt.title('Individual Route Lengths by Algorithm')
    
    plt.tight_layout()
    plt.show()

def plot_translation_comparison(routes):
    """Visualize translation method differences"""
    plt.figure(figsize=(12, 8))
    
    # Grouped bar chart
    plt.subplot(2, 2, 1)
    trans_compare = routes.groupby(['algorithm_name', 'translation'])['distance_meters'].mean().unstack()
    trans_compare.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
    plt.xlabel('Algorithm')
    plt.ylabel('Average Route Length (m)')
    plt.title('Average Route Length by Algorithm and Translation')
    plt.xticks(rotation=0)
    plt.legend(title='Translation')
    
    # Violin plot
    plt.subplot(2, 2, 2)
    sns.violinplot(x='algorithm_name', y='distance_meters', hue='translation',
                   data=routes, palette='Set2', split=True)
    plt.xlabel('Algorithm')
    plt.ylabel('Route Length (m)')
    plt.title('Route Length Distribution by Translation Method')
    
    # Heatmap
    plt.subplot(2, 2, 3)
    heatmap_data = routes.pivot_table(values='distance_meters',
                                    index='algorithm_name',
                                    columns='translation',
                                    aggfunc='mean')
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd')
    plt.title('Average Route Length (m)\nby Algorithm and Translation')
    plt.xlabel('Translation Method')
    plt.ylabel('Algorithm')
    
    # Cumulative length comparison
    plt.subplot(2, 2, 4)
    cum_length = routes.groupby(['algorithm_name', 'translation'])['distance_meters'].sum().unstack()
    cum_length.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
    plt.xlabel('Algorithm')
    plt.ylabel('Total Route Length (m)')
    plt.title('Total Route Length by Translation Method')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.show()

def plot_network_characteristics(nodes, edges):
    """Visualize network topology characteristics"""
    plt.figure(figsize=(12, 6))
    
    # Node types
    plt.subplot(1, 2, 1)
    node_counts = nodes['node_type'].value_counts()
    plt.pie(node_counts, labels=node_counts.index, autopct='%1.1f%%',
            colors=['#66c2a5', '#fc8d62'])
    plt.title('Distribution of Node Types')
    
    # Edge length histogram
    plt.subplot(1, 2, 2)
    sns.histplot(edges['length'], bins=20, kde=True, color='#8da0cb')
    plt.xlabel('Edge Length (m)')
    plt.ylabel('Count')
    plt.title('Distribution of Edge Lengths')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run all analysis"""
    print("Loading data...")
    algorithm, routes, nodes, edges, translation, network = load_data()
    
    print("Cleaning and preparing data...")
    routes, edges = clean_data(routes, edges, algorithm, translation)
    
    # Verify we have the expected columns
    print("\nRoutes columns:", routes.columns.tolist())
    print("Edges columns:", edges.columns.tolist())
    
    print("\n=== Network Overview ===")
    print(f"Network Name: {network['name'][0]}")
    print(f"Network Type: {network['network_type'][0]}")
    print(f"Total Nodes: {len(nodes)} (Producers: {sum(nodes['node_type'] == 'producer')}, Consumers: {sum(nodes['node_type'] == 'consumer')})")
    print(f"Total Edges: {len(edges)}")
    
    print("\n=== Algorithm Summary ===")
    print(algorithm[['name', 'total_edge_length']].to_string(index=False))
    
    print("\nGenerating visualizations...")
    plot_algorithm_comparison(edges)
    plot_translation_comparison(routes)
    plot_network_characteristics(nodes, edges)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()