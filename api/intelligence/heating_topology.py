"""Fixed radial topology snapshots for demand sensitivity; never silently prune loops."""
import hashlib
import json
import math
import networkx as nx
from api.models import Network, Node, Edge, EdgeConnectionAlgorithm, Building, BuildingCluster


def validate_radial_case(nodes, edges, buildings):
    graph = nx.Graph()
    graph.add_nodes_from(n['id'] for n in nodes)
    roots = [n['id'] for n in nodes if n['kind'] == 'producer']
    if len(roots) != 1:
        raise ValueError('Select a radial case with exactly one producer. Multiple sources need a dispatch model.')
    for edge in edges:
        a, b = edge['start'], edge['end']
        if a not in graph or b not in graph or a == b or graph.has_edge(a, b):
            raise ValueError('Topology contains invalid, self-loop or parallel edges; select one connection layout.')
        if not math.isfinite(edge['length_m']) or edge['length_m'] <= 0:
            raise ValueError('Every branch requires a positive finite length in metres.')
        graph.add_edge(a, b, record=edge)
    if not edges or not nx.is_tree(graph):
        raise ValueError('The selected layout must be connected and radial. Create/select branch edges; loops and isolated nodes are not discarded.')
    assigned = [bid for n in nodes for bid in n['building_ids']]
    if len(assigned) != len(set(assigned)) or set(assigned) != {b['id'] for b in buildings}:
        raise ValueError('Each consumer building must be assigned exactly once, directly or through a cluster.')
    if next(n for n in nodes if n['id'] == roots[0])['building_ids']:
        raise ValueError('The producer cannot also carry consumer demand in this study.')
    directed = nx.bfs_tree(graph, roots[0])
    downstream = {n['id']: list(n['building_ids']) for n in nodes}
    for node in nx.dfs_postorder_nodes(directed, roots[0]):
        for child in directed.successors(node):
            downstream[node].extend(downstream[child])
    branches = []
    for a, b in directed.edges:
        branches.append({**graph[a][b]['record'], 'start': a, 'end': b, 'building_ids': downstream[b]})
    case = dict(nodes=nodes, branches=branches, buildings=buildings, root=roots[0])
    case['sha256'] = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
    return case


def demonstration_case():
    """Assumed geometry and areas, not a survey of actual buildings or a saved network."""
    specs = [('Home A','residential',180), ('Home B','residential',240), ('Apartments','residential',1200),
             ('Office A','commercial',900), ('Office B','commercial',1400), ('Office C','commercial',600)]
    buildings = [dict(id=i+1, name=name, building_type=kind, floor_area=area, volume_m3=area*3,
                     input_status={'area':'assumed demonstration area', 'volume':'assumed area × 3 m', 'type':'assumed', 'vintage':'missing', 'envelope':'estimated study input'})
                 for i,(name,kind,area) in enumerate(specs)]
    nodes = [dict(id=0,name='Heat source',kind='producer',building_ids=[]),dict(id=7,name='District junction',kind='junction',building_ids=[])]
    nodes += [dict(id=b['id'],name=b['name'],kind='consumer',building_ids=[b['id']]) for b in buildings]
    edges = [dict(id=1,name='District trunk',start=0,end=7,length_m=120.,length_source='assumed')]
    edges += [dict(id=i+2,name=b['name'],start=7,end=b['id'],length_m=float(30+i*10),length_source='assumed') for i,b in enumerate(buildings)]
    return {**validate_radial_case(nodes,edges,buildings), 'name':'Illustrative six-building Canadian district', 'demonstration':True}


def snapshot_saved_network(project_id, network_id, connection_id=None):
    network = Network.objects.filter(pk=network_id,project_id=project_id).first()
    if network is None:
        raise ValueError('Select a network belonging to this project.')
    rows = list(Node.objects.filter(network=network).select_related('building','cluster'))
    edges = Edge.objects.filter(network=network).order_by('id')
    if connection_id:
        connection = EdgeConnectionAlgorithm.objects.filter(pk=connection_id,network=network).first()
        if connection is None:
            raise ValueError('Connection layout does not belong to this network.')
        edges = edges.filter(edge_connections_edge__edge_connection_algorithm=connection).distinct()
    nodes, ids = [], []
    for node in rows:
        if node.node_type in ('storage','boiler','network'):
            raise ValueError('Storage, auxiliary boilers and nested networks require dispatch modelling and are not supported here.')
        bids = []
        if node.node_type != 'producer':
            if node.building_id and node.cluster_id:
                raise ValueError('A consumer node cannot represent both a building and a cluster.')
            if node.cluster_id:
                bids = list(BuildingCluster.objects.filter(cluster_id=node.cluster_id).values_list('building_id',flat=True))
            elif node.building_id:
                bids = [node.building_id]
            elif node.node_type not in ('junction','other'):
                raise ValueError(f'Consumer node {node.id} has no building or cluster assignment.')
        ids.extend(bids)
        nodes.append(dict(id=node.id,name=node.name or f'Node {node.id}',kind=node.node_type,building_ids=bids))
    buildings = []
    for b in Building.objects.filter(id__in=ids).order_by('id'):
        if b.project_id != project_id or b.waste_heat_source_type:
            raise ValueError('Consumer assignment contains a cross-project building or a heat source.')
        if b.floor_area is None or not math.isfinite(b.floor_area) or b.floor_area <= 0:
            raise ValueError(f'Building {b.id} requires a positive floor area.')
        if b.building_type not in ('residential','commercial'):
            raise ValueError(f'Building {b.id}: this study supports residential and commercial types only.')
        volume = b.volume_m3
        valid_volume = volume is not None and math.isfinite(volume) and volume > 0
        buildings.append(dict(id=b.id,name=b.name or f'Building {b.id}',building_type=b.building_type,
                             floor_area=b.floor_area,volume_m3=volume if valid_volume else b.floor_area*3,
                             input_status={'area':'available database value; conditioned-area interpretation unverified',
                                           'volume':'available' if valid_volume else 'estimated: area × 3 m',
                                           'type':'available','vintage':'not used','envelope':'estimated study input'}))
    records = [dict(id=e.id,name=e.name or f'Branch {e.id}',start=e.start_node_id,end=e.end_node_id,
                    length_m=float(e.length),length_source='Edge.length, fixed snapshot; route alternatives not substituted') for e in edges]
    if len(buildings)>100 or len(records)>150:
        raise ValueError('Use a manageable case with at most 100 buildings and 150 branches.')
    return {**validate_radial_case(nodes,records,buildings),'name':network.name,'network_id':network.id,'demonstration':False}
