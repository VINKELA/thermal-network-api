import math

from django.db import transaction

from api.models import (
    Building,
    BuildingCluster,
    Cluster,
    ClusterAlgorithm,
    ClusterConfiguration,
    Network,
    Node,
    validate_project_isolation,
)


def create_complementary_cluster_configuration_from_network(base_network, *, max_radius_km=0.5, name=None, description=None):
    """
    Create complementary clusters directly from all project buildings not yet in
    the selected network.

    This is the primary interpretation of "complementary": the network defines
    the excluded set, and the remaining buildings are grouped spatially.
    """
    base_network = Network.objects.select_related("project").get(pk=base_network.pk)
    validate_project_isolation(base_network)

    max_radius_km = _positive_float(max_radius_km, 0.5)
    existing_building_ids = set(
        Node.objects
        .filter(network=base_network, building__isnull=False)
        .values_list("building_id", flat=True)
    )
    all_unconnected = list(
        Building.objects
        .filter(project=base_network.project)
        .exclude(id__in=existing_building_ids)
        .order_by("id")
    )
    valid_buildings = [
        building for building in all_unconnected
        if building.latitude is not None and building.longitude is not None
    ]
    skipped_no_coordinates = len(all_unconnected) - len(valid_buildings)
    if not valid_buildings:
        raise ValueError("No unconnected project buildings with map coordinates were found for this network.")

    groups = _radius_groups(valid_buildings, max_radius_km * 1000.0)
    with transaction.atomic():
        algo, _ = ClusterAlgorithm.objects.get_or_create(
            name="Complementary Network Radius",
            project=base_network.project,
            defaults={"algorithm": ClusterAlgorithm.ClusterAlgorithmType.RADIUS_BASED},
        )
        config = ClusterConfiguration.objects.create(
            cluster_algorithm=algo,
            name=name or f"Complementary clusters for {base_network.name}",
            description=description or f"Unconnected project buildings grouped around {base_network.name}.",
            parameters={
                "complementary_to_network_id": base_network.id,
                "method": "network_complement_radius",
                "max_radius_km": max_radius_km,
                "excluded_existing_building_count": len(existing_building_ids),
                "candidate_building_count": len(all_unconnected),
                "skipped_buildings_without_coordinates": skipped_no_coordinates,
            },
            cluster_type=ClusterConfiguration.ClusterConfigurationType.COMPLEMENTARY,
            base_network=base_network,
        )

        created = []
        for index, members in enumerate(groups, start=1):
            created.append(_create_cluster_from_members(
                config,
                None,
                members,
                fallback_name=f"Complementary Cluster {index}",
                fallback_description=f"Unconnected project buildings grouped within {max_radius_km:g} km.",
            ))

        config.parameters = {
            **(config.parameters or {}),
            "complementary_cluster_count": len(created),
            "complementary_building_count": sum(cluster.no_of_buildings or 0 for cluster in created),
        }
        config.save(update_fields=["parameters"])

    return {
        "status": "success",
        "config_id": config.id,
        "configuration": config,
        "clusters_created": len(created),
        "buildings_included": sum(cluster.no_of_buildings or 0 for cluster in created),
        "skipped_source_clusters": 0,
        "skipped_buildings_without_coordinates": skipped_no_coordinates,
    }


def create_complementary_cluster_configuration(base_network, source_configuration, *, name=None, description=None):
    """
    Build a reusable network-relative cluster configuration.

    Each output cluster contains the subset of a source cluster's buildings that
    are in the same project but not already present as building nodes in the
    selected base network.
    """
    base_network = Network.objects.select_related("project").get(pk=base_network.pk)
    source_configuration = ClusterConfiguration.objects.select_related(
        "cluster_algorithm",
        "base_network",
        "source_cluster_configuration",
    ).get(pk=source_configuration.pk)

    validate_project_isolation(base_network)
    validate_project_isolation(source_configuration)
    if source_configuration.cluster_algorithm.project_id != base_network.project_id:
        raise ValueError("The source cluster configuration must belong to the same project as the selected network.")

    existing_building_ids = set(
        Node.objects
        .filter(network=base_network, building__isnull=False)
        .values_list("building_id", flat=True)
    )

    source_clusters = list(
        Cluster.objects
        .filter(cluster_configuration=source_configuration)
        .prefetch_related("buildingcluster_set__building")
        .order_by("id")
    )
    if not source_clusters:
        raise ValueError("The source cluster configuration has no clusters to complement.")

    with transaction.atomic():
        config = ClusterConfiguration.objects.create(
            cluster_algorithm=source_configuration.cluster_algorithm,
            name=name or f"{source_configuration.name or 'Cluster configuration'} - Complement to {base_network.name}",
            description=description or f"Buildings from {source_configuration.name or source_configuration.id} not yet connected to {base_network.name}.",
            parameters={
                **(source_configuration.parameters or {}),
                "complementary_to_network_id": base_network.id,
                "source_cluster_configuration_id": source_configuration.id,
                "excluded_existing_building_count": len(existing_building_ids),
            },
            cluster_type=ClusterConfiguration.ClusterConfigurationType.COMPLEMENTARY,
            base_network=base_network,
            source_cluster_configuration=source_configuration,
        )

        created = []
        skipped = []
        assigned_building_ids = set()
        for source_cluster in source_clusters:
            members = [
                link.building
                for link in source_cluster.buildingcluster_set.all()
                if link.building_id not in existing_building_ids and link.building_id not in assigned_building_ids
            ]
            members = [
                building
                for building in members
                if building.project_id == base_network.project_id and building.latitude is not None and building.longitude is not None
            ]
            if not members:
                skipped.append(source_cluster.id)
                continue

            for building in members:
                assigned_building_ids.add(building.id)

            cluster = _create_cluster_from_members(config, source_cluster, members)
            created.append(cluster)

        config.parameters = {
            **(config.parameters or {}),
            "source_cluster_count": len(source_clusters),
            "complementary_cluster_count": len(created),
            "complementary_building_count": sum(cluster.no_of_buildings or 0 for cluster in created),
            "skipped_source_cluster_ids": skipped,
        }
        config.save(update_fields=["parameters"])

    return {
        "status": "success",
        "config_id": config.id,
        "configuration": config,
        "clusters_created": len(created),
        "buildings_included": sum(cluster.no_of_buildings or 0 for cluster in created),
        "skipped_source_clusters": len(skipped),
        "skipped_buildings_without_coordinates": 0,
    }


def _create_cluster_from_members(config, source_cluster, members, *, fallback_name=None, fallback_description=None):
    latitude, longitude, effective_radius = _cluster_geometry(members)
    source_kw = sum(_source_capacity_kw(building) for building in members)
    sink_kw = sum(_sink_capacity_kw(building) for building in members)
    status = _balance_status(source_kw, sink_kw)

    if source_cluster:
        cluster_name = f"Complementary {source_cluster.name or f'Cluster {source_cluster.id}'}"
        cluster_description = f"Unconnected members from source cluster {source_cluster.id}."
        cluster_type = source_cluster.type or "distance_based"
    else:
        cluster_name = fallback_name or "Complementary Cluster"
        cluster_description = fallback_description or "Unconnected project buildings outside the selected network."
        cluster_type = "distance_based"

    cluster = Cluster.objects.create(
        name=cluster_name,
        description=cluster_description,
        type=cluster_type,
        cluster_configuration=config,
        cluster_algorithm=config.cluster_algorithm,
        latitude=latitude,
        longitude=longitude,
        no_of_buildings=len(members),
        effective_radius=effective_radius,
        heating_demand=sink_kw,
        cooling_demand=source_kw,
        total_source_capacity=source_kw,
        total_sink_capacity=sink_kw,
        energy_balance_status=status,
    )
    BuildingCluster.objects.bulk_create([
        BuildingCluster(building=building, cluster=cluster)
        for building in members
    ])
    return cluster


def _radius_groups(buildings, radius_m):
    remaining = set(range(len(buildings)))
    groups = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        queue = [seed]
        while queue:
            current = queue.pop()
            linked = [
                idx for idx in list(remaining)
                if _haversine_m(
                    buildings[current].latitude,
                    buildings[current].longitude,
                    buildings[idx].latitude,
                    buildings[idx].longitude,
                ) <= radius_m
            ]
            for idx in linked:
                remaining.remove(idx)
                component.add(idx)
                queue.append(idx)
        groups.append([buildings[idx] for idx in sorted(component)])
    groups.sort(key=lambda group: (-len(group), min(building.id for building in group)))
    return groups


def _cluster_geometry(buildings):
    weighted_total = sum(max(_sink_capacity_kw(building), _source_capacity_kw(building), 1.0) for building in buildings)
    if weighted_total > 0:
        latitude = sum(float(building.latitude) * max(_sink_capacity_kw(building), _source_capacity_kw(building), 1.0) for building in buildings) / weighted_total
        longitude = sum(float(building.longitude) * max(_sink_capacity_kw(building), _source_capacity_kw(building), 1.0) for building in buildings) / weighted_total
    else:
        latitude = sum(float(building.latitude) for building in buildings) / len(buildings)
        longitude = sum(float(building.longitude) for building in buildings) / len(buildings)

    radius = 50.0
    if len(buildings) > 1:
        radius = max(_haversine_m(latitude, longitude, building.latitude, building.longitude) for building in buildings)
    return latitude, longitude, max(radius, 50.0)


def _is_source(building):
    source_type = str(building.waste_heat_source_type or "").strip().lower()
    load_type = str(building.load_type or "").strip().lower()
    return bool(source_type and source_type not in {"none", "nan", "null"}) or load_type == "cooling_dominant"


def _source_capacity_kw(building):
    if not _is_source(building):
        return 0.0
    return max(_float_value(building.cooling_demand), _float_value(building.heating_demand), 0.0)


def _sink_capacity_kw(building):
    if _is_source(building):
        return 0.0
    return max(_float_value(building.heating_demand), 0.0)


def _balance_status(source_kw, sink_kw):
    max_load = max(source_kw, sink_kw)
    tolerance = 0.05 * max_load if max_load > 0 else 0.0
    if source_kw > sink_kw + tolerance:
        return "excess_heat"
    if source_kw < sink_kw - tolerance:
        return "heat_deficit"
    if max_load > 0:
        return "balanced"
    return "unknown"


def _float_value(value):
    try:
        if value is None:
            return 0.0
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return 0.0
        return number
    except (TypeError, ValueError):
        return 0.0


def _positive_float(value, fallback):
    number = _float_value(value)
    return number if number > 0 else fallback


def _haversine_m(lat1, lon1, lat2, lon2):
    radius_m = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
