import numpy as np
from django.db import transaction
from django.db.models import Sum
from api.models import (
    Building, Cluster, ClusterAlgorithm, ClusterConfiguration, 
    BuildingCluster, LoadProfile
)

class SourceAnchoredClusteringService:
    """
    Groups buildings into thermal clusters based on distance to waste heat sources.
    Assumes Seasonal Thermal Storage, using Sum() for Annual Net Energy Balance.
    """
    def __init__(self, project_id, max_radius_km=1, name=None):
        self.project_id = project_id
        self.max_radius_km = float(max_radius_km)
        self.name = (name or "").strip()
        self.EARTH_RADIUS_KM = 6371.0

    def _haversine_matrix(self, source_coords, sink_coords):
        s_lat = np.radians(source_coords[:, 0])
        s_lon = np.radians(source_coords[:, 1])
        k_lat = np.radians(sink_coords[:, 0])
        k_lon = np.radians(sink_coords[:, 1])

        dlat = k_lat[:, np.newaxis] - s_lat
        dlon = k_lon[:, np.newaxis] - s_lon

        a = np.sin(dlat / 2.0)**2 + np.cos(k_lat[:, np.newaxis]) * np.cos(s_lat) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return self.EARTH_RADIUS_KM * c

    def _haversine_distance_km(self, a_lat, a_lon, b_lat, b_lon):
        coords = np.array([[[float(a_lat), float(a_lon)]], [[float(b_lat), float(b_lon)]]])
        return float(self._haversine_matrix(coords[0], coords[1])[0][0])

    def _valid_buildings(self):
        return [
            b for b in Building.objects.filter(project_id=self.project_id)
            if b.latitude is not None and b.longitude is not None
        ]

    def _is_source(self, building):
        source_type = str(building.waste_heat_source_type or '').strip().lower()
        load_type = str(building.load_type or '').strip().lower()
        return bool(source_type and source_type not in {'none', 'nan', 'null'}) or load_type == 'cooling_dominant'

    def _cluster_geometry(self, buildings):
        if not buildings:
            return None, None, None

        lat = sum(float(b.latitude) for b in buildings) / len(buildings)
        lon = sum(float(b.longitude) for b in buildings) / len(buildings)
        radius_m = 100.0
        if len(buildings) > 1:
            radius_m = max(
                self._haversine_distance_km(lat, lon, b.latitude, b.longitude) * 1000
                for b in buildings
            )
        return lat, lon, max(radius_m, 50.0)

    def _calculate_balance(self, cluster, building_ids):
        # 1. Aggregate SINK Load (Total Annual Heating Demand in kWh)
        total_sink_kwh = LoadProfile.objects.filter(
            building_id__in=building_ids,
            profile_type='heating'
        ).aggregate(Sum('value'))['value__sum'] or 0.0

        # 2. Aggregate SOURCE Load (Total Annual Waste Heat Supply in kWh)
        total_source_kwh = LoadProfile.objects.filter(
            building_id__in=building_ids,
            profile_type='waste_heat'
        ).aggregate(Sum('value'))['value__sum'] or 0.0

        # 3. Determine Status (5% tolerance)
        max_load = max(total_sink_kwh, total_source_kwh)
        tolerance = 0.05 * max_load if max_load > 0 else 0
        
        if total_source_kwh > (total_sink_kwh + tolerance):
            status = 'excess_heat'
        elif total_source_kwh < (total_sink_kwh - tolerance):
            status = 'heat_deficit'
        else:
            status = 'balanced'

        # 4. Save to Cluster
        cluster.energy_balance_status = status
        cluster.total_source_capacity = total_source_kwh
        cluster.total_sink_capacity = total_sink_kwh
        cluster.heating_demand = total_sink_kwh
        cluster.cooling_demand = total_source_kwh
        
        cluster.save(update_fields=[
            'energy_balance_status', 'total_source_capacity', 'total_sink_capacity',
            'heating_demand', 'cooling_demand'
        ])
        
        return status, total_source_kwh, total_sink_kwh

    def perform_clustering(self):
        buildings = self._valid_buildings()
        if not buildings:
            return {"status": "error", "message": "No buildings with map coordinates were found for this project."}

        sources, sinks = [], []
        
        for b in buildings:
            is_source = self._is_source(b)
            lat = float(b.latitude)
            lon = float(b.longitude)
            data = {'obj': b, 'coords': [lat, lon]}
            
            if is_source: sources.append(data)
            else: sinks.append(data)

        if not sources:
            return self._create_unbalanced_cluster(sinks, "No Sources Found")

        source_coords = np.array([s['coords'] for s in sources])
        assignments = {i: [] for i in range(len(sources))}
        unclustered_sinks = []

        # Phase 1: Assign Sinks to Sources
        if len(sinks) > 0:
            sink_coords = np.array([s['coords'] for s in sinks])
            dist_matrix_km = self._haversine_matrix(source_coords, sink_coords)
            
            for sink_idx in range(len(sinks)):
                closest_source_idx = np.argmin(dist_matrix_km[sink_idx])
                min_dist_km = dist_matrix_km[sink_idx][closest_source_idx]

                if min_dist_km <= self.max_radius_km:
                    assignments[closest_source_idx].append(sink_idx)
                else:
                    unclustered_sinks.append(sink_idx)

        # Phase 2: Absorb Lonely Sources
        active_source_indices = [i for i, assigned in assignments.items() if len(assigned) > 0]
        lonely_source_indices = [i for i, assigned in assignments.items() if len(assigned) == 0]
        absorbed_sources_map = {i: [] for i in range(len(sources))}
        absorbed_indices_set = set()

        if active_source_indices and lonely_source_indices:
            active_coords = np.array([sources[i]['coords'] for i in active_source_indices])
            lonely_coords = np.array([sources[i]['coords'] for i in lonely_source_indices])
            dist_lonely_active = self._haversine_matrix(active_coords, lonely_coords)
            
            for i, lonely_idx in enumerate(lonely_source_indices):
                closest_active_local_idx = np.argmin(dist_lonely_active[i])
                dist_to_active = dist_lonely_active[i][closest_active_local_idx]
                if dist_to_active <= self.max_radius_km:
                    target_active_idx = active_source_indices[closest_active_local_idx]
                    absorbed_sources_map[target_active_idx].append(lonely_idx)
                    absorbed_indices_set.add(lonely_idx)

        # Phase 3: Database Commit
        with transaction.atomic():
            algo, _ = ClusterAlgorithm.objects.get_or_create(
                name="Source-Anchored (Haversine)",
                project_id=self.project_id,
                defaults={'algorithm': 'radius_based'}
            )
            config = ClusterConfiguration.objects.create(
                cluster_algorithm=algo,
                name=self.name or f"Source-Anchored: {self.max_radius_km:g} km",
                parameters={"max_radius_km": self.max_radius_km},
                description="Clusters centered on sources."
            )

            results_summary = []
            
            for source_idx in range(len(sources)):
                if source_idx in absorbed_indices_set: continue
                
                leader_obj = sources[source_idx]['obj']
                cluster_members = [leader_obj]
                
                for sink_i in assignments[source_idx]: cluster_members.append(sinks[sink_i]['obj'])
                for absorbed_i in absorbed_sources_map[source_idx]: cluster_members.append(sources[absorbed_i]['obj'])
                latitude, longitude, effective_radius = self._cluster_geometry(cluster_members)

                cluster = Cluster.objects.create(
                    name=f"Cluster {leader_obj.id} (Source Anchored)",
                    type='distance_based',
                    cluster_configuration=config,
                    latitude=latitude,
                    longitude=longitude,
                    no_of_buildings=len(cluster_members),
                    effective_radius=effective_radius
                )

                building_ids = []
                for b in cluster_members:
                    BuildingCluster.objects.create(building=b, cluster=cluster)
                    building_ids.append(b.id)
                
                status, t_source, t_sink = self._calculate_balance(cluster, building_ids)
                results_summary.append({"cluster_id": cluster.id, "status": status})

            # Handle isolated sinks (The Ghost Sinks Fix)
            for sink_idx in unclustered_sinks:
                lonely_sink_obj = sinks[sink_idx]['obj']
                latitude, longitude, effective_radius = self._cluster_geometry([lonely_sink_obj])
                cluster = Cluster.objects.create(
                    name=f"Isolated Deficit Cluster {lonely_sink_obj.id}",
                    type='distance_based',
                    cluster_configuration=config,
                    latitude=latitude,
                    longitude=longitude,
                    no_of_buildings=1,
                    effective_radius=effective_radius
                )
                
                BuildingCluster.objects.create(building=lonely_sink_obj, cluster=cluster)
                status, t_source, t_sink = self._calculate_balance(cluster, [lonely_sink_obj.id])
                results_summary.append({"cluster_id": cluster.id, "status": status})

        return {"status": "success", "config_id": config.id, "clusters_created": len(results_summary), "details": results_summary}

    def _create_unbalanced_cluster(self, sinks, reason):
        with transaction.atomic():
            algo, _ = ClusterAlgorithm.objects.get_or_create(
                name="Fallback",
                project_id=self.project_id,
                defaults={'algorithm': ClusterAlgorithm.ClusterAlgorithmType.RADIUS_BASED},
            )
            config = ClusterConfiguration.objects.create(
                cluster_algorithm=algo,
                name=self.name or "Unbalanced Run",
                parameters={"max_radius_km": self.max_radius_km},
                description=reason,
            )
            members = [s['obj'] for s in sinks]
            latitude, longitude, effective_radius = self._cluster_geometry(members)
            cluster = Cluster.objects.create(
                name="All Buildings (Unbalanced)",
                type='distance_based',
                cluster_configuration=config,
                no_of_buildings=len(sinks),
                latitude=latitude,
                longitude=longitude,
                effective_radius=effective_radius,
            )
            ids = []
            for s in sinks:
                BuildingCluster.objects.create(building=s['obj'], cluster=cluster)
                ids.append(s['obj'].id)
            self._calculate_balance(cluster, ids)
        return {"status": "warning", "message": reason, "config_id": config.id}
