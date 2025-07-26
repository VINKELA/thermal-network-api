from django.db import models
from django.utils.translation import gettext_lazy as _

# This is a base class for geographical locations
class Location(models.Model):
    latitude = models.FloatField(blank=True, null=True)
    longitude = models.FloatField(blank=True, null=True)
    class Meta:
        abstract = True
class Country(Location):
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=3, unique=True)

    def __str__(self):
        return self.name
class Province(Location):
    name = models.CharField(max_length=100)
    country = models.ForeignKey(Country, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.name}, {self.country}"
class City(Location):
    name = models.CharField(max_length=100)
    province = models.ForeignKey(Province, on_delete=models.CASCADE, blank=True, null=True)
    country = models.ForeignKey(Country, on_delete=models.CASCADE, blank=True, null=True)
    population = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return f"{self.name}, {self.province if self.province else self.country}"
class Project(models.Model):
    province = models.ForeignKey(Province, on_delete=models.CASCADE, blank=True, null=True)
    country = models.ForeignKey(Country, on_delete=models.CASCADE)
    city = models.ForeignKey(City, on_delete=models.CASCADE, blank=True, null=True)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    start_date = models.DateField()
    end_date = models.DateField(blank=True, null=True)
    budget = models.DecimalField(max_digits=12, decimal_places=2, blank=True, null=True)
    status = models.CharField(max_length=50, choices=[
        ('planned', 'Planned'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled')
    ], default='planned')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
class Building(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    building_type = models.CharField(max_length=100, choices=[
        ('residential', 'Residential'),
        ('commercial', 'Commercial'),
        ('industrial', 'Industrial'),
        ('public', 'Public'),
        ('other', 'Other')
    ])
    floor_area = models.FloatField(blank=True, null=True)  # in square meters
    heating_demand = models.FloatField(blank=True, null=True)  # in kW
    cooling_demand = models.FloatField(blank=True, null=True)  # in kW


# Network and Nodes
class Network(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    network_type = models.CharField(max_length=100, choices=[
        ('district_heating', 'District Heating'),
        ('district_cooling', 'District Cooling'),
        ('combined', 'Combined Heating and Cooling')
    ])
    total_length = models.FloatField(blank=True, null=True)  # in meters
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.project})"
class Node(Location):
    network = models.ForeignKey(Network, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, blank=True, null=True)
    node_type = models.CharField(max_length=100, choices=[
        ('supply', 'Supply'),
        ('return', 'Return'),
        ('junction', 'Junction'),
        ('consumer', 'Consumer'),
        ('producer', 'Producer'),
        ('storage', 'Storage')
    ])
class EdgeAlgorithm(models.Model):
    class EdgeAlgorithmType(models.TextChoices):
        DELAUNAY = 'delaunay', _('Delaunay')
        KNN = 'knn', _('KNN')
        DISTANCE_THRESHOLD = 'distance_threshold', _('Distance Threshold')
        MST = 'mst', _('Minimum Spanning Tree')
        GABRIEL = 'gabriel', _('Gabriel Graph')
        BETA_SKELETON = 'beta_skeleton', _('Beta Skeleton')
        RNG = 'rng', _('Relative Neighborhood Graph')
        TSP = 'tsp', _('Traveling Salesman Problem')
        CUSTOM = 'custom', _('Custom')
    algorithm = models.CharField(
        max_length=32,
        choices=EdgeAlgorithmType.choices,
        default=EdgeAlgorithmType.DELAUNAY
    )
    network = models.ForeignKey(Network, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    distance_threshold = models.FloatField(blank=True, null=True)  # for distance threshold method
    description = models.TextField(blank=True, null=True)
    is_default = models.BooleanField(default=False)  # Indicates if this is the default algorithm
    parameters = models.JSONField(blank=True, null=True, help_text="Additional parameters for the algorithm")
    total_edge_length = models.FloatField(blank=True, null=True, help_text="Total length of all edges generated by this algorithm (in meters)")

    def __str__(self):
        return self.name
class Edge(models.Model):
    network = models.ForeignKey(Network, on_delete=models.CASCADE)
    edge_algorithm = models.ForeignKey(EdgeAlgorithm, on_delete=models.SET_NULL, null=True, blank=True, related_name='edges')
    name = models.CharField(max_length=255, blank=True, null=True)
    start_node = models.ForeignKey(Node, related_name='outgoing_edges', on_delete=models.CASCADE)
    end_node = models.ForeignKey(Node, related_name='incoming_edges', on_delete=models.CASCADE)
    length = models.FloatField()  # in meters
    is_default = models.BooleanField(default=False)  # Indicates if this is the default edge
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    def __str__(self):
        return f"{self.start_node} to {self.end_node}"
class EdgeTranslation(models.Model):
    class EdgeTranslationType(models.TextChoices):
        WALKING = 'walking', _('Walking')
        BICYCLING = 'bicycling', _('Bicycling')
        DRIVING = 'driving', _('Driving')
        TRANSIT = 'transit', _('Transit')  # Fixed from 'mst'
        CUSTOM = 'custom', _('Custom')
        
    translation = models.CharField(
        max_length=32,
        choices=EdgeTranslationType.choices,
        default=EdgeTranslationType.DRIVING
    )
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    is_default = models.BooleanField(default=False)
    parameters = models.JSONField(blank=True, null=True)
    total_edge_length = models.FloatField(blank=True, null=True)
    algorithm = models.ForeignKey(EdgeAlgorithm, on_delete=models.CASCADE, null=True)
    def __str__(self):
        return self.name
class Route(models.Model):
    id = models.AutoField(primary_key=True)
    edge = models.ForeignKey(Edge, related_name='edge', on_delete=models.CASCADE)
    translation = models.ForeignKey(EdgeTranslation, related_name='routes', on_delete=models.CASCADE, null=True, blank=True)
    sub_route = models.ForeignKey('self', related_name='routes', on_delete=models.CASCADE, null=True, blank=True)
    polyline = models.TextField()  # Encoded path
    distance_meters = models.FloatField()
    duration_seconds = models.FloatField()


## Equipment and Components
class Pump(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)

    pump_type = models.CharField(max_length=100, choices=[
        ('circulation', 'Circulation'),
        ('booster', 'Booster'),
        ('submersible', 'Submersible'),
        ('other', 'Other')
    ])
    flow_rate = models.FloatField()  # in m³/h
    head = models.FloatField()  # in meters
    power = models.FloatField()  # in kW
    efficiency = models.FloatField(blank=True, null=True)  # as percentage
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')
class Valve(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    valve_type = models.CharField(max_length=100, choices=[
        ('gate', 'Gate'),
        ('ball', 'Ball'),
        ('butterfly', 'Butterfly'),
        ('check', 'Check'),
        ('pressure_reducing', 'Pressure Reducing'),
        ('other', 'Other')
    ])
    diameter = models.FloatField()  # in mm
    status = models.CharField(max_length=50, choices=[
        ('open', 'Open'),
        ('closed', 'Closed'),
        ('partially_open', 'Partially Open'),
        ('malfunction', 'Malfunction')
    ], default='closed')
class HeatPump(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    heat_pump_type = models.CharField(max_length=100, choices=[
        ('air_source', 'Air Source'),
        ('ground_source', 'Ground Source'),
        ('water_source', 'Water Source'),
        ('hybrid', 'Hybrid')
    ])
    heating_capacity = models.FloatField()  # in kW
    cooling_capacity = models.FloatField(blank=True, null=True)  # in kW
    cop = models.FloatField()  # Coefficient of Performance
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')
class HeatExchanger(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    heat_exchanger_type = models.CharField(max_length=100, choices=[
        ('plate', 'Plate'),
        ('shell_tube', 'Shell & Tube'),
        ('double_pipe', 'Double Pipe'),
        ('other', 'Other')
    ])
    heat_transfer_area = models.FloatField()  # in m²
    effectiveness = models.FloatField()  # as fraction (0-1)
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')
class Junction(Location):
    name = models.CharField(max_length=255, null=True, blank=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    edge = models.ForeignKey(Edge, on_delete=models.CASCADE)
    junction_type = models.CharField(max_length=100, choices=[
        ('tee', 'Tee'),
        ('cross', 'Cross'),
        ('y', 'Y'),
        ('other', 'Other')
    ])
    diameter = models.FloatField()  # in mm
    material = models.CharField(max_length=100)
class Pipe(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    edge = models.ForeignKey(Edge, on_delete=models.CASCADE)
    junction = models.ForeignKey(Junction, on_delete=models.CASCADE)
    diameter = models.FloatField()  # in mm
    length = models.FloatField()  # in meters
    material = models.CharField(max_length=100)
    insulation_type = models.CharField(max_length=100, blank=True, null=True)
    insulation_thickness = models.FloatField(blank=True, null=True)  # in mm
    max_pressure = models.FloatField()  # in bar
    max_temperature = models.FloatField()  # in °C
    installation_date = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')
class PVSystem(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    peak_power = models.FloatField()  # in kWp
    area = models.FloatField()  # in m²
    efficiency = models.FloatField()  # as percentage
    orientation = models.FloatField()  # in degrees (0-360)
    tilt_angle = models.FloatField()  # in degrees
    installation_date = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')
class GeothermalSystem(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    system_type = models.CharField(max_length=100, choices=[
        ('closed_loop', 'Closed Loop'),
        ('open_loop', 'Open Loop'),
        ('direct_exchange', 'Direct Exchange')
    ])
    depth = models.FloatField()  # in meters
    number_of_boreholes = models.IntegerField()
    total_heating_capacity = models.FloatField()  # in kW
    total_cooling_capacity = models.FloatField(blank=True, null=True)  # in kW
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')
class Boiler(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    boiler_type = models.CharField(max_length=100, choices=[
        ('gas', 'Gas'),
        ('oil', 'Oil'),
        ('biomass', 'Biomass'),
        ('electric', 'Electric'),
        ('hybrid', 'Hybrid')
    ])
    capacity = models.FloatField()  # in kW
    efficiency = models.FloatField()  # as percentage
    fuel_type = models.CharField(max_length=100)
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')
class ThermalStorage(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    storage_type = models.CharField(max_length=100, choices=[
        ('hot_water', 'Hot Water'),
        ('phase_change', 'Phase Change'),
        ('sensible_heat', 'Sensible Heat'),
        ('other', 'Other')
    ])
    capacity = models.FloatField()  # in kWh
    volume = models.FloatField()  # in m³
    max_temperature = models.FloatField()  # in °C
    min_temperature = models.FloatField()  # in °C
    insulation_type = models.CharField(max_length=100)
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')


# Additional relevant domains
class Sensor(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    sensor_type = models.CharField(max_length=100, choices=[
        ('temperature', 'Temperature'),
        ('pressure', 'Pressure'),
        ('flow', 'Flow'),
        ('humidity', 'Humidity'),
        ('energy', 'Energy'),
        ('other', 'Other')
    ])
    model = models.CharField(max_length=100)
    serial_number = models.CharField(max_length=100, unique=True)
    installation_date = models.DateField()
    last_calibration = models.DateField(blank=True, null=True)
    accuracy = models.FloatField(blank=True, null=True)
    unit = models.CharField(max_length=20)
class Meter(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    meter_type = models.CharField(max_length=100, choices=[
        ('heat', 'Heat'),
        ('cooling', 'Cooling'),
        ('electricity', 'Electricity'),
        ('water', 'Water'),
        ('gas', 'Gas')
    ])
    model = models.CharField(max_length=100)
    serial_number = models.CharField(max_length=100, unique=True)
    installation_date = models.DateField()
    last_calibration = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')