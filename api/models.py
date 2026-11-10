from django.db import models
from django.utils.translation import gettext_lazy as _
from django.db import models
from django.db.models import JSONField 
from django.core.exceptions import ValidationError
from django.conf import settings
from django.db.models.signals import pre_save
from django.dispatch import receiver
from .utils.clustering_algorithms import *
import networkx as nx
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
    country = models.ForeignKey(Country, on_delete=models.CASCADE, blank=True, null=True)
    city = models.ForeignKey(City, on_delete=models.CASCADE, blank=True, null=True)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    start_date = models.DateField(blank=True, null=True)
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


class ProjectMembership(models.Model):
    class Role(models.TextChoices):
        OWNER = 'owner', _('Owner')
        EDITOR = 'editor', _('Editor')
        VIEWER = 'viewer', _('Viewer')

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='memberships')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='project_memberships')
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.VIEWER)
    invited_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='project_memberships_created',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['project', 'user'], name='unique_project_user_membership'),
        ]
        indexes = [
            models.Index(fields=['user', 'project']),
            models.Index(fields=['project', 'role']),
        ]

    @property
    def can_edit(self):
        return self.role in {self.Role.OWNER, self.Role.EDITOR}

    def __str__(self):
        return f"{self.user} - {self.project} ({self.role})"


class Building(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    building_type = models.CharField(max_length=100, choices=[
        ('residential', 'Residential'),
        ('commercial', 'Commercial'),
        ('industrial', 'Industrial'),
        ('public', 'Public'),
        ('other', 'Other')
    ], default='other')  
    fsa = models.CharField(max_length=100, blank=True, null=True)  # Forward Sortation Area
    tag = models.CharField(
        max_length=120,
        blank=True,
        null=True,
        db_index=True,
        help_text="Optional user-defined grouping tag for bulk network selection."
    )
    load_type = models.CharField(max_length=100, choices=[
        ('cooling_dominant', 'cooling_dominant'),
        ('heating_dominant', 'heating_dominant')
    ], default='heating_dominant')
    floor_area = models.FloatField(blank=True, null=True)  # in square meters
    heating_demand = models.FloatField(blank=True, null=True)  # in kW
    cooling_demand = models.FloatField(blank=True, null=True)  # in kW
    height = models.FloatField(blank=True, null=True)  # in meters
    number_of_floors = models.IntegerField(blank=True, null=True)
# New Field: Categorizes the building for waste heat estimation
    WASTE_HEAT_SOURCE_CHOICES = [
        ('data_centre', 'Data Centre'),
        ('grocery_store', 'Grocery Store'),
        ('cold_storage', 'Cold Storage'),
        ('ice_arena', 'Ice Arena'),
        ('space_cooling', 'Space Cooling (Large Building)'),
        ('restaurant', 'Restaurant'),
        ('wastewater', 'Wastewater Treatment Plant'),
    ]
    
    waste_heat_source_type = models.CharField(
        max_length=50, 
        choices=WASTE_HEAT_SOURCE_CHOICES, 
        null=True, 
        blank=True,
        help_text="The primary process generating recoverable waste heat."
    )
        # You may need these fields if they aren't already present
    volume_m3 = models.FloatField(null=True, blank=True, help_text="Required for Cold Storage estimation")
    is_year_round_operation = models.BooleanField(default=True, help_text="For Ice Arenas: True if operating 12 months, False if seasonal (Sept-April)")

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
class LoadProfile(models.Model):
    profile_type = models.CharField(max_length=100, choices=[
        ('heating', 'Heating'),
        ('cooling', 'Cooling'),
        ('electricity', 'Electricity'),
        ('gas', 'Gas'),
        ('waste_heat', 'Waste Heat'),
    ])
    value = models.FloatField()  # Value in kW or m³/h
    unit = models.CharField(max_length=20, choices=[
        ('kW', 'kW'),])
    start_date = models.DateTimeField() 
    end_date = models.DateTimeField()    
    building = models.ForeignKey(Building, on_delete=models.CASCADE, related_name='load_profiles', null=True, blank=True)
    boiler = models.ForeignKey('Boiler', on_delete=models.CASCADE, related_name='load_profiles', null=True, blank=True)
    network = models.ForeignKey('Network', on_delete=models.CASCADE, related_name='load_profiles', null=True, blank=True)
    source_model = models.CharField(
        max_length=100, 
        default='ontario_heating_v1',
        help_text="The name/version of the model used to generate this profile"
    )    

    class Meta:
        indexes = [
            models.Index(fields=['building', 'profile_type']),
            models.Index(fields=['boiler', 'profile_type']),
            models.Index(fields=['network', 'profile_type']),
        ]
    

class DEAReferenceData(models.Model):
    technology = models.CharField(max_length=255)
    category = models.CharField(max_length=255, blank=True, null=True)
    parameter = models.CharField(max_length=512)
    unit = models.CharField(max_length=120, blank=True, null=True)
    estimate_type = models.CharField(max_length=50, default='ctrl')
    target_year = models.IntegerField()
    value = models.FloatField()

    class Meta:
        indexes = [
            models.Index(
                fields=['technology', 'parameter', 'target_year', 'estimate_type'],
                name='api_dea_ref_lookup_idx',
            ),
        ]

    def __str__(self):
        return f"{self.technology} - {self.parameter} ({self.target_year}, {self.estimate_type})"


class CarbonTaxRate(models.Model):
    year = models.IntegerField(unique=True)
    price_cad_per_tonne = models.FloatField()

    class Meta:
        ordering = ['year']

    def __str__(self):
        return f"{self.year}: ${self.price_cad_per_tonne}/tCO2e"


class OntarioNaturalGasRate(models.Model):
    utility = models.CharField(max_length=160, default='Enbridge Gas Inc.')
    region = models.CharField(max_length=160, default='Union South Rate Zone')
    effective_date = models.DateField()
    year = models.IntegerField()
    commodity_price_cents_per_m3 = models.FloatField(blank=True, null=True)
    gas_cost_adjustment_cents_per_m3 = models.FloatField(blank=True, null=True)
    effective_price_cents_per_m3 = models.FloatField()
    cad_per_m3 = models.FloatField()
    cad_per_mwh = models.FloatField()
    cad_per_mwh_useful_heat = models.FloatField()
    kwh_per_m3 = models.FloatField(default=10.55)
    assumed_boiler_efficiency = models.FloatField(default=0.90)
    source_name = models.CharField(max_length=255, default='Ontario Energy Board')
    source_url = models.URLField(max_length=500)
    retrieved_date = models.DateField(blank=True, null=True)

    class Meta:
        ordering = ['-effective_date', 'utility', 'region']
        unique_together = [('utility', 'region', 'effective_date')]
        indexes = [
            models.Index(fields=['region', 'year', 'effective_date']),
            models.Index(fields=['utility', 'region']),
        ]

    def __str__(self):
        return f"{self.region} {self.effective_date}: {self.effective_price_cents_per_m3} c/m3"


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
class Source(Location):
    source_type = models.CharField(max_length=100, choices=[
        ('renewable', 'Renewable'),
        ('non_renewable', 'Non-Renewable'),
        ('hybrid', 'Hybrid')
    ])
    profile = models.ForeignKey(LoadProfile, on_delete=models.CASCADE, null=True, blank=True)
    capacity = models.FloatField()  # in kW
    efficiency = models.FloatField()  # as percentage
    fuel_type = models.CharField(max_length=100, blank=True, null=True)
    installation_date = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')
    def __str__(self):
        return self.name if self.name else f"Source {self.id} in {self.project.name}"  # type: ignore
class Sink(Location):
    sink_type = models.CharField(max_length=100, choices=[
        ('heat_sink', 'Heat Sink'),
        ('cooling_sink', 'Cooling Sink'),
        ('waste_sink', 'Waste Sink')
    ])
    capacity = models.FloatField()  # in kW
    efficiency = models.FloatField()  # as percentage
    installation_date = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')
    profile = models.ForeignKey(LoadProfile, on_delete=models.CASCADE, null=True, blank=True)
    def __str__(self):
        return self.name if self.name else f"Sink {self.id} in {self.project.name}"  # type: ignore
class Converter(Location):
    converter_type = models.CharField(max_length=100, choices=[
        ('heat_to_cooling', 'Heat to Cooling'),
        ('cooling_to_heat', 'Cooling to Heat'),
        ('electricity_to_heat', 'Electricity to Heat'),
        ('heat_to_electricity', 'Heat to Electricity')
    ])
    efficiency = models.FloatField()  # as percentage
    installation_date = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')
    
    def __str__(self):
        return self.name if self.name else f"Converter {self.id} in {self.project.name}" # type: ignore    
class Storage(Location):
    tank_type = models.CharField(max_length=100, choices=[
        ('hot_water', 'Hot Water'),
        ('cold_water', 'Cold Water'),
        ('thermal', 'Thermal')
    ])
    capacity = models.FloatField()  # in kWh
    insulation_type = models.CharField(max_length=100, blank=True, null=True)
    insulation_thickness = models.FloatField(blank=True, null=True)  # in mm
    installation_date = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')
    
    def __str__(self):
        return self.name if self.name else f"Storage Tank {self.id} in {self.project.name}" # type: ignore
     
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

# Network and Nodes
class Network(Location):
    class DerivationStage(models.TextChoices):
        BASE = 'base', _('Base Network')
        EDGE_GENERATION = 'edge_generation', _('Edge Generation')
        EDGE_CONNECTION = 'edge_connection', _('Edge Connection')
        ROUTE_TRACING = 'route_tracing', _('Route Tracing')
        ROUTE_INFILL = 'route_infill', _('Route Infill Expansion')
        PIPE_SIZING = 'pipe_sizing', _('Pipe Sizing')
        OTHER = 'other', _('Other')

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    network_type = models.CharField(max_length=100, choices=[
        ('district_heating', 'District Heating'),
        ('district_cooling', 'District Cooling'),
        ('combined', 'Combined Heating and Cooling')
    ])
    parent_network = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True)
    total_length = models.FloatField(blank=True, null=True)  # in meters
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    radius = models.FloatField(blank=True, null=True)  # in km
    cluster_algorithm = models.ForeignKey('ClusterAlgorithm', on_delete=models.CASCADE, null=True, blank=True, related_name='clusters')
    derivation_stage = models.CharField(max_length=50, choices=DerivationStage.choices, default=DerivationStage.BASE)
    derivation_method = models.CharField(max_length=100, blank=True, null=True)
    derivation_parameters = models.JSONField(blank=True, null=True)
    def __str__(self):
        return f"{self.name} ({self.project})"

class NetworkHeatInventory(models.Model):
    network = models.OneToOneField(Network, on_delete=models.CASCADE, related_name='heat_inventory')
    total_heating_load_kw = models.FloatField(blank=True, null=True)
    total_cooling_load_kw = models.FloatField(blank=True, null=True)
    total_waste_heat_kw = models.FloatField(blank=True, null=True)

def default_standard_pipe_catalog():
    return [
        {'dn': 20,  'di': 0.020, 'cost': 15,  'roughness': 0.000045},
        {'dn': 25,  'di': 0.025, 'cost': 20,  'roughness': 0.000045},
        {'dn': 32,  'di': 0.032, 'cost': 30,  'roughness': 0.000045},
        {'dn': 40,  'di': 0.040, 'cost': 40,  'roughness': 0.000045},
        {'dn': 50,  'di': 0.050, 'cost': 55,  'roughness': 0.000045},
        {'dn': 65,  'di': 0.065, 'cost': 70,  'roughness': 0.000045},
        {'dn': 80,  'di': 0.080, 'cost': 90,  'roughness': 0.000045},
        {'dn': 100, 'di': 0.100, 'cost': 120, 'roughness': 0.000045},
        {'dn': 125, 'di': 0.125, 'cost': 160, 'roughness': 0.000045},
        {'dn': 150, 'di': 0.150, 'cost': 200, 'roughness': 0.000045},
        {'dn': 200, 'di': 0.200, 'cost': 300, 'roughness': 0.000045},
        {'dn': 250, 'di': 0.250, 'cost': 450, 'roughness': 0.000045},
        {'dn': 300, 'di': 0.300, 'cost': 600, 'roughness': 0.000045},
        {'dn': 350, 'di': 0.350, 'cost': 800, 'roughness': 0.000045},
        {'dn': 400, 'di': 0.400, 'cost': 1100, 'roughness': 0.000045},
    ]

class NetworkDesignConstants(models.Model):
    """
    Per-network physical, hydraulic, and economic assumptions used by sizing
    and costing.
    """
    network = models.OneToOneField(Network, on_delete=models.CASCADE, related_name='design_constants')

    rho = models.FloatField(default=971.8)
    cp = models.FloatField(default=4.19)
    mu = models.FloatField(default=0.000355)

    max_velocity = models.FloatField(default=2.5)
    min_velocity = models.FloatField(default=0.5)
    target_pressure_drop = models.FloatField(default=100.0)

    electricity_cost = models.FloatField(default=0.15)
    pump_efficiency = models.FloatField(default=0.75)
    fixed_om_rate = models.FloatField(default=0.02)
    discount_rate = models.FloatField(default=0.04)
    source_cop = models.FloatField(default=1.0)
    lifespan_years = models.IntegerField(default=30)
    hours_per_year = models.IntegerField(default=4000)

    supply_temperature_c = models.FloatField(default=80.0)
    return_temperature_c = models.FloatField(default=50.0)
    ambient_temperature_c = models.FloatField(default=10.0)
    thermal_loss_coefficient_w_m_k = models.FloatField(default=0.25)

    fallback_pipe_cost_per_meter = models.FloatField(default=150.0)
    pipe_catalog = models.JSONField(default=default_standard_pipe_catalog)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Node(Location):
    network = models.ForeignKey(Network, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, blank=True, null=True)
    node_type = models.CharField(max_length=100, choices=[
        ('building', 'Building'),
        ('pv', 'PV'),
        ('storage', 'Storage'),
        ('geothermal', 'Geothermal'),
        ('cluster', 'Cluster'),
        ('consumer', 'Consumer'),
        ('producer', 'Producer'),
        ('junction', 'Junction / Tap'),
        ('boiler', 'Boiler'),
        ('network', 'Network'),
        ('other', 'Other')
    ])
    building = models.ForeignKey(Building, on_delete=models.CASCADE, null=True, blank=True)
    pv= models.ForeignKey(PVSystem, on_delete=models.CASCADE, null=True, blank=True)
    geothermal = models.ForeignKey(GeothermalSystem, on_delete=models.CASCADE, null=True, blank=True)
    boiler = models.ForeignKey(Boiler, on_delete=models.CASCADE, null=True, blank   =True)
    thermal_storage = models.ForeignKey(ThermalStorage, on_delete=models.CASCADE, null=True, blank=True)
    cluster = models.ForeignKey('cluster', on_delete=models.CASCADE, blank=True, null=True, related_name='nodes')
    network_node = models.ForeignKey(Network, on_delete=models.CASCADE, blank=True, null=True, related_name='nodes')
    dea_technology_name = models.CharField(max_length=255, null=True, blank=True)
    other_information = models.JSONField(blank=True, null=True, help_text="Any additional information about the node")
    is_default = models.BooleanField(default=False)  # Indicates if this is the default node
    #we would add clusterbuilding to hold  the buildings in a cluster later

class BuildingCluster(models.Model):
    building = models.ForeignKey(Building, on_delete=models.CASCADE)
    cluster = models.ForeignKey('Cluster', on_delete=models.CASCADE)

class EdgeAlgorithm(models.Model):
    class EdgeAlgorithmType(models.TextChoices):
        DELAUNAY = 'delaunay', _('Delaunay')
        KNN = 'knn', _('KNN')
        MUTUAL_KNN = 'mutual_knn', _('Mutual KNN')
        DISTANCE_THRESHOLD = 'distance_threshold', _('Distance Threshold')
        COMPLETE = 'complete', _('Complete Graph')
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
    source_network = models.ForeignKey(Network, on_delete=models.SET_NULL, null=True, blank=True, related_name='derived_edge_algorithms')
    name = models.CharField(max_length=100)
    distance_threshold = models.FloatField(blank=True, null=True)  # for distance threshold method
    description = models.TextField(blank=True, null=True)
    is_default = models.BooleanField(default=False)  # Indicates if this is the default algorithm
    parameters = models.JSONField(blank=True, null=True, help_text="Additional parameters for the algorithm")
    total_edge_length = models.FloatField(blank=True, null=True, help_text="Total length of all edges generated by this algorithm (in meters)")

    @classmethod
    def generation_choices(cls):
        """Algorithms that generate candidate edges for a new network design."""
        active = {
            cls.EdgeAlgorithmType.DELAUNAY,
            cls.EdgeAlgorithmType.KNN,
            cls.EdgeAlgorithmType.MUTUAL_KNN,
            cls.EdgeAlgorithmType.DISTANCE_THRESHOLD,
            cls.EdgeAlgorithmType.COMPLETE,
            cls.EdgeAlgorithmType.GABRIEL,
            cls.EdgeAlgorithmType.BETA_SKELETON,
            cls.EdgeAlgorithmType.RNG,
        }
        return [(value, label) for value, label in cls.EdgeAlgorithmType.choices if value in active]

    @classmethod
    def generation_values(cls):
        return {value for value, _ in cls.generation_choices()}

    def __str__(self):
        return self.name
class EdgeConnectionAlgorithm(models.Model):
    class EdgeConnectionAlgorithmType(models.TextChoices):
        MST = 'mst', _('Minimum Spanning Tree')
        SHORTEST_PATH_TREE = 'shortest_path_tree', _('Source-Rooted Shortest Path Tree')
        STEINER_TREE = 'steiner_tree', _('Approximate Steiner Tree')
        REDUNDANT_MST = 'redundant_mst', _('Reliability-Augmented MST')
        ALL_CANDIDATES = 'all_candidates', _('Use All Candidate Edges')
        CUSTOM = 'custom', _('Custom')
    algorithm = models.CharField(
        max_length=32,
        choices=EdgeConnectionAlgorithmType.choices,
        default=EdgeConnectionAlgorithmType.MST
    )
    network = models.ForeignKey(Network, on_delete=models.CASCADE)
    source_network = models.ForeignKey(Network, on_delete=models.SET_NULL, null=True, blank=True, related_name='derived_edge_connection_algorithms')
    edge_algorithm = models.ForeignKey(EdgeAlgorithm, on_delete=models.CASCADE, related_name='edge_connections', null=True, blank=True)
    name = models.CharField(max_length=100)
    distance_threshold = models.FloatField(blank=True, null=True)  # for distance threshold method
    description = models.TextField(blank=True, null=True)
    is_default = models.BooleanField(default=False)  # Indicates if this is the default algorithm
    parameters = models.JSONField(blank=True, null=True, help_text="Additional parameters for the algorithm")
    total_edge_length = models.FloatField(blank=True, null=True, help_text="Total length of all edges generated by this algorithm (in meters)")

    @classmethod
    def selection_choices(cls):
        active = {
            cls.EdgeConnectionAlgorithmType.MST,
            cls.EdgeConnectionAlgorithmType.SHORTEST_PATH_TREE,
            cls.EdgeConnectionAlgorithmType.STEINER_TREE,
            cls.EdgeConnectionAlgorithmType.REDUNDANT_MST,
        }
        return [(value, label) for value, label in cls.EdgeConnectionAlgorithmType.choices if value in active]

    @classmethod
    def selection_values(cls):
        return {value for value, _ in cls.selection_choices()}

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
    other_information = models.JSONField(blank=True, null=True, help_text="Any additional information about the edge")
    def __str__(self):
        return f"{self.start_node} to {self.end_node}"
class EdgeConnectionAlgorithmEdge(models.Model):
    edge_connection_algorithm = models.ForeignKey(EdgeConnectionAlgorithm, on_delete=models.CASCADE, related_name='edges_connection')
    edge = models.ForeignKey(Edge, on_delete=models.CASCADE, related_name='edge_connections_edge')
class EdgeTranslation(models.Model):
    class EdgeTranslationType(models.TextChoices):
        DIRECT = 'direct', _('Direct Geodesic Baseline')
        WALKING = 'walking', _('Walking')
        BICYCLING = 'bicycling', _('Bicycling')
        DRIVING = 'driving', _('Driving')
        TRANSIT = 'transit', _('Transit')

    @classmethod
    def routing_choices(cls):
        active = {
            cls.EdgeTranslationType.DIRECT,
            cls.EdgeTranslationType.DRIVING,
            cls.EdgeTranslationType.WALKING,
            cls.EdgeTranslationType.BICYCLING,
        }
        return [(value, label) for value, label in cls.EdgeTranslationType.choices if value in active]

    @classmethod
    def routing_values(cls):
        return {value for value, _ in cls.routing_choices()}

        
    translation = models.CharField(
        max_length=32,
        choices=EdgeTranslationType.choices,
        default=EdgeTranslationType.WALKING

    )
    network = models.ForeignKey(Network, on_delete=models.CASCADE, null=True, blank=True, related_name='edge_translations')
    source_network = models.ForeignKey(Network, on_delete=models.SET_NULL, null=True, blank=True, related_name='derived_edge_translations')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    is_default = models.BooleanField(default=False)
    parameters = models.JSONField(blank=True, null=True)
    total_edge_length = models.FloatField(blank=True, null=True)
    algorithm = models.ForeignKey(EdgeConnectionAlgorithm, on_delete=models.CASCADE, null=True)
    edge_connection_algorithm = models.ForeignKey(EdgeConnectionAlgorithm, on_delete=models.CASCADE, null=True, blank=True, related_name='edge_translations')
    def __str__(self):
        return self.name
class Route(models.Model):
    id = models.AutoField(primary_key=True)
    route_type = models.CharField(max_length=100, choices=[
        ('supply', 'Supply'),
        ('return', 'Return'),
        ('service', 'Service Connection'),
    ], default='return')
    edge = models.ForeignKey(Edge, related_name='edge', on_delete=models.CASCADE)
    translation = models.ForeignKey(EdgeTranslation, related_name='routes', on_delete=models.CASCADE, null=True, blank=True)
    sub_route = models.ForeignKey('self', related_name='routes', on_delete=models.CASCADE, null=True, blank=True)
    polyline = models.TextField()  # Encoded path
    distance_meters = models.FloatField()
    duration_seconds = models.FloatField()

class NetworkExpansionRun(models.Model):
    source_network = models.ForeignKey(Network, on_delete=models.CASCADE, related_name='expansion_runs_started')
    output_network = models.OneToOneField(Network, on_delete=models.CASCADE, related_name='expansion_run')
    source_edge_translation = models.ForeignKey(EdgeTranslation, on_delete=models.SET_NULL, null=True, blank=True, related_name='expansion_runs')
    name = models.CharField(max_length=255)
    max_distance_m = models.FloatField(default=100.0)
    parameters = models.JSONField(blank=True, null=True)
    candidate_count = models.PositiveIntegerField(default=0)
    attached_count = models.PositiveIntegerField(default=0)
    rejected_count = models.PositiveIntegerField(default=0)
    added_heat_kw = models.FloatField(default=0.0)
    added_service_length_m = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class NetworkExpansionAttachment(models.Model):
    run = models.ForeignKey(NetworkExpansionRun, on_delete=models.CASCADE, related_name='attachments')
    building = models.ForeignKey(Building, on_delete=models.CASCADE)
    new_node = models.ForeignKey(Node, on_delete=models.SET_NULL, null=True, blank=True, related_name='expansion_building_attachments')
    tap_node = models.ForeignKey(Node, on_delete=models.SET_NULL, null=True, blank=True, related_name='expansion_tap_attachments')
    source_route = models.ForeignKey(Route, on_delete=models.SET_NULL, null=True, blank=True, related_name='expansion_attachments')
    source_edge = models.ForeignKey(Edge, on_delete=models.SET_NULL, null=True, blank=True, related_name='expansion_attachments')
    service_edge = models.ForeignKey(Edge, on_delete=models.SET_NULL, null=True, blank=True, related_name='expansion_service_attachments')
    service_route = models.ForeignKey(Route, on_delete=models.SET_NULL, null=True, blank=True, related_name='expansion_service_attachments')
    status = models.CharField(max_length=32, choices=[
        ('attached', 'Attached'),
        ('rejected', 'Rejected'),
    ], default='rejected')
    reason = models.CharField(max_length=255, blank=True, null=True)
    distance_to_route_m = models.FloatField(blank=True, null=True)
    snap_latitude = models.FloatField(blank=True, null=True)
    snap_longitude = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)


class PhasingStudy(models.Model):
    class Status(models.TextChoices):
        DRAFT = 'draft', _('Draft')
        RUNNING = 'running', _('Running')
        COMPLETED = 'completed', _('Completed')
        FAILED = 'failed', _('Failed')

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='phasing_studies')
    base_network = models.ForeignKey(Network, on_delete=models.CASCADE, related_name='phasing_studies')
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.DRAFT)
    parameters = models.JSONField(blank=True, null=True)
    solver = models.CharField(max_length=100, default='deterministic_greedy')
    objective_value = models.FloatField(blank=True, null=True)
    task_id = models.CharField(max_length=255, blank=True, null=True)
    error = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['project', 'base_network']),
            models.Index(fields=['status', 'created_at']),
        ]

    def __str__(self):
        return self.name


class PlanningPeriod(models.Model):
    study = models.ForeignKey(PhasingStudy, on_delete=models.CASCADE, related_name='periods')
    sequence = models.PositiveIntegerField(default=1)
    year = models.PositiveIntegerField()
    label = models.CharField(max_length=100, blank=True, null=True)
    available_heat_kw = models.FloatField(blank=True, null=True)
    capex_budget_cad = models.FloatField(blank=True, null=True)

    class Meta:
        ordering = ['sequence', 'year']
        unique_together = [('study', 'year')]
        indexes = [
            models.Index(fields=['study', 'year']),
        ]

    def __str__(self):
        return self.label or str(self.year)


class PlanningScenario(models.Model):
    study = models.ForeignKey(PhasingStudy, on_delete=models.CASCADE, related_name='scenarios')
    name = models.CharField(max_length=120)
    probability = models.FloatField(default=1.0)
    discount_rate = models.FloatField(default=0.045)
    demand_multiplier = models.FloatField(default=1.0)
    fuel_price_multiplier = models.FloatField(default=1.0)
    uncertainty_revealed_year = models.PositiveIntegerField(blank=True, null=True)

    class Meta:
        ordering = ['id']
        indexes = [
            models.Index(fields=['study', 'name']),
        ]

    def __str__(self):
        return self.name


class CandidateConnection(models.Model):
    class CandidateType(models.TextChoices):
        BUILDING = 'building', _('Building')
        CLUSTER = 'cluster', _('Cluster')

    class CandidateRole(models.TextChoices):
        CONSUMER = 'consumer', _('Consumer')
        SOURCE = 'source', _('Heat Source')
        MIXED = 'mixed', _('Mixed')

    class Status(models.TextChoices):
        ELIGIBLE = 'eligible', _('Eligible')
        REJECTED = 'rejected', _('Rejected')
        SELECTED = 'selected', _('Selected')

    study = models.ForeignKey(PhasingStudy, on_delete=models.CASCADE, related_name='candidates')
    candidate_type = models.CharField(max_length=32, choices=CandidateType.choices, default=CandidateType.BUILDING)
    building = models.ForeignKey(Building, on_delete=models.CASCADE, blank=True, null=True, related_name='phasing_candidates')
    cluster = models.ForeignKey('Cluster', on_delete=models.CASCADE, blank=True, null=True, related_name='phasing_candidates')
    name = models.CharField(max_length=255)
    candidate_role = models.CharField(max_length=32, choices=CandidateRole.choices, default=CandidateRole.CONSUMER)
    available_from_year = models.PositiveIntegerField(blank=True, null=True)
    peak_heat_kw = models.FloatField(default=0.0)
    annual_heat_mwh = models.FloatField(default=0.0)
    available_heat_kw = models.FloatField(default=0.0)
    distance_to_trunk_m = models.FloatField(blank=True, null=True)
    connection_length_m = models.FloatField(blank=True, null=True)
    estimated_connection_cost_cad = models.FloatField(default=0.0)
    linear_heat_density_mwh_per_m_year = models.FloatField(default=0.0)
    suitability_score = models.FloatField(default=0.0)
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.ELIGIBLE)
    rejection_reason = models.CharField(max_length=255, blank=True, null=True)
    snap_latitude = models.FloatField(blank=True, null=True)
    snap_longitude = models.FloatField(blank=True, null=True)
    source_route = models.ForeignKey(Route, on_delete=models.SET_NULL, blank=True, null=True, related_name='phasing_candidates')
    source_edge = models.ForeignKey(Edge, on_delete=models.SET_NULL, blank=True, null=True, related_name='phasing_candidates')
    metadata = models.JSONField(blank=True, null=True)

    class Meta:
        ordering = ['-suitability_score', 'distance_to_trunk_m', 'id']
        indexes = [
            models.Index(fields=['study', 'candidate_type', 'status']),
            models.Index(fields=['study', 'suitability_score']),
        ]

    def __str__(self):
        return self.name


class PhasingResult(models.Model):
    study = models.ForeignKey(PhasingStudy, on_delete=models.CASCADE, related_name='results')
    scenario = models.ForeignKey(PlanningScenario, on_delete=models.CASCADE, related_name='results')
    period = models.ForeignKey(PlanningPeriod, on_delete=models.CASCADE, related_name='results')
    candidate = models.ForeignKey(CandidateConnection, on_delete=models.CASCADE, related_name='results')
    output_network = models.ForeignKey(Network, on_delete=models.SET_NULL, blank=True, null=True, related_name='phasing_results')

    # Indexed decision variables stored as rows:
    node_connected = models.BooleanField(default=False)       # N_i,t,m
    node_newly_connected = models.BooleanField(default=False) # Nbar_i,t,m
    edge_exists = models.BooleanField(default=False)          # E_i,j,t,m
    edge_newly_built = models.BooleanField(default=False)     # Ebar_i,j,t,m
    plant_exists = models.BooleanField(default=False)         # P_i,t,m
    heat_supplied_kw = models.FloatField(default=0.0)         # Q_i,t,m

    capex_cad = models.FloatField(default=0.0)
    discounted_net_benefit_cad = models.FloatField(default=0.0)
    cumulative_heat_kw = models.FloatField(default=0.0)
    cumulative_capex_cad = models.FloatField(default=0.0)
    reason = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['period__sequence', 'scenario_id', 'candidate_id']
        indexes = [
            models.Index(fields=['study', 'period', 'scenario']),
            models.Index(fields=['study', 'candidate']),
        ]

    def __str__(self):
        return f"{self.study} - {self.period} - {self.candidate}"

class Cluster(models.Model):
    network = models.ForeignKey(Network, on_delete=models.CASCADE, related_name='clusters', null=True, blank=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    type = models.CharField(max_length=100, choices=[
        ('distance_based', 'Distance Based')
    ])
    BALANCE_STATUS_CHOICES = [
        ('excess_heat', 'Excess Heat (Source Heavy)'),
        ('heat_deficit', 'Heat Deficit (Sink Heavy)'),
        ('balanced', 'Balanced'),
        ('unknown', 'Unknown (No Data)')
    ]
    energy_balance_status = models.CharField(
        max_length=50, 
        choices=BALANCE_STATUS_CHOICES, 
        default='unknown',
        help_text="Thermodynamic state based on aggregated load profiles."
    )
    # Optional: Store the actual sums for frontend display
    total_source_capacity = models.FloatField(default=0.0, help_text="Total Waste Heat/Cooling Load (kW)")
    total_sink_capacity = models.FloatField(default=0.0, help_text="Total Heating Load (kW)")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    heating_demand = models.FloatField(blank=True, null=True)  # in kW
    cooling_demand = models.FloatField(blank=True, null=True)  # in kW
    latitude = models.FloatField(blank=True, null=True)
    longitude = models.FloatField(blank=True, null=True)
    no_of_buildings = models.IntegerField(blank=True, null=True)
    total_edge_length = models.FloatField(blank=True, null=True) 
    cluster_algorithm =  models.ForeignKey('ClusterAlgorithm', on_delete=models.CASCADE, null=True, blank=True, related_name='cluster_algorithm')
    effective_radius = models.FloatField(blank=True, null=True)  # in m
    cluster_configuration =  models.ForeignKey('ClusterConfiguration', on_delete=models.CASCADE, null=True, blank=True, related_name='cluster_configuration')
    def __str__(self):
        return self.name if self.name else f"Cluster {self.id} in {self.network.name}" # type: ignore
class ClusterAlgorithm(models.Model):
    class ClusterAlgorithmType(models.TextChoices):
        DBSCAN = 'dbscan', _('DBSCAN')
        OPTICS = 'optics', _('OPTICS')
        KMEANS = 'kmeans', _('K-Means')
        HIERARCHICAL = 'hierarchical', _('Agglomerative Hierarchical')
        MST = 'mst', _('Minimum Spanning Tree')
        GMM = 'gmm', _('Gaussian Mixture Model')
        SPECTRAL = 'spectral', _('Spectral Clustering')
        RADIUS_BASED = 'radius_based', _('Radius Based')
    class ClusteringMethodology(models.TextChoices):
        GENETIC_ALGORITHM = 'genetic_algorithm', _('Genetic Algorithm')

    algorithm = models.CharField(
        max_length=32,
        choices=ClusterAlgorithmType.choices,
        default=ClusterAlgorithmType.DBSCAN,
    )
    name = models.CharField(max_length=100)
    parameters = models.JSONField(blank=True, null=True, help_text="Additional parameters for the algorithm")
    description = models.TextField(blank=True, null=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True, blank=True)
    algorithm_type = models.CharField(max_length=100, choices=ClusterAlgorithmType.choices, null=True, blank=True)
    methodology = models.CharField(max_length=100, choices=ClusterAlgorithmType.choices, null=True, blank=True)
class ClusterConfiguration(models.Model):
    class ClusterConfigurationType(models.TextChoices):
        STANDARD = 'standard', _('Standard')
        COMPLEMENTARY = 'complementary', _('Complementary to Network')

    cluster_algorithm = models.ForeignKey(ClusterAlgorithm, on_delete=models.CASCADE, related_name='configurations')
    name= models.CharField(max_length=100, null=True, blank=True)
    parameters = models.JSONField(blank=True, null=True, help_text="Configuration parameters for the clustering algorithm")
    description = models.TextField(blank=True, null=True)
    cluster_type = models.CharField(
        max_length=32,
        choices=ClusterConfigurationType.choices,
        default=ClusterConfigurationType.STANDARD,
        db_index=True,
    )
    base_network = models.ForeignKey(
        Network,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='complementary_cluster_configurations',
        help_text="Network this complementary configuration is relative to.",
    )
    source_cluster_configuration = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='derived_complementary_configurations',
        help_text="Original cluster configuration used to derive this complementary configuration.",
    )

    class Meta:
        indexes = [
            models.Index(fields=['cluster_type', 'base_network']),
            models.Index(fields=['source_cluster_configuration']),
        ]

class ClusterConfigurationNetwork(models.Model):
    cluster_configuration = models.ForeignKey(ClusterConfiguration, on_delete=models.CASCADE, related_name='networks')
    network = models.ForeignKey(Network, on_delete=models.CASCADE, related_name='cluster_configurations')
class EnergyControllerCenter(Location):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    network = models.ForeignKey(Network, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, null=True, blank=True)
    controller_type = models.CharField(max_length=100, choices=[
        ('central', 'Central'),
        ('decentralized', 'Decentralized'),
        ('hybrid', 'Hybrid')
    ])
    model = models.CharField(max_length=100)
    serial_number = models.CharField(max_length=100, unique=True)
    installation_date = models.DateField()
    last_maintenance = models.DateField(blank=True, null=True)  
    status = models.CharField(max_length=50, choices=[
        ('operational', 'Operational'),
        ('maintenance', 'Maintenance'),
        ('out_of_service', 'Out of Service')
    ], default='operational')
    latitude = models.FloatField(blank=True, null=True)
    longitude = models.FloatField(blank=True, null=True)
    def __str__(self):
        return self.name if self.name else f"Controller {self.id} in {self.network.name}"  # type: ignore
class PipeSizing(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    network = models.ForeignKey(Network, on_delete=models.CASCADE, related_name='pipe_sizings', null=True, blank=True)
    source_network = models.ForeignKey(Network, on_delete=models.SET_NULL, related_name='derived_pipe_sizings', null=True, blank=True)
    method = models.CharField(max_length=100, choices=[
        ('velocity_limited', 'Velocity-limited catalogue sizing'),
        ('pressure_gradient', 'Velocity and pressure-gradient constrained'),
        ('lifecycle_cost', 'Lifecycle-economic sizing'),
        ('peak_sizing', 'Legacy peak sizing'),
    ], default='velocity_limited')
    name = models.CharField(max_length=100, null=True, blank=True)
    description = models.TextField(blank=True, null=True)
    cost_per_meter = models.FloatField(default=0)  # Cost per meter of pipe
    source= models.CharField(max_length=100, blank=True, null=True)  # Source of costing data

class EdgePipeSizing(models.Model):
    edge = models.ForeignKey(Edge, on_delete=models.CASCADE, related_name="pipe_sizing")
    pipe_sizing = models.ForeignKey(PipeSizing, on_delete=models.CASCADE, related_name="edge_pipe_sizings")
    peak_load_kw = models.FloatField(default=0)       # kW
    mass_flow_kg_s = models.FloatField(default=0)     # kg/s
    volume_flow_m3_s = models.FloatField(default=0)   # m³/s
    diameter_m = models.FloatField(default=0)         # m
    velocity_m_s = models.FloatField(default=0)       # chosen velocity
    delta_t = models.FloatField(default=20)  # K
    pressure_drop_pa = models.FloatField(null=True, blank=True, help_text="Total pressure drop across this segment in Pascals")
    route= models.ForeignKey(Route, on_delete=models.CASCADE, null=True, blank=True)
    dea_pipe_category = models.CharField(max_length=255, default="Twin pipe, buried")
    length_m = models.FloatField(null=True, blank=True, help_text="Physical pipe segment length in meters")
    source_route_ids = models.JSONField(default=list, blank=True)
    source_edge_ids = models.JSONField(default=list, blank=True)
    overlap_count = models.PositiveIntegerField(default=1)
    is_shared_segment = models.BooleanField(default=False)
    segment_path = models.JSONField(null=True, blank=True, help_text="Physical segment path as [{lat, lng}, ...]")


    #calculate pipe length
class PipeCosting(models.Model):
    edge = models.ForeignKey(Edge, on_delete=models.CASCADE, related_name="pipe_costing")
    material_cost_per_meter = models.FloatField()  # Cost per meter of pipe material
    installation_cost_per_meter = models.FloatField()  # Cost per meter of installation
    total_cost = models.FloatField()  # Total cost for the entire length of the pipe
    created_at = models.DateTimeField(auto_now_add=True)
    # add how network is connected to the node
    edge_algorithm = models.ForeignKey(EdgeAlgorithm, on_delete=models.CASCADE, null=True, blank=True)
    #add how the network was traced
    edge_translation = models.ForeignKey(EdgeTranslation, on_delete=models.CASCADE, null=True, blank=True)
class NetworkCostEstimate(models.Model):
    network = models.OneToOneField(Network, on_delete=models.CASCADE, related_name="network_cost_estimate")
    estimated_material_cost = models.FloatField()  # Estimated total material cost for all pipes
    estimated_installation_cost = models.FloatField()  # Estimated total installation cost for all pipes
    estimated_equipment_cost = models.FloatField()  # Estimated total cost for all equipment (pumps, valves, etc.)
    estimated_pipe_cost_per_meter = models.FloatField()  # Estimated average cost per meter of pipe
    created_at = models.DateTimeField(auto_now_add=True)    

class NetworkCostResult(models.Model):
    network = models.OneToOneField(Network, on_delete=models.CASCADE, related_name="network_cost_result")
    pipe_sizing = models.OneToOneField(PipeSizing, on_delete=models.CASCADE, related_name="network_cost_result")
    summary = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
class NetworkCostingConstants(models.Model):
    network = models.OneToOneField(Network, on_delete=models.CASCADE, related_name="network_costing")
    total_material_cost = models.FloatField()  # Total material cost for all pipes
    total_installation_cost = models.FloatField()  # Total installation cost for all pipes
    total_equipment_cost = models.FloatField()  # Total cost for all equipment (pumps, valves, etc.)
    pipe_cost_per_meter = models.FloatField()  # Average cost per meter of pipe
    #add the following Easement compensation per meter
    #One-time easement fee
    one_time_easement_fee = models.FloatField()
    #permit fees
    permit_fees = models.FloatField()
    #Inspection fees
    inspection_fees = models.FloatField()
    #Legal fees
    legal_fees = models.FloatField()
    #Capex per meter in paved sections
    capex_per_meter_paved = models.FloatField()
    #Capex per meter in unpaved sections
    capex_per_meter_unpaved = models.FloatField()
    #Unit cost per crossing
    unit_cost_per_crossing = models.FloatField()
    #Pipe material cost per meter
    pipe_material_cost_per_meter = models.FloatField()
    #Installation labor cost per meter
    installation_labor_cost_per_meter = models.FloatField()
    #Restoration cost per meter
    restoration_cost_per_meter = models.FloatField()
    #Annual energy loss per meter
    annual_energy_loss_per_meter = models.FloatField()
    #Annual pumping energy per meter
    annual_pumping_energy_per_meter = models.FloatField()
    #Utility clash probability
    utility_clash_probability = models.FloatField()
   
    created_at = models.DateTimeField(auto_now_add=True)
class NetworkSimulationResult(models.Model):
    """
    Stores metadata and results for a single oemof simulation run.
    """
    network = models.ForeignKey(
        'api.Network',
        on_delete=models.CASCADE,
        related_name='simulation_results'
    )
    run_at = models.DateTimeField(auto_now_add=True)

    # Simulation parameters
    hours = models.PositiveIntegerField(help_text="Number of simulated hours")
    solver = models.CharField(max_length=50, default="cbc")
    supply_temperature = models.FloatField(help_text="Supply temperature in °C", null=True, blank=True)
    return_temperature = models.FloatField(help_text="Return temperature in °C", null=True, blank=True)
    ambient_temperature = models.FloatField(help_text="Ambient temperature in °C", null=True, blank=True)

    # Raw results from oemof (flows, buses, components, etc.)
    results_json = JSONField(null=True, blank=True)

    # Solver status
    status = models.CharField(max_length=50, default="success")
    termination_condition = models.CharField(max_length=100, null=True, blank=True)
    solver_message = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ['-run_at']

    def __str__(self):
        return f"Simulation {self.id} for {self.network.name} at {self.run_at:%Y-%m-%d %H:%M}"

# Suggested additions to models.py

class FluidProperties(models.Model):
    """
    Defines the medium (e.g., Water, Glycol).
    Reference: ASHRAE Fundamentals [General Knowledge]
    """
    name = models.CharField(max_length=50, default="Water")
    density = models.FloatField(default=960) # kg/m³ at 80°C
    specific_heat_capacity = models.FloatField(default=4180) # J/kg·K

class DesignCriteria(models.Model):
    """
    Engineering constraints for the sizing algorithm.
    """
    pipe_sizing = models.OneToOneField('PipeSizing', on_delete=models.CASCADE)
    max_velocity = models.FloatField(default=2.0) # m/s (Erosion limit)
    min_velocity = models.FloatField(default=0.5) # m/s (Air entrainment limit)
    max_friction_loss = models.FloatField(default=150) # Pa/m (Pump head limit)
    design_delta_t = models.FloatField(default=20.0) # K (Supply - Return)


def _project_id_for_instance(obj):
    if obj is None:
        return None

    direct_project_id = getattr(obj, "project_id", None)
    if direct_project_id is not None:
        return direct_project_id

    if isinstance(obj, Project):
        return obj.pk
    if isinstance(obj, NetworkHeatInventory):
        return _project_id_for_instance(obj.network)
    if isinstance(obj, NetworkDesignConstants):
        return _project_id_for_instance(obj.network)
    if isinstance(obj, Node):
        return _project_id_for_instance(obj.network)
    if isinstance(obj, BuildingCluster):
        return _project_id_for_instance(obj.cluster)
    if isinstance(obj, EdgeAlgorithm):
        return _project_id_for_instance(obj.network)
    if isinstance(obj, EdgeConnectionAlgorithm):
        return _project_id_for_instance(obj.network)
    if isinstance(obj, Edge):
        return _project_id_for_instance(obj.network)
    if isinstance(obj, EdgeConnectionAlgorithmEdge):
        return _project_id_for_instance(obj.edge_connection_algorithm)
    if isinstance(obj, EdgeTranslation):
        return (
            _project_id_for_instance(obj.network) or
            _project_id_for_instance(obj.edge_connection_algorithm) or
            _project_id_for_instance(obj.algorithm) or
            _project_id_for_instance(obj.source_network)
        )
    if isinstance(obj, Route):
        return _project_id_for_instance(obj.edge)
    if isinstance(obj, Cluster):
        return (
            _project_id_for_instance(obj.network) or
            _project_id_for_instance(obj.cluster_configuration) or
            _project_id_for_instance(obj.cluster_algorithm)
        )
    if isinstance(obj, ClusterConfiguration):
        return (
            _project_id_for_instance(obj.cluster_algorithm) or
            _project_id_for_instance(obj.base_network) or
            _project_id_for_instance(obj.source_cluster_configuration)
        )
    if isinstance(obj, ClusterConfigurationNetwork):
        return _project_id_for_instance(obj.network)
    if isinstance(obj, EnergyControllerCenter):
        return obj.project_id
    if isinstance(obj, PipeSizing):
        return _project_id_for_instance(obj.network) or _project_id_for_instance(obj.source_network)
    if isinstance(obj, EdgePipeSizing):
        return _project_id_for_instance(obj.pipe_sizing) or _project_id_for_instance(obj.edge)
    if isinstance(obj, PipeCosting):
        return _project_id_for_instance(obj.edge)
    if isinstance(obj, NetworkCostEstimate):
        return _project_id_for_instance(obj.network)
    if isinstance(obj, NetworkCostingConstants):
        return _project_id_for_instance(obj.network)
    if isinstance(obj, NetworkSimulationResult):
        return _project_id_for_instance(obj.network)
    if isinstance(obj, DesignCriteria):
        return _project_id_for_instance(obj.pipe_sizing)
    if isinstance(obj, NetworkExpansionRun):
        return _project_id_for_instance(obj.output_network) or _project_id_for_instance(obj.source_network)
    if isinstance(obj, NetworkExpansionAttachment):
        return _project_id_for_instance(obj.run)
    if isinstance(obj, PhasingStudy):
        return obj.project_id or _project_id_for_instance(obj.base_network)
    if isinstance(obj, PlanningPeriod):
        return _project_id_for_instance(obj.study)
    if isinstance(obj, PlanningScenario):
        return _project_id_for_instance(obj.study)
    if isinstance(obj, CandidateConnection):
        return _project_id_for_instance(obj.study)
    if isinstance(obj, PhasingResult):
        return _project_id_for_instance(obj.study)
    if isinstance(obj, LoadProfile):
        return (
            _project_id_for_instance(obj.building) or
            _project_id_for_instance(obj.boiler) or
            _project_id_for_instance(obj.network)
        )

    return None


def _network_id_for_instance(obj):
    if obj is None:
        return None

    direct_network_id = getattr(obj, "network_id", None)
    if direct_network_id is not None:
        return direct_network_id

    if isinstance(obj, Network):
        return obj.pk
    if isinstance(obj, Edge):
        return obj.network_id
    if isinstance(obj, Route):
        return _network_id_for_instance(obj.edge)
    if isinstance(obj, EdgePipeSizing):
        return _network_id_for_instance(obj.pipe_sizing) or _network_id_for_instance(obj.edge)
    if isinstance(obj, DesignCriteria):
        return _network_id_for_instance(obj.pipe_sizing)
    if isinstance(obj, NetworkExpansionRun):
        return _network_id_for_instance(obj.output_network)
    if isinstance(obj, NetworkExpansionAttachment):
        return _network_id_for_instance(obj.run)
    if isinstance(obj, PhasingStudy):
        return obj.base_network_id
    if isinstance(obj, PlanningPeriod):
        return _network_id_for_instance(obj.study)
    if isinstance(obj, PlanningScenario):
        return _network_id_for_instance(obj.study)
    if isinstance(obj, CandidateConnection):
        return _network_id_for_instance(obj.study)
    if isinstance(obj, PhasingResult):
        return _network_id_for_instance(obj.output_network) or _network_id_for_instance(obj.study)
    if isinstance(obj, ClusterConfiguration):
        return obj.base_network_id

    return None


def _same_value(left, right):
    return left is not None and right is not None and str(left) == str(right)


def _raise_scope_error(field, message):
    raise ValidationError({field: message})


def _ensure_same_project(owner, field, related, expected_project_id):
    if related is None or expected_project_id is None:
        return
    related_project_id = _project_id_for_instance(related)
    if related_project_id is not None and not _same_value(related_project_id, expected_project_id):
        _raise_scope_error(
            field,
            f"{field} belongs to project {related_project_id}, but this record belongs to project {expected_project_id}.",
        )


def _ensure_same_network(owner, field, related, expected_network_id):
    if related is None or expected_network_id is None:
        return
    related_network_id = _network_id_for_instance(related)
    if related_network_id is not None and not _same_value(related_network_id, expected_network_id):
        _raise_scope_error(
            field,
            f"{field} belongs to network {related_network_id}, but this record belongs to network {expected_network_id}.",
        )


def validate_project_isolation(instance):
    if isinstance(instance, Network):
        _ensure_same_project(instance, "parent_network", instance.parent_network, instance.project_id)
        _ensure_same_project(instance, "cluster_algorithm", instance.cluster_algorithm, instance.project_id)
        return

    if isinstance(instance, LoadProfile):
        expected_project_id = _project_id_for_instance(instance)
        _ensure_same_project(instance, "building", instance.building, expected_project_id)
        _ensure_same_project(instance, "boiler", instance.boiler, expected_project_id)
        _ensure_same_project(instance, "network", instance.network, expected_project_id)
        return

    if isinstance(instance, Pipe):
        _ensure_same_project(instance, "junction", instance.junction, instance.project_id)
        return

    if isinstance(instance, Node):
        project_id = _project_id_for_instance(instance.network)
        _ensure_same_project(instance, "building", instance.building, project_id)
        _ensure_same_project(instance, "pv", instance.pv, project_id)
        _ensure_same_project(instance, "geothermal", instance.geothermal, project_id)
        _ensure_same_project(instance, "boiler", instance.boiler, project_id)
        _ensure_same_project(instance, "thermal_storage", instance.thermal_storage, project_id)
        _ensure_same_project(instance, "cluster", instance.cluster, project_id)
        _ensure_same_project(instance, "network_node", instance.network_node, project_id)
        return

    if isinstance(instance, BuildingCluster):
        _ensure_same_project(instance, "building", instance.building, _project_id_for_instance(instance.cluster))
        return

    if isinstance(instance, EdgeAlgorithm):
        _ensure_same_project(instance, "source_network", instance.source_network, _project_id_for_instance(instance.network))
        return

    if isinstance(instance, EdgeConnectionAlgorithm):
        project_id = _project_id_for_instance(instance.network)
        _ensure_same_project(instance, "source_network", instance.source_network, project_id)
        _ensure_same_project(instance, "edge_algorithm", instance.edge_algorithm, project_id)
        return

    if isinstance(instance, Edge):
        project_id = _project_id_for_instance(instance.network)
        _ensure_same_project(instance, "start_node", instance.start_node, project_id)
        _ensure_same_project(instance, "end_node", instance.end_node, project_id)
        _ensure_same_project(instance, "edge_algorithm", instance.edge_algorithm, project_id)
        _ensure_same_network(instance, "start_node", instance.start_node, instance.network_id)
        _ensure_same_network(instance, "end_node", instance.end_node, instance.network_id)
        if instance.edge_algorithm_id and instance.edge_algorithm.network_id != instance.network_id:
            _raise_scope_error("edge_algorithm", "edge_algorithm must belong to the same network as the edge.")
        return

    if isinstance(instance, EdgeConnectionAlgorithmEdge):
        _ensure_same_project(instance, "edge", instance.edge, _project_id_for_instance(instance.edge_connection_algorithm))
        _ensure_same_network(instance, "edge", instance.edge, instance.edge_connection_algorithm.network_id)
        return

    if isinstance(instance, EdgeTranslation):
        project_id = _project_id_for_instance(instance)
        _ensure_same_project(instance, "network", instance.network, project_id)
        _ensure_same_project(instance, "source_network", instance.source_network, project_id)
        _ensure_same_project(instance, "algorithm", instance.algorithm, project_id)
        _ensure_same_project(instance, "edge_connection_algorithm", instance.edge_connection_algorithm, project_id)
        if instance.network_id:
            _ensure_same_network(instance, "algorithm", instance.algorithm, instance.network_id)
            _ensure_same_network(instance, "edge_connection_algorithm", instance.edge_connection_algorithm, instance.network_id)
        return

    if isinstance(instance, Route):
        project_id = _project_id_for_instance(instance.edge)
        _ensure_same_project(instance, "translation", instance.translation, project_id)
        _ensure_same_project(instance, "sub_route", instance.sub_route, project_id)
        if instance.translation_id and instance.translation.network_id:
            _ensure_same_network(instance, "translation", instance.translation, instance.edge.network_id)
        _ensure_same_network(instance, "sub_route", instance.sub_route, instance.edge.network_id)
        return

    if isinstance(instance, Cluster):
        project_id = _project_id_for_instance(instance)
        _ensure_same_project(instance, "network", instance.network, project_id)
        _ensure_same_project(instance, "cluster_algorithm", instance.cluster_algorithm, project_id)
        _ensure_same_project(instance, "cluster_configuration", instance.cluster_configuration, project_id)
        return

    if isinstance(instance, ClusterConfiguration):
        project_id = _project_id_for_instance(instance.cluster_algorithm)
        _ensure_same_project(instance, "base_network", instance.base_network, project_id)
        _ensure_same_project(instance, "source_cluster_configuration", instance.source_cluster_configuration, project_id)
        if instance.cluster_type == ClusterConfiguration.ClusterConfigurationType.COMPLEMENTARY:
            if not instance.base_network_id:
                _raise_scope_error("base_network", "Complementary cluster configurations must reference a base network.")
            if instance.source_cluster_configuration_id == instance.pk and instance.pk is not None:
                _raise_scope_error("source_cluster_configuration", "A complementary configuration cannot derive from itself.")
        return

    if isinstance(instance, ClusterConfigurationNetwork):
        _ensure_same_project(instance, "cluster_configuration", instance.cluster_configuration, _project_id_for_instance(instance.network))
        return

    if isinstance(instance, EnergyControllerCenter):
        _ensure_same_project(instance, "network", instance.network, instance.project_id)
        return

    if isinstance(instance, PipeSizing):
        _ensure_same_project(instance, "source_network", instance.source_network, _project_id_for_instance(instance.network))
        return

    if isinstance(instance, EdgePipeSizing):
        project_id = _project_id_for_instance(instance.pipe_sizing)
        _ensure_same_project(instance, "edge", instance.edge, project_id)
        _ensure_same_project(instance, "route", instance.route, project_id)
        _ensure_same_network(instance, "edge", instance.edge, instance.pipe_sizing.network_id)
        if instance.route_id:
            _ensure_same_network(instance, "route", instance.route, instance.edge.network_id)
        return

    if isinstance(instance, PipeCosting):
        project_id = _project_id_for_instance(instance.edge)
        _ensure_same_project(instance, "edge_algorithm", instance.edge_algorithm, project_id)
        _ensure_same_project(instance, "edge_translation", instance.edge_translation, project_id)
        return

    if isinstance(instance, NetworkExpansionRun):
        project_id = _project_id_for_instance(instance.output_network)
        _ensure_same_project(instance, "source_network", instance.source_network, project_id)
        _ensure_same_project(instance, "source_edge_translation", instance.source_edge_translation, project_id)
        return

    if isinstance(instance, NetworkExpansionAttachment):
        project_id = _project_id_for_instance(instance.run)
        _ensure_same_project(instance, "building", instance.building, project_id)
        _ensure_same_project(instance, "new_node", instance.new_node, project_id)
        _ensure_same_project(instance, "tap_node", instance.tap_node, project_id)
        _ensure_same_project(instance, "source_route", instance.source_route, project_id)
        _ensure_same_project(instance, "source_edge", instance.source_edge, project_id)
        _ensure_same_project(instance, "service_edge", instance.service_edge, project_id)
        _ensure_same_project(instance, "service_route", instance.service_route, project_id)
        return

    if isinstance(instance, PhasingStudy):
        _ensure_same_project(instance, "base_network", instance.base_network, instance.project_id)
        return

    if isinstance(instance, PlanningPeriod):
        _ensure_same_project(instance, "study", instance.study, _project_id_for_instance(instance.study))
        return

    if isinstance(instance, PlanningScenario):
        _ensure_same_project(instance, "study", instance.study, _project_id_for_instance(instance.study))
        return

    if isinstance(instance, CandidateConnection):
        project_id = _project_id_for_instance(instance.study)
        _ensure_same_project(instance, "building", instance.building, project_id)
        _ensure_same_project(instance, "cluster", instance.cluster, project_id)
        _ensure_same_project(instance, "source_route", instance.source_route, project_id)
        _ensure_same_project(instance, "source_edge", instance.source_edge, project_id)
        if instance.source_route_id:
            _ensure_same_network(instance, "source_route", instance.source_route, instance.study.base_network_id)
        if instance.source_edge_id:
            _ensure_same_network(instance, "source_edge", instance.source_edge, instance.study.base_network_id)
        return

    if isinstance(instance, PhasingResult):
        project_id = _project_id_for_instance(instance.study)
        _ensure_same_project(instance, "scenario", instance.scenario, project_id)
        _ensure_same_project(instance, "period", instance.period, project_id)
        _ensure_same_project(instance, "candidate", instance.candidate, project_id)
        _ensure_same_project(instance, "output_network", instance.output_network, project_id)
        return


def assert_network_project_integrity(network):
    """
    Validate that a network and its direct calculation assets stay inside one
    project before a derivation, sizing, or costing calculation trusts them.
    """
    network = Network.objects.select_related("project", "parent_network", "cluster_algorithm").get(pk=network.pk)
    validate_project_isolation(network)

    for node in Node.objects.filter(network=network).select_related(
        "network",
        "building",
        "pv",
        "geothermal",
        "boiler",
        "thermal_storage",
        "cluster",
        "network_node",
    ):
        validate_project_isolation(node)

    cluster_ids = []
    for cluster in Cluster.objects.filter(network=network).select_related(
        "network",
        "cluster_algorithm",
        "cluster_configuration",
        "cluster_configuration__cluster_algorithm",
    ):
        validate_project_isolation(cluster)
        cluster_ids.append(cluster.id)

    for link in BuildingCluster.objects.filter(cluster_id__in=cluster_ids).select_related(
        "building",
        "cluster",
        "cluster__network",
        "cluster__cluster_configuration",
        "cluster__cluster_configuration__cluster_algorithm",
    ):
        validate_project_isolation(link)

    for algorithm in EdgeAlgorithm.objects.filter(network=network).select_related("network", "source_network"):
        validate_project_isolation(algorithm)

    for connection in EdgeConnectionAlgorithm.objects.filter(network=network).select_related(
        "network",
        "source_network",
        "edge_algorithm",
        "edge_algorithm__network",
    ):
        validate_project_isolation(connection)

    for edge in Edge.objects.filter(network=network).select_related(
        "network",
        "edge_algorithm",
        "edge_algorithm__network",
        "start_node",
        "start_node__network",
        "end_node",
        "end_node__network",
    ):
        validate_project_isolation(edge)

    for link in EdgeConnectionAlgorithmEdge.objects.filter(
        edge_connection_algorithm__network=network
    ).select_related(
        "edge_connection_algorithm",
        "edge_connection_algorithm__network",
        "edge",
        "edge__network",
    ):
        validate_project_isolation(link)

    for translation in EdgeTranslation.objects.filter(network=network).select_related(
        "network",
        "source_network",
        "algorithm",
        "algorithm__network",
        "edge_connection_algorithm",
        "edge_connection_algorithm__network",
    ):
        validate_project_isolation(translation)

    for route in Route.objects.filter(edge__network=network).select_related(
        "edge",
        "edge__network",
        "translation",
        "translation__network",
        "sub_route",
        "sub_route__edge",
        "sub_route__edge__network",
    ):
        validate_project_isolation(route)

    for sizing in PipeSizing.objects.filter(network=network).select_related("network", "source_network"):
        validate_project_isolation(sizing)

    for sizing_row in EdgePipeSizing.objects.filter(pipe_sizing__network=network).select_related(
        "edge",
        "edge__network",
        "pipe_sizing",
        "pipe_sizing__network",
        "route",
        "route__edge",
        "route__edge__network",
    ):
        validate_project_isolation(sizing_row)


PROJECT_ISOLATED_MODELS = {
    "Network",
    "LoadProfile",
    "Pipe",
    "Node",
    "BuildingCluster",
    "EdgeAlgorithm",
    "EdgeConnectionAlgorithm",
    "Edge",
    "EdgeConnectionAlgorithmEdge",
    "EdgeTranslation",
    "Route",
    "Cluster",
    "ClusterConfiguration",
    "ClusterConfigurationNetwork",
    "EnergyControllerCenter",
    "PipeSizing",
    "EdgePipeSizing",
    "PipeCosting",
    "PhasingStudy",
    "PlanningPeriod",
    "PlanningScenario",
    "CandidateConnection",
    "PhasingResult",
}


@receiver(pre_save)
def enforce_project_isolation(sender, instance, **kwargs):
    if sender.__name__ in PROJECT_ISOLATED_MODELS:
        validate_project_isolation(instance)


class ImportedHeatingShape(models.Model):
    """Shared hourly shape; energies stay per building, avoiding duplicate series."""
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='imported_heating_shapes')
    year = models.PositiveIntegerField()
    timestamps = models.JSONField()
    hourly_weights = models.JSONField()
    provenance = models.JSONField(default=dict)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['project', 'year'], name='unique_imported_shape_year')]


class ImportedBuildingHeating(models.Model):
    building = models.OneToOneField(Building, on_delete=models.CASCADE, related_name='imported_heating')
    source_building_id = models.CharField(max_length=40)
    annual_energy_kwh = models.JSONField()
    annual_peak_kw = models.JSONField()
    source_model = models.CharField(max_length=100, default='burlington_hdd18_import_v1')


class BurlingtonComparisonProfile(models.Model):
    """One imported alternative, kept separate from active operational profiles."""
    building = models.ForeignKey(Building, on_delete=models.CASCADE, related_name='comparison_profiles')
    model_id = models.CharField(max_length=80)
    year = models.PositiveIntegerField()
    timestamps = models.JSONField()
    hourly_kw = models.JSONField()
    provenance = models.JSONField(default=dict)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['building', 'model_id', 'year'], name='unique_burlington_comparison')]
