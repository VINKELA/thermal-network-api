from rest_framework import serializers
from .models import *

class CountrySerializer(serializers.ModelSerializer):
    class Meta:
        model = Country
        fields = '__all__'

class ProvinceSerializer(serializers.ModelSerializer):
    country = CountrySerializer()
    
    class Meta:
        model = Province
        fields = '__all__'

class CitySerializer(serializers.ModelSerializer):
    province = ProvinceSerializer(required=False)
    country = CountrySerializer(required=False)
    
    class Meta:
        model = City
        fields = '__all__'

class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = '__all__'

class NetworkSerializer(serializers.ModelSerializer):
    class Meta:
        model = Network
        fields = '__all__'

class LocationSerializer(serializers.ModelSerializer):    
    class Meta:
        model = Location
        fields = '__all__'
        abstract = True

class BuildingSerializer(LocationSerializer):
    class Meta:
        model = Building
        fields = '__all__'

class NodeSerializer(LocationSerializer):
    network = serializers.PrimaryKeyRelatedField(queryset=Network.objects.all())
    
    class Meta:
        model = Node
        fields = '__all__'

class EdgeSerializer(serializers.ModelSerializer):
    network = serializers.PrimaryKeyRelatedField(queryset=Network.objects.all())
    start_node = serializers.PrimaryKeyRelatedField(queryset=Node.objects.all())
    end_node = serializers.PrimaryKeyRelatedField(queryset=Node.objects.all())
    
    class Meta:
        model = Edge
        fields = '__all__'

class PumpSerializer(LocationSerializer):
    class Meta:
        model = Pump
        fields = '__all__'

class ValveSerializer(LocationSerializer):
    class Meta:
        model = Valve
        fields = '__all__'

class HeatPumpSerializer(LocationSerializer):
    class Meta:
        model = HeatPump
        fields = '__all__'

class HeatExchangerSerializer(LocationSerializer):
    class Meta:
        model = HeatExchanger
        fields = '__all__'

class PipeSerializer(LocationSerializer):
    class Meta:
        model = Pipe
        fields = '__all__'

class JunctionSerializer(LocationSerializer):
    class Meta:
        model = Junction
        fields = '__all__'

class PVSystemSerializer(LocationSerializer):
    class Meta:
        model = PVSystem
        fields = '__all__'

class GeothermalSystemSerializer(LocationSerializer):
    class Meta:
        model = GeothermalSystem
        fields = '__all__'

class BoilerSerializer(LocationSerializer):
    class Meta:
        model = Boiler
        fields = '__all__'

class ThermalStorageSerializer(LocationSerializer):
    class Meta:
        model = ThermalStorage
        fields = '__all__'

class SensorSerializer(LocationSerializer):
    class Meta:
        model = Sensor
        fields = '__all__'

class MeterSerializer(LocationSerializer):
    class Meta:
        model = Meter
        fields = '__all__'