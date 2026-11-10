from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0071_building_tag'),
    ]

    operations = [
        migrations.AlterField(
            model_name='network',
            name='derivation_stage',
            field=models.CharField(choices=[('base', 'Base Network'), ('edge_generation', 'Edge Generation'), ('edge_connection', 'Edge Connection'), ('route_tracing', 'Route Tracing'), ('route_infill', 'Route Infill Expansion'), ('pipe_sizing', 'Pipe Sizing'), ('other', 'Other')], default='base', max_length=50),
        ),
        migrations.AlterField(
            model_name='node',
            name='node_type',
            field=models.CharField(choices=[('building', 'Building'), ('pv', 'PV'), ('storage', 'Storage'), ('geothermal', 'Geothermal'), ('cluster', 'Cluster'), ('consumer', 'Consumer'), ('producer', 'Producer'), ('junction', 'Junction / Tap'), ('boiler', 'Boiler'), ('network', 'Network'), ('other', 'Other')], max_length=100),
        ),
        migrations.AlterField(
            model_name='route',
            name='route_type',
            field=models.CharField(choices=[('supply', 'Supply'), ('return', 'Return'), ('service', 'Service Connection')], default='return', max_length=100),
        ),
        migrations.CreateModel(
            name='NetworkExpansionRun',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('max_distance_m', models.FloatField(default=100.0)),
                ('parameters', models.JSONField(blank=True, null=True)),
                ('candidate_count', models.PositiveIntegerField(default=0)),
                ('attached_count', models.PositiveIntegerField(default=0)),
                ('rejected_count', models.PositiveIntegerField(default=0)),
                ('added_heat_kw', models.FloatField(default=0.0)),
                ('added_service_length_m', models.FloatField(default=0.0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('output_network', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='expansion_run', to='api.network')),
                ('source_edge_translation', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='expansion_runs', to='api.edgetranslation')),
                ('source_network', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='expansion_runs_started', to='api.network')),
            ],
        ),
        migrations.CreateModel(
            name='NetworkExpansionAttachment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('status', models.CharField(choices=[('attached', 'Attached'), ('rejected', 'Rejected')], default='rejected', max_length=32)),
                ('reason', models.CharField(blank=True, max_length=255, null=True)),
                ('distance_to_route_m', models.FloatField(blank=True, null=True)),
                ('snap_latitude', models.FloatField(blank=True, null=True)),
                ('snap_longitude', models.FloatField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('building', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.building')),
                ('new_node', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='expansion_building_attachments', to='api.node')),
                ('run', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='attachments', to='api.networkexpansionrun')),
                ('service_edge', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='expansion_service_attachments', to='api.edge')),
                ('service_route', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='expansion_service_attachments', to='api.route')),
                ('source_edge', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='expansion_attachments', to='api.edge')),
                ('source_route', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='expansion_attachments', to='api.route')),
                ('tap_node', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='expansion_tap_attachments', to='api.node')),
            ],
        ),
    ]
