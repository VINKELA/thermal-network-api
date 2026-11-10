# Generated manually for route tracing network lineage.

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0065_network_lineage_and_operation_sources'),
    ]

    operations = [
        migrations.AlterField(
            model_name='network',
            name='derivation_stage',
            field=models.CharField(
                choices=[
                    ('base', 'Base Network'),
                    ('edge_generation', 'Edge Generation'),
                    ('edge_connection', 'Edge Connection'),
                    ('route_tracing', 'Route Tracing'),
                    ('pipe_sizing', 'Pipe Sizing'),
                    ('other', 'Other'),
                ],
                default='base',
                max_length=50,
            ),
        ),
        migrations.AddField(
            model_name='edgetranslation',
            name='network',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name='edge_translations',
                to='api.network',
            ),
        ),
        migrations.AddField(
            model_name='edgetranslation',
            name='source_network',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='derived_edge_translations',
                to='api.network',
            ),
        ),
    ]
