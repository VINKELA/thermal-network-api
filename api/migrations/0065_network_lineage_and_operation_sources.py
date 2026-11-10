# Generated manually for network version lineage.

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0064_edgepipesizing_pressure_drop_pa'),
    ]

    operations = [
        migrations.AddField(
            model_name='network',
            name='derivation_stage',
            field=models.CharField(
                choices=[
                    ('base', 'Base Network'),
                    ('edge_generation', 'Edge Generation'),
                    ('edge_connection', 'Edge Connection'),
                    ('pipe_sizing', 'Pipe Sizing'),
                    ('other', 'Other'),
                ],
                default='base',
                max_length=50,
            ),
        ),
        migrations.AddField(
            model_name='network',
            name='derivation_method',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='network',
            name='derivation_parameters',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='edgealgorithm',
            name='source_network',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='derived_edge_algorithms',
                to='api.network',
            ),
        ),
        migrations.AddField(
            model_name='edgeconnectionalgorithm',
            name='source_network',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='derived_edge_connection_algorithms',
                to='api.network',
            ),
        ),
        migrations.AddField(
            model_name='pipesizing',
            name='network',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name='pipe_sizings',
                to='api.network',
            ),
        ),
        migrations.AddField(
            model_name='pipesizing',
            name='source_network',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='derived_pipe_sizings',
                to='api.network',
            ),
        ),
    ]
