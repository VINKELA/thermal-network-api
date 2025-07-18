# Generated by Django 5.2.3 on 2025-07-08 09:39

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0006_route'),
    ]

    operations = [
        migrations.AddField(
            model_name='edge',
            name='is_default',
            field=models.BooleanField(default=False),
        ),
        migrations.CreateModel(
            name='EdgeAlgorithm',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('algorithm', models.CharField(choices=[('delaunay', 'Delaunay'), ('knn', 'KNN'), ('distance_threshold', 'Distance Threshold'), ('mst', 'Minimum Spanning Tree'), ('custom', 'Custom')], default='delaunay', max_length=32)),
                ('name', models.CharField(max_length=100)),
                ('distance_threshold', models.FloatField(blank=True, null=True)),
                ('description', models.TextField(blank=True, null=True)),
                ('is_default', models.BooleanField(default=False)),
                ('network', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.network')),
            ],
        ),
        migrations.AddField(
            model_name='edge',
            name='edge_algorithm',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='edges', to='api.edgealgorithm'),
        ),
    ]
