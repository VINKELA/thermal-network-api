# Generated by Django 5.2.3 on 2025-07-08 10:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0008_edgealgorithm_parameters'),
    ]

    operations = [
        migrations.AddField(
            model_name='edgealgorithm',
            name='total_edge_length',
            field=models.FloatField(blank=True, help_text='Total length of all edges generated by this algorithm (in meters)', null=True),
        ),
    ]
