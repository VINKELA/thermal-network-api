# Generated by Django 5.2.3 on 2025-06-20 14:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_building_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='pump',
            name='name',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
