from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [('api', '0084_edge_translation_direct')]

    operations = [
        migrations.AlterField(
            model_name='pipesizing',
            name='method',
            field=models.CharField(
                choices=[
                    ('velocity_limited', 'Velocity-limited catalogue sizing'),
                    ('pressure_gradient', 'Velocity and pressure-gradient constrained'),
                    ('lifecycle_cost', 'Lifecycle-economic sizing'),
                    ('peak_sizing', 'Legacy peak sizing'),
                ],
                default='velocity_limited',
                max_length=100,
            ),
        ),
    ]
