from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("api", "0083_edge_connection_choices")]

    operations = [
        migrations.AlterField(
            model_name="edgetranslation",
            name="translation",
            field=models.CharField(
                choices=[
                    ("direct", "Direct Geodesic Baseline"),
                    ("walking", "Walking"),
                    ("bicycling", "Bicycling"),
                    ("driving", "Driving"),
                    ("transit", "Transit"),
                ],
                default="walking",
                max_length=32,
            ),
        ),
    ]
