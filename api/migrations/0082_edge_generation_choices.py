from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("api", "0081_burlington_comparison_profiles")]

    operations = [
        migrations.AlterField(
            model_name="edgealgorithm",
            name="algorithm",
            field=models.CharField(
                choices=[
                    ("delaunay", "Delaunay"),
                    ("knn", "KNN"),
                    ("mutual_knn", "Mutual KNN"),
                    ("distance_threshold", "Distance Threshold"),
                    ("complete", "Complete Graph"),
                    ("mst", "Minimum Spanning Tree"),
                    ("gabriel", "Gabriel Graph"),
                    ("beta_skeleton", "Beta Skeleton"),
                    ("rng", "Relative Neighborhood Graph"),
                    ("tsp", "Traveling Salesman Problem"),
                    ("custom", "Custom"),
                ],
                default="delaunay",
                max_length=32,
            ),
        ),
    ]
