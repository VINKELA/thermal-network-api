from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("api", "0082_edge_generation_choices")]

    operations = [
        migrations.AlterField(
            model_name="edgeconnectionalgorithm",
            name="algorithm",
            field=models.CharField(
                choices=[
                    ("mst", "Minimum Spanning Tree"),
                    ("shortest_path_tree", "Source-Rooted Shortest Path Tree"),
                    ("steiner_tree", "Approximate Steiner Tree"),
                    ("redundant_mst", "Reliability-Augmented MST"),
                    ("all_candidates", "Use All Candidate Edges"),
                    ("custom", "Custom"),
                ],
                default="mst",
                max_length=32,
            ),
        ),
    ]
