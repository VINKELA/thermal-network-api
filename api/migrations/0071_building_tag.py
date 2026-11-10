from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0070_edgepipesizing_physical_segment_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='building',
            name='tag',
            field=models.CharField(
                blank=True,
                db_index=True,
                help_text='Optional user-defined grouping tag for bulk network selection.',
                max_length=120,
                null=True,
            ),
        ),
    ]
