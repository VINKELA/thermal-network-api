from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [('api', '0085_pipe_sizing_methods')]

    operations = [
        migrations.CreateModel(
            name='NetworkCostResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('summary', models.JSONField(default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('network', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='network_cost_result', to='api.network')),
                ('pipe_sizing', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='network_cost_result', to='api.pipesizing')),
            ],
        ),
    ]
