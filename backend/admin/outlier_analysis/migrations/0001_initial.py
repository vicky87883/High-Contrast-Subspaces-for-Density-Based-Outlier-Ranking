# Generated by Django 5.0.7 on 2025-03-18 15:06

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ThyroidCancerData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('age', models.IntegerField(blank=True, null=True)),
                ('sex', models.CharField(blank=True, max_length=10, null=True)),
                ('on_thyroxine', models.BooleanField(default=False)),
                ('query_on_thyroxine', models.BooleanField(default=False)),
                ('on_antithyroid_medication', models.BooleanField(default=False)),
                ('sick', models.BooleanField(default=False)),
                ('pregnant', models.BooleanField(default=False)),
                ('thyroid_surgery', models.BooleanField(default=False)),
                ('I131_treatment', models.BooleanField(default=False)),
                ('query_hypothyroid', models.BooleanField(default=False)),
                ('query_hyperthyroid', models.BooleanField(default=False)),
                ('lithium', models.BooleanField(default=False)),
                ('goitre', models.BooleanField(default=False)),
                ('tumor', models.BooleanField(default=False)),
                ('hypopituitary', models.BooleanField(default=False)),
                ('psych', models.BooleanField(default=False)),
                ('TSH', models.FloatField(blank=True, null=True)),
                ('T3_measured', models.BooleanField(default=False)),
                ('TT4_measured', models.BooleanField(default=False)),
                ('T4U_measured', models.BooleanField(default=False)),
                ('FTI_measured', models.BooleanField(default=False)),
                ('outlier_label', models.BooleanField(default=False)),
            ],
            options={
                'db_table': 'mytable',
            },
        ),
    ]
