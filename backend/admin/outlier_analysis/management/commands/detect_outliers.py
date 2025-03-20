from django.core.management.base import BaseCommand
from outlier_analysis.ml.outlier_detection import detect_outliers

class Command(BaseCommand):
    help = "Run outlier detection on thyroid cancer data"

    def handle(self, *args, **kwargs):
        detect_outliers()
        self.stdout.write(self.style.SUCCESS("Outlier detection completed!"))
