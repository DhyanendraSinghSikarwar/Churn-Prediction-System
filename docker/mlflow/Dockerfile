# Minimal MLflow tracking server image: the upstream mlflow package alone
# doesn't bundle a Postgres driver or an S3 client, and the backend store
# (Postgres) + artifact store (MinIO/S3) both need them.
FROM python:3.12-slim

# setuptools>=81 dropped pkg_resources entirely; mlflow 2.14 still imports
# it at startup, so pin to the last version that bundles it.
RUN pip install --no-cache-dir "setuptools<81" mlflow==2.14.1 psycopg2-binary==2.9.9 boto3==1.34.129

EXPOSE 5000
