# 1. Define your variables for easy updates
export PROJECT_ID=$(gcloud config get-value project)
export REPO_PATH="us-central1-docker.pkg.dev/${PROJECT_ID}/simulations/covid-jax-tpu"

# 2. Build the image for the correct architecture (linux/amd64)
# We use --platform to ensure it runs correctly on the GCP TPU nodes
docker build --platform linux/amd64 -t $REPO_PATH:latest .

# 3. Push the image to the cloud
docker push $REPO_PATH:latest