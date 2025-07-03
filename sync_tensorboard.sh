#!/bin/bash

# Sync TensorBoard logs from training instance
INSTANCE_IP="54.157.217.145"
REMOTE_PATH="llava_s3_bucket_1751489259_LLaVA-NeXT_new_development_branch/work_dirs/*/runs/"
LOCAL_PATH="tensorboard_logs/"

echo "Starting TensorBoard sync for training on $INSTANCE_IP"

while true; do
    echo "$(date): Syncing TensorBoard logs..."
    rsync -avz -e "ssh -o StrictHostKeyChecking=no -i /home/veesion/aws/scalable_training_ireland.pem" \
        ec2-user@$INSTANCE_IP:$REMOTE_PATH $LOCAL_PATH 2>/dev/null || echo "Sync failed, retrying..."
    
    sleep 30  # Sync every 30 seconds
done 