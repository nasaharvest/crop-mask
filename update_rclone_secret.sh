# This script simplifies the process of updated the rclone secret on AWS.
# The rclone credential file expires very often (because of Google Drive) 
# and thus needs to be constantly updated.

# Checks rclone is working
rclone version

# Refreshes token
rclone lsd remote2:

# Uploads secret
aws secretsmanager update-secret \
    --secret-id ivan/rclone.conf \
    --secret-string "$(rclone config show)"
