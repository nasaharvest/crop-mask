# Checks rclone is working
rclone version

# Refreshes token
rclone lsd remote2:

# Uploads secret
aws secretsmanager update-secret \
    --secret-id ivan/rclone.conf \
    --secret-string "$(rclone config show)"