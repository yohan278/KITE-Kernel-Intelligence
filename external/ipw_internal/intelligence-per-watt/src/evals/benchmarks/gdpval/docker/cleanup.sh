#!/bin/bash
# Cleanup script for GDPval Docker containers

echo "Stopping and removing GDPval containers..."

# Stop and remove all containers using the gdpval image
docker ps -a --filter ancestor=gdpval --format "{{.ID}}" | xargs -r docker rm -f

# Show remaining containers (if any)
REMAINING=$(docker ps -a --filter ancestor=gdpval --format "{{.ID}}" | wc -l)

if [ "$REMAINING" -eq 0 ]; then
    echo "✓ All GDPval containers cleaned up"
else
    echo "⚠ Warning: $REMAINING containers still running"
fi

# Optional: Remove the gdpval image itself (uncomment if needed)
# docker rmi gdpval

