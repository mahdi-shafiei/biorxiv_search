#!/bin/bash

# Run your Python script to update the database
python update_database.py

# Add changes to git
git add .

# Commit the changes with a message that includes "update" and the current date
DATE=$(date '+%Y-%m-%d')
git commit -m "update $DATE"

# Push the changes back to your GitHub repository
git push origin main