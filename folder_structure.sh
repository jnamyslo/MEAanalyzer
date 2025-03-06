#!/bin/bash

cd /app/data
declare -A group_ids
# <EDIT START HERE>

# Define your experimental groups and their associated IDs
# Modify these arrays according to your actual grouping
group_ids["SHAM"]="02 04 06 08"
group_ids["BDNF"]="01 03 05 07"

chipprefix="ID2024"
# ADD GROUPS HERE group_ids["GROUP"]="XX XX XX"

# < EDIT END HERE >


# Create group folders
for group in "${!group_ids[@]}"; do
  mkdir -p "$group"
  
  # Create ID folders for each group
  for id in ${group_ids[$group]}; do
    mkdir -p "$group/$chipprefix-$id"
  done
done

# Find all .bxr files in the current directory and move them to appropriate folders
find . -maxdepth 1 -name "*.bxr" | while read file; do
  filename=$(basename "$file")
  
  # Extract ID from filename (assuming format ID2024-XX in the filename)
  if [[ $filename =~ $chipprefix-([0-9][0-9]) ]]; then
    id="${BASH_REMATCH[1]}"
    
    # Determine which group this ID belongs to
    target_group=""
    for group in "${!group_ids[@]}"; do
      if [[ " ${group_ids[$group]} " =~ " $id " ]]; then
        target_group=$group
        break
      fi
    done
    
    # Move file if we found a matching group
    if [ ! -z "$target_group" ]; then
      echo "Moving $filename to $target_group/$chipprefix-$id/"
      mv "$file" "$target_group/$chipprefix-$id/"
    else
      echo "Warning: No group found for ID $id in file $filename"
    fi
  else
    echo "Warning: Could not extract ID from filename $filename"
  fi
done

echo "File organization complete"



# MISCELLANEOUS (Handy commands)

# Revert it with 
#find . -type f -name "*.bxr" -exec mv {} . \;

# Remove specific files
#find . -type f -name "*_NBT*" -exec rm -f {} +

# Remove specific folders
#find . -type d -name "*Rasterplots*" -exec rm -rf {} +

