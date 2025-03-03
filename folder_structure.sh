#!/bin/bash

# Define your experimental groups and their associated IDs
# Modify these arrays according to your actual grouping
declare -A group_ids
group_ids["SHAM"]="04 08"
group_ids["LSD"]="01 02 03 05 06 07"

# Create group folders
for group in "${!group_ids[@]}"; do
  mkdir -p "$group"
  
  # Create ID folders for each group
  for id in ${group_ids[$group]}; do
    mkdir -p "$group/ID2024-$id"
  done
done

# Find all .bxr files in the current directory and move them to appropriate folders
find . -maxdepth 1 -name "*.bxr" | while read file; do
  filename=$(basename "$file")
  
  # Extract ID from filename (assuming format ID2024-XX in the filename)
  if [[ $filename =~ ID2024-([0-9][0-9]) ]]; then
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
      echo "Moving $filename to $target_group/ID2024-$id/"
      mv "$file" "$target_group/ID2024-$id/"
    else
      echo "Warning: No group found for ID $id in file $filename"
    fi
  else
    echo "Warning: Could not extract ID from filename $filename"
  fi
done

echo "File organization complete"


# Revert it with 
#find . -type f -name "*.bxr" -exec mv {} . \;

# Remove specific files
#find . -type f -name "*_NBT*" -exec rm -f {} +

# Remove specific folders
#find . -type d -name "*Rasterplots*" -exec rm -rf {} +

