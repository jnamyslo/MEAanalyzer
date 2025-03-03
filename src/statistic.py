import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

def check_normality(data):
    """
    Check if data is normally distributed using Shapiro-Wilk test
    Returns True if data is normally distributed (p > 0.05)
    """
    if len(data) < 3:  # Need at least 3 samples for Shapiro-Wilk test
        return False
        
    try:
        _, p_value = stats.shapiro(data)
        return p_value > 0.05
    except:
        return False

def perform_statistical_test(group1_data, group2_data):
    """
    Determine and perform appropriate statistical test:
    - If both groups are normally distributed with similar variance: t-test
    - Otherwise: Mann-Whitney U test
    
    Handles both scalar and array-like data
    """
    # Ensure we have scalar values
    group1_flat = []
    group2_flat = []
    
    # Extract scalar values from group1
    for item in group1_data:
        # For Spike Contrast (synchrony_with_trace), extract the synchrony value
        if isinstance(item, dict) and 'synchrony' in item:
            group1_flat.append(item['synchrony'])
        elif isinstance(item, (list, np.ndarray)) and hasattr(item, 'size') and item.size > 1:
            # For other array-like data, use max
            group1_flat.append(np.max(item))
        else:
            # For scalar values
            group1_flat.append(item)
    
    # Extract scalar values from group2
    for item in group2_data:
        # For Spike Contrast (synchrony_with_trace), extract the synchrony value
        if isinstance(item, dict) and 'synchrony' in item:
            group2_flat.append(item['synchrony'])
        elif isinstance(item, (list, np.ndarray)) and hasattr(item, 'size') and item.size > 1:
            # For other array-like data, use max
            group2_flat.append(np.max(item))
        else:
            # For scalar values
            group2_flat.append(item)
    
    # Now continue with original function logic
    if len(group1_flat) == 0 or len(group2_flat) == 0:
        return 1.0, "No data"
    
    # Convert to flat numpy arrays to handle various input types
    group1_data_flat = np.asarray(group1_flat).flatten()
    group2_data_flat = np.asarray(group2_flat).flatten()
    
    # Check for normality
    is_normal_group1 = check_normality(group1_data_flat)
    is_normal_group2 = check_normality(group2_data_flat)
    
    # Check for equal variance if normally distributed
    if is_normal_group1 and is_normal_group2:
        # Levene test for equal variances
        _, p_var = stats.levene(group1_data_flat, group2_data_flat)
        equal_var = p_var > 0.05
        
        # Perform t-test if normally distributed
        stat, p_value = stats.ttest_ind(group1_data_flat, group2_data_flat, equal_var=equal_var)
        test_name = "t-test" if equal_var else "Welch's t-test"
    else:
        # Perform Mann-Whitney U test if not normally distributed
        stat, p_value = stats.mannwhitneyu(group1_data_flat, group2_data_flat, alternative='two-sided')
        test_name = "Mann-Whitney U"
    
    # Ensure we return a scalar p-value
    if hasattr(p_value, 'item'):
        p_value = p_value.item()
    
    return p_value, test_name

def add_significance_marker(ax, x1, x2, y, p_value, height=0.05):
    """Add significance bars and stars to the plot"""
    # Calculate significance marker
    if p_value <= 0.001:
        sig_symbol = '***'
    elif p_value <= 0.01:
        sig_symbol = '**'
    elif p_value <= 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
        
    # Draw the line
    bar_height = height * (ax.get_ylim()[1] - ax.get_ylim()[0])
    y_pos = y + bar_height * 0.1
    
    # Plot the bar
    ax.plot([x1, x1, x2, x2], [y_pos, y_pos + bar_height, y_pos + bar_height, y_pos], color='black', lw=1)
    
    # Add the significance star
    ax.text((x1 + x2) / 2, y_pos + bar_height, sig_symbol, 
           ha='center', va='bottom', color='black')

def plot_feature_values_with_stats(all_features_data, output_dir, custom_labels=None):
    """Create boxplots with statistical significance markers, normalized to reference"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Statistical test results table
    stat_results = []
    
    for feature_name, time_dict in all_features_data.items():
        # Skip Spike Contrast Trace features that we don't need
        if feature_name.startswith('Spike Contrast Trace'):
            continue
            
        time_indices_sorted = sorted(time_dict.keys(), key=lambda x: int(x))
        if custom_labels and len(custom_labels) == len(time_indices_sorted):
            x_labels = custom_labels
        else:
            x_labels = time_indices_sorted
            print(f"Warning: Custom labels not found or count mismatch for {feature_name}")
        
        group_names = sorted(list(set().union(*[time_dict[t].keys() for t in time_indices_sorted])))
        
        # Only proceed if we have exactly 2 groups (SHAM and treatment)
        if len(group_names) != 2:
            print(f"Warning: Expected 2 groups, found {len(group_names)} for {feature_name}")
            continue
        
        # Prepare data for boxplots - handle special case for Spike Contrast
        data_for_boxplot = []
        for t in time_indices_sorted:
            group_list_for_this_time = []
            for g in group_names:
                group_values = time_dict[t].get(g, [])
                
                # Process the values to ensure we have scalar values
                processed_values = []
                for val in group_values:
                    # For Spike Contrast (synchrony_with_trace), extract the synchrony value
                    if isinstance(val, dict) and 'synchrony' in val:
                        processed_values.append(val['synchrony'])
                    elif isinstance(val, (list, np.ndarray)) and hasattr(val, 'size') and val.size > 1:
                        processed_values.append(np.max(val))
                    else:
                        processed_values.append(val)
                
                group_list_for_this_time.append(processed_values)
            data_for_boxplot.append(group_list_for_this_time)
        
        # Create normalized data relative to reference measurements
        data_for_boxplot_relative = []
        ref_means_per_group = []
        first_time_idx_data = data_for_boxplot[0]  # Reference measurement
        
        # Get reference values for each group
        for group_idx, group_values in enumerate(first_time_idx_data):
            if len(group_values) > 0:
                ref_means_per_group.append(np.median(group_values))
            else:
                ref_means_per_group.append(1.0)  # Avoid division by zero
        
        # Normalize all values relative to reference
        for time_idx, group_values_list in enumerate(data_for_boxplot):
            group_list_for_this_time_rel = []
            for group_idx, group_values in enumerate(group_values_list):
                if ref_means_per_group[group_idx] != 0:
                    # Values are normalized and then -1 so reference becomes 0
                    norm_values = [(val / ref_means_per_group[group_idx]) - 1 for val in group_values]
                else:
                    norm_values = [0 for _ in group_values]
                group_list_for_this_time_rel.append(norm_values)
            data_for_boxplot_relative.append(group_list_for_this_time_rel)
        
        # Calculate statistics for each time point after reference using normalized data
        p_values = []
        test_types = []
        max_values = []
        
        for time_idx in range(1, len(time_indices_sorted)):  # Skip first time point (reference)
            sham_data = data_for_boxplot_relative[time_idx][0]
            treatment_data = data_for_boxplot_relative[time_idx][1]
            
            p_value, test_type = perform_statistical_test(sham_data, treatment_data)
            
            # Ensure p_value is a scalar
            if hasattr(p_value, 'size') and p_value.size > 1:
                p_value = float(p_value[0])
            elif hasattr(p_value, 'item'):
                p_value = p_value.item()
            
            p_values.append(p_value)
            test_types.append(test_type)
            
            # Find maximum value for positioning the significance bar
            all_values = sham_data + treatment_data
            if all_values:
                # Use list comprehension to ensure we only compare scalar values
                scalar_values = [v for v in all_values if np.isscalar(v)]
                max_val = max(scalar_values) if scalar_values else 0
            else:
                max_val = 0
                
            max_values.append(max_val)
            
            # Save results to table
            stat_results.append({
                'Feature': feature_name,
                'Time Point': x_labels[time_idx],
                'Test Type': test_type,
                'p-value': p_value,
                'Significant': 'Yes' if p_value <= 0.05 else 'No'
            })
        
        # Create symlog plot with significance markers (similar to boxplots.py relative plot)
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.set_style("whitegrid")
        
        n_times = len(time_indices_sorted)
        n_groups = len(group_names)
        width = 0.6 / n_groups
        colors = plt.cm.Set1.colors
        
        # Create boxplots using relative data
        for time_idx, group_values_list in enumerate(data_for_boxplot_relative):
            for group_idx, group_values in enumerate(group_values_list):
                position = time_idx + group_idx * width - (width * (n_groups - 1) / 2)
                bp = ax.boxplot(group_values,
                               positions=[position],
                               widths=width,
                               patch_artist=True,
                               manage_ticks=False,
                               showfliers=False)
                
                for box in bp['boxes']:
                    box.set(facecolor=colors[group_idx % len(colors)])
        
        # Add significance markers for each time point after reference
        for time_idx in range(1, len(time_indices_sorted)):
            pos1 = time_idx - width/2
            pos2 = time_idx + width/2
            p_value = p_values[time_idx-1]
            
            if not np.isnan(max_values[time_idx-1]):
                add_significance_marker(
                    ax, pos1, pos2, max_values[time_idx-1],
                    p_value, height=0.07
                )
        
        # Set plot labels and title
        ax.set_title(f"Relative Change in {feature_name} with Statistical Significance", fontsize=14)
        ax.set_ylabel(f"Relative Change (normalized to reference)", fontsize=12)
        
        # Use symlog scale like in boxplots.py
        #ax.set_yscale('symlog', linthresh=0.5)
        
        # Add reference line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        ax.set_xticks(range(n_times))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
        ax.set_xlabel("Time Point", fontsize=12)
        
        # Add legend
        legend_patches = [
            Patch(color=colors[idx % len(colors)], label=group)
            for idx, group in enumerate(group_names)
        ]
        
        sig_legend = [
            Patch(color='white', label='* p ≤ 0.05'),
            Patch(color='white', label='** p ≤ 0.01'),
            Patch(color='white', label='*** p ≤ 0.001'),
            Patch(color='white', label='ns p > 0.05')
        ]
        
        all_legends = legend_patches + sig_legend
        ax.legend(handles=all_legends, loc='upper left', ncol=2, 
                 bbox_to_anchor=(1, 1), fontsize=10)
        
        plt.tight_layout()
        filename_safe = feature_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        plt.savefig(os.path.join(output_dir, f'{filename_safe}_rel_stat_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save statistical test results to CSV
    if stat_results:
        df_stats = pd.DataFrame(stat_results)
        df_stats.to_csv(os.path.join(output_dir, 'statistical_test_results.csv'), index=False)
        print(f"Statistical test results saved to {output_dir}/statistical_test_results.csv")

def analyze_repeated_measures(all_features_data, output_dir, custom_labels=None):
    """
    Perform repeated measures analysis across time points
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    rm_results = []
    
    for feature_name, time_dict in all_features_data.items():
        time_indices_sorted = sorted(time_dict.keys(), key=lambda x: int(x))
        group_names = sorted(list(set().union(*[time_dict[t].keys() for t in time_indices_sorted])))
        
        # Skip if we don't have exactly 2 groups
        if len(group_names) != 2:
            continue
        
        # Check if we have data for all chips across all time points
        sham_chips = set()
        treatment_chips = set()
        
        for t_idx in time_indices_sorted:
            for g_idx, group in enumerate(group_names):
                values = time_dict[t_idx].get(group, [])
                if g_idx == 0:  # SHAM
                    sham_chips.update(range(len(values)))
                else:  # Treatment
                    treatment_chips.update(range(len(values)))
        
        # Prepare data for two-way repeated measures ANOVA
        # This is simplified and would need more precise chip tracking in real implementation
        try:
            # For each time point, get SHAM vs treatment data
            sham_data = []
            treatment_data = []
            
            for t_idx in time_indices_sorted:
                sham_values = time_dict[t_idx].get(group_names[0], [])
                treatment_values = time_dict[t_idx].get(group_names[1], [])
                
                sham_data.append(sham_values)
                treatment_data.append(treatment_values)
            
            # If we have data for all time points, try to do a Friedman test (non-parametric repeated measures)
            if min(len(s) for s in sham_data) > 0 and min(len(t) for t in treatment_data) > 0:
                # For each group separately (within-group effect of time)
                _, p_sham = stats.friedmanchisquare(*sham_data)
                _, p_treatment = stats.friedmanchisquare(*treatment_data)
                
                rm_results.append({
                    'Feature': feature_name,
                    'Group': group_names[0],
                    'Test': 'Friedman',
                    'p-value': p_sham,
                    'Significant': 'Yes' if p_sham <= 0.05 else 'No'
                })
                
                rm_results.append({
                    'Feature': feature_name,
                    'Group': group_names[1],
                    'Test': 'Friedman',
                    'p-value': p_treatment,
                    'Significant': 'Yes' if p_treatment <= 0.05 else 'No'
                })
        except Exception as e:
            print(f"Error in repeated measures analysis for {feature_name}: {e}")
    
    # Save repeated measures results
    if rm_results:
        df_rm = pd.DataFrame(rm_results)
        df_rm.to_csv(os.path.join(output_dir, 'repeated_measures_results.csv'), index=False)
        print(f"Repeated measures analysis saved to {output_dir}/repeated_measures_results.csv")

def main():
    parent_dir = "data"
    
    # Check for custom labels
    label_file_path = os.path.join(parent_dir, "labels.txt")
    if os.path.isfile(label_file_path):
        with open(label_file_path, "r", encoding="utf-8") as f:
            labels_str = f.read().strip()
            custom_labels = [e.strip() for e in labels_str.split(";")] if labels_str else None
            print("Custom labels found.")
    else:
        custom_labels = None
        print("No custom labels found.")
    
    # Find all group directories
    groups = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    all_features_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Load feature data from all npz files
    for group in groups:
        group_path = os.path.join(parent_dir, group)
        chip_dirs = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
        
        for chip_dir in chip_dirs:
            chip_path = os.path.join(group_path, chip_dir)
            npz_files = sorted([f for f in os.listdir(chip_path) if f.endswith('.npz')])
            
            for idx, npz_file in enumerate(npz_files, start=1):
                npz_path = os.path.join(chip_path, npz_file)
                print(f"Reading features from: Group={group}, Chip={chip_dir}, File={npz_file}, Time Index={idx}")
                
                with np.load(npz_path, allow_pickle=True) as data:
                    if 'features' in data:
                        features = data['features'].item()
                        for feat_key, feat_val in features.items():
                            # Skip Spike Contrast Trace features
                            if feat_key.startswith('Spike Contrast Trace'):
                                continue
                                
                            # Add feature value to our data structure
                            if feat_key == 'Synchrony (Spike Contrast)':
                                # Store synchrony_with_trace directly
                                all_features_data[feat_key][str(idx)][group].append(feat_val)
                            else:
                                # Store other features normally
                                all_features_data[feat_key][str(idx)][group].append(feat_val)
    
    # Output directories
    stats_output_dir = os.path.join(parent_dir, 'Statistical_Analysis')
    
    # Generate plots with statistical significance
    plot_feature_values_with_stats(all_features_data, stats_output_dir, custom_labels)
    
    # Perform repeated measures analysis
    analyze_repeated_measures(all_features_data, stats_output_dir, custom_labels)
    
    print("Finished. Statistical analysis and boxplots have been created.")

if __name__ == "__main__":
    main()
