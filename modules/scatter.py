import os
import glob
import random
import omni.replicator.core as rep

def load_distractor_pools(assets_root_dir, distractor_config):
    """
    Loads distractor assets based on configuration.
    Returns a dictionary: { 'type_name': [path1, path2, ...] }
    """
    distractors_root = os.path.join(assets_root_dir, "distractors")
    loaded_pools = {}
    
    print(f"--- Loading Distractor Pools from: {distractors_root} ---")
    
    for category, settings in distractor_config.items():
        if not settings.get("active", False):
            continue
            
        category_path = os.path.join(distractors_root, category)
        if not os.path.exists(category_path):
             print(f"[WARN] Distractor category folder not found: {category_path}")
             continue

        # Recursive search for USDs
        found_usds = glob.glob(os.path.join(category_path, "**", "*.usd*"), recursive=True)
        
        # Filter/Limit based on pool_size config
        pool_size = settings.get("pool_size", 999)
        if len(found_usds) > pool_size:
            print(f"   -> Subsampling {pool_size} assets from {len(found_usds)} found for '{category}'")
            found_usds = random.sample(found_usds, pool_size)
        
        if found_usds:
            loaded_pools[category] = found_usds
            print(f"   -> Ready '{category}': {len(found_usds)} assets.")
        else:
            print(f"   -> No assets found for '{category}'")
            
    return loaded_pools

def create_distractor_instances(distractor_pools, distractor_config):
    """
    Creates the maximum number of instances needed for each category 
    but keeps them hidden initially.
    Returns: { 'category': [rep_item_list] }
    """
    scene_distractors = {}
    
    print("--- Instantiating Distractors (Hidden Pool) ---")
    
    for category, paths in distractor_pools.items():
        settings = distractor_config[category]
        # Calculate max density to reserve pool
        # density_range example: (5, 20) -> we need 20 instances max per frame
        max_count = settings.get("density_range", (0, 10))[1]
        
        items = []
        # We create 'max_count' instances. 
        # To avoid repetition, we cycle through the available USD paths.
        for i in range(max_count):
            usd_path = paths[i % len(paths)] # Cycle paths
            unique_name = f"distractor_{category}_{i}"
            
            # Create item using Replicator (hidden by default logic later)
            rep_item = rep.create.from_usd(
                usd_path,
                semantics=[('class', 'distractor')], # Label them so we can filter later if needed
                name=unique_name
            )
            
            # Modify: Hide initially and scale to config
            min_s, max_s = settings.get("scale_range", (1.0, 1.0))
            scale_val = random.uniform(min_s, max_s)
            
            with rep_item:
                rep.modify.visibility(False)
                rep.modify.pose(scale=scale_val)
                
            items.append(rep_item)
            
        scene_distractors[category] = items
        print(f"   -> Created {len(items)} instances for '{category}'")
        
    return scene_distractors
