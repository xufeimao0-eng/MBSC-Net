"""
Weibo Dataset Preprocessing Pipeline:
  1. Remove duplicate & low-quality images
  2. Single-pass event clustering from post text
  3. Event-level 7:1:2 train/val/test split
"""

import os, sys, json, random
import hashlib
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from PIL import Image
import imagehash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# ─── Step 1: Image cleaning ─────────────────────────────────────────────

def is_low_quality(img_path, min_size_kb=2, min_dim=32):
    """Check if image is too small or corrupt."""
    try:
        size_kb = os.path.getsize(img_path) / 1024
        if size_kb < min_size_kb:
            return True
        with Image.open(img_path) as im:
            w, h = im.size
            if w < min_dim or h < min_dim:
                return True
        return False
    except Exception:
        return True  # corrupt


def deduplicate_images(image_dir):
    """
    Scan image directory, remove near-duplicates via perceptual hash.
    Returns: {filename: 'keep'/'duplicate'}
    """
    print(f"\n[Step 1] Scanning images in {image_dir} ...")
    files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    print(f"  Total images: {len(files)}")

    seen_hashes = {}
    status = {}  # filename → 'keep' | 'duplicate' | 'low_quality'
    low_quality_count = 0

    for fname in tqdm(files, desc="  Hashing"):
        fpath = os.path.join(image_dir, fname)

        if is_low_quality(fpath):
            status[fname] = 'low_quality'
            low_quality_count += 1
            continue

        try:
            im = Image.open(fpath)
            ph = imagehash.phash(im)
        except Exception:
            status[fname] = 'low_quality'
            low_quality_count += 1
            continue

        # Check for near-duplicate (hamming distance <= 5)
        is_dup = False
        for existing_hash in list(seen_hashes.keys()):
            if ph - existing_hash <= 5:
                status[fname] = 'duplicate'
                is_dup = True
                break

        if not is_dup:
            seen_hashes[ph] = fname
            status[fname] = 'keep'

    dup_count = sum(1 for v in status.values() if v == 'duplicate')
    keep_count = sum(1 for v in status.values() if v == 'keep')
    print(f"  Kept: {keep_count}  |  Duplicates: {dup_count}  |  Low-quality: {low_quality_count}")
    return status


# ─── Step 2: Event clustering ────────────────────────────────────────────

def cluster_events(posts, similarity_threshold=0.3, max_features=5000):
    """
    Single-pass clustering: assign each post to the nearest existing cluster,
    or create a new cluster if no match above threshold.

    Args:
        posts: list of dicts with 'id', 'text'
        similarity_threshold: cosine similarity threshold for cluster assignment
        max_features: max TF-IDF features

    Returns:
        posts with 'event_id' field added
    """
    print(f"\n[Step 2] Clustering {len(posts)} posts into events ...")
    print(f"  Threshold: {similarity_threshold}")

    # Build TF-IDF matrix
    texts = [p['text'] for p in posts]
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        analyzer='char',
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Single-pass clustering
    from scipy.sparse import vstack
    cluster_centroids = []  # list of (cluster_id, centroid_vector)
    event_ids = [-1] * len(posts)
    n_clusters = 0

    for i in tqdm(range(len(posts)), desc="  Clustering"):
        vec = tfidf_matrix[i]

        if not cluster_centroids:
            # First post → first cluster
            cluster_centroids.append((n_clusters, vec))
            event_ids[i] = n_clusters
            n_clusters += 1
            continue

        # Compute similarity to all existing clusters
        centroids_sparse = vstack([c[1] for c in cluster_centroids])
        centroid_matrix = cosine_similarity(vec, centroids_sparse)[0]

        best_idx = np.argmax(centroid_matrix)
        best_sim = centroid_matrix[best_idx]

        if best_sim >= similarity_threshold:
            # Assign to existing cluster
            cid = cluster_centroids[best_idx][0]
            event_ids[i] = cid
            # Update centroid (moving average)
            old_centroid = cluster_centroids[best_idx][1]
            n_in_cluster = sum(1 for e in event_ids[:i+1] if e == cid)
            new_centroid = old_centroid + (vec - old_centroid) / n_in_cluster
            cluster_centroids[best_idx] = (cid, new_centroid)
        else:
            # Create new cluster
            cluster_centroids.append((n_clusters, vec))
            event_ids[i] = n_clusters
            n_clusters += 1

    for i, eid in enumerate(event_ids):
        posts[i]['event_id'] = int(eid)

    print(f"  Events found: {n_clusters}")
    # Event size distribution
    event_sizes = defaultdict(int)
    for eid in event_ids:
        event_sizes[eid] += 1
    sizes = sorted(event_sizes.values(), reverse=True)
    print(f"  Largest 5 events: {sizes[:5]}")
    print(f"  Median event size: {np.median(sizes):.0f}")
    print(f"  Singleton events: {sum(1 for s in sizes if s == 1)}")

    return posts


# ─── Step 3: Event-level split ──────────────────────────────────────────

def split_by_events(posts, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    Split posts into train/val/test ensuring no event appears in multiple splits.
    """
    print(f"\n[Step 3] Splitting by events ({train_ratio}:{val_ratio}:{test_ratio}) ...")

    # Group posts by event
    event_groups = defaultdict(list)
    for p in posts:
        event_groups[p['event_id']].append(p)

    events = list(event_groups.keys())
    print(f"  Total events: {len(events)}")

    rng = random.Random(seed)
    rng.shuffle(events)

    n_train = int(len(events) * train_ratio)
    n_val = int(len(events) * val_ratio)

    train_events = set(events[:n_train])
    val_events = set(events[n_train:n_train + n_val])
    test_events = set(events[n_train + n_val:])

    train_posts = []
    val_posts = []
    test_posts = []

    for eid, group in event_groups.items():
        if eid in train_events:
            train_posts.extend(group)
        elif eid in val_events:
            val_posts.extend(group)
        else:
            test_posts.extend(group)

    print(f"  Train: {len(train_posts)} posts ({len(train_events)} events)")
    print(f"  Val:   {len(val_posts)} posts ({len(val_events)} events)")
    print(f"  Test:  {len(test_posts)} posts ({len(test_events)} events)")

    # Verify no event overlap
    train_event_set = set(p['event_id'] for p in train_posts)
    val_event_set = set(p['event_id'] for p in val_posts)
    test_event_set = set(p['event_id'] for p in test_posts)
    assert train_event_set.isdisjoint(val_event_set), "Train-Val event overlap!"
    assert train_event_set.isdisjoint(test_event_set), "Train-Test event overlap!"
    assert val_event_set.isdisjoint(test_event_set), "Val-Test event overlap!"
    print("  ✓ No event overlap between splits")

    # Show label distribution
    for name, split_posts in [('Train', train_posts), ('Val', val_posts), ('Test', test_posts)]:
        real = sum(1 for p in split_posts if p['label'] == 0)
        fake = sum(1 for p in split_posts if p['label'] == 1)
        print(f"  {name}: {real} real + {fake} fake (real ratio: {real/(real+fake):.1%})")

    return train_posts, val_posts, test_posts


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Weibo Dataset Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Clean images
    rumor_status = deduplicate_images(Config.WEIBO21_RUMOR_IMG)
    nonrumor_status = deduplicate_images(Config.WEIBO21_NONRUMOR_IMG)

    # Step 2: Load data, filter bad images
    from data.preprocess import load_weibo21_data

    # Load all training data (samples with full info)
    print("\nLoading training data ...")
    raw_posts = load_weibo21_data(split="train", val_ratio=0.0)  # all train data

    # Filter out posts with duplicate/low-quality images
    all_status = {**rumor_status, **nonrumor_status}
    cleaned_posts = []
    removed_count = 0
    for p in raw_posts:
        # Keep only posts whose images are all 'keep' status
        bad_images = [img for img in p['image_paths']
                      if os.path.basename(img) in all_status
                      and all_status[os.path.basename(img)] != 'keep']
        if bad_images:
            removed_count += 1
            continue
        # If no images after filtering but original had images → remove
        valid_images = [img for img in p['image_paths']
                        if os.path.basename(img) in all_status
                        and all_status[os.path.basename(img)] == 'keep']
        if valid_images:
            p['image_paths'] = valid_images
            cleaned_posts.append(p)
        elif not p['image_paths']:
            # No images originally → keep (text-only)
            cleaned_posts.append(p)
        else:
            # Had images but all bad → remove
            removed_count += 1

    print(f"  Before cleaning: {len(raw_posts)} posts")
    print(f"  After cleaning:  {len(cleaned_posts)} posts ({removed_count} removed)")

    # Step 3: Event clustering
    cleaned_posts = cluster_events(cleaned_posts, similarity_threshold=0.3)

    # Step 4: Event-level split
    train, val, test = split_by_events(cleaned_posts)

    # Step 5: Save processed splits
    output_dir = os.path.join(Config.WEIBO21_DIR, "processed")
    os.makedirs(output_dir, exist_ok=True)

    for name, split_data in [('train', train), ('val', val), ('test', test)]:
        # Save as JSON (id, text, image_paths, label, event_id)
        output_path = os.path.join(output_dir, f"{name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {name}: {len(split_data)} posts → {output_path}")

    # Also save image status for reference
    status_path = os.path.join(output_dir, "image_status.json")
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(all_status, f, ensure_ascii=False)
    print(f"  Saved image status → {status_path}")

    # Summary
    real_train = sum(1 for p in train if p['label'] == 0)
    fake_train = sum(1 for p in train if p['label'] == 1)
    print(f"\n{'=' * 60}")
    print(f"Preprocessing complete!")
    print(f"  Train: {len(train)} ({real_train} real + {fake_train} fake)")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")
    print(f"  Total: {len(train)+len(val)+len(test)}")
    print(f"  Events: {len(set(p['event_id'] for p in cleaned_posts))}")
    print(f"  Removed: {removed_count} posts (bad images)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
