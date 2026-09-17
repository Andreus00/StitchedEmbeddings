"""
    Standalone evaluation of garment pattern predictions against ground truth.
    
    Operates directly on specification dicts (as loaded from JSON files)
    without requiring model objects, data loaders, or data statistics.
    
    Uses geometry-based panel matching (Hungarian algorithm) so that
    panels are matched by shape similarity, not by name.
"""

import numpy as np
import os
from scipy.optimize import linear_sum_assignment


# ======================== Helper functions ========================

def _extract_panels(spec):
    """Extract panel data from a specification dict.
    
    Returns a dict: panel_name -> {
        'vertices': np.array (N, 2),
        'edges': list of edge dicts,
        'num_edges': int,
        'translation': np.array (3,),
        'rotation': np.array (3,),
        'label': str or None
    }
    """
    panels = {}
    for name, panel_data in spec['pattern']['panels'].items():
        panels[name] = {
            'vertices': np.array(panel_data['vertices'], dtype=np.float64),
            'edges': panel_data['edges'],
            'num_edges': len(panel_data['edges']),
            'translation': np.array(panel_data['translation'], dtype=np.float64),
            'rotation': np.array(panel_data['rotation'], dtype=np.float64),
            'label': panel_data.get('label', None),
        }
    return panels


def _extract_stitches(spec):
    """Extract stitches as a set of frozensets for order-invariant comparison.
    
    Each stitch is: frozenset({(panel_name, edge_idx), (panel_name, edge_idx)})
    """
    stitches = set()
    for stitch_pair in spec['pattern'].get('stitches', []):
        side_a = (stitch_pair[0]['panel'], stitch_pair[0]['edge'])
        side_b = (stitch_pair[1]['panel'], stitch_pair[1]['edge'])
        stitches.add(frozenset({side_a, side_b}))
    return stitches


def _center_align_vertices(vertices):
    """Center-align vertices by subtracting the centroid."""
    centroid = np.mean(vertices, axis=0)
    return vertices - centroid


def _vertex_rmse(verts_a, verts_b):
    """Compute RMSE between two vertex arrays of the same shape.
    Returns per-vertex L2 mean (i.e. mean of per-vertex Euclidean distances).
    """
    diff = verts_a - verts_b
    per_vertex_l2 = np.sqrt(np.sum(diff ** 2, axis=1))
    return np.mean(per_vertex_l2)


def _pad_vertices_to_length(vertices, target_len):
    """Pad a vertex array with zeros to reach target_len rows."""
    if len(vertices) >= target_len:
        return vertices[:target_len]
    padding = np.zeros((target_len - len(vertices), vertices.shape[1]))
    return np.vstack([vertices, padding])


def _panel_geometry_distance(panel_a, panel_b):
    """Compute a geometry-based distance between two panels.
    
    Uses center-aligned vertices, padded to the same length.
    Returns a scalar distance.
    """
    verts_a = _center_align_vertices(panel_a['vertices'])
    verts_b = _center_align_vertices(panel_b['vertices'])
    
    max_len = max(len(verts_a), len(verts_b))
    verts_a = _pad_vertices_to_length(verts_a, max_len)
    verts_b = _pad_vertices_to_length(verts_b, max_len)
    
    # Try both orderings (original and reversed) to handle edge loop direction
    dist_forward = _vertex_rmse(verts_a, verts_b)
    dist_reversed = _vertex_rmse(verts_a, verts_b[::-1])
    
    # Also try all cyclic rotations to handle different starting edges
    best_dist = min(dist_forward, dist_reversed)
    for shift in range(1, max_len):
        rolled = np.roll(verts_b, shift, axis=0)
        best_dist = min(best_dist, _vertex_rmse(verts_a, rolled))
        rolled_rev = np.roll(verts_b[::-1], shift, axis=0)
        best_dist = min(best_dist, _vertex_rmse(verts_a, rolled_rev))
    
    return best_dist


def _match_panels_by_geometry(pred_panels, gt_panels):
    """Match predicted panels to GT panels using the Hungarian algorithm
    on geometry-based distances.
    
    Returns:
        matches: list of (pred_name, gt_name) tuples
        unmatched_pred: list of pred panel names with no GT match
        unmatched_gt: list of GT panel names with no pred match
    """
    pred_names = list(pred_panels.keys())
    gt_names = list(gt_panels.keys())
    
    n_pred = len(pred_names)
    n_gt = len(gt_names)
    
    if n_pred == 0 or n_gt == 0:
        return [], pred_names, gt_names
    
    # Build cost matrix
    cost_matrix = np.zeros((n_pred, n_gt))
    for i, pname in enumerate(pred_names):
        for j, gname in enumerate(gt_names):
            cost_matrix[i, j] = _panel_geometry_distance(pred_panels[pname], gt_panels[gname])
    
    # Hungarian algorithm (works on rectangular matrices)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    matches = []
    matched_pred = set()
    matched_gt = set()
    for r, c in zip(row_indices, col_indices):
        matches.append((pred_names[r], gt_names[c]))
        matched_pred.add(pred_names[r])
        matched_gt.add(gt_names[c])
    
    unmatched_pred = [n for n in pred_names if n not in matched_pred]
    unmatched_gt = [n for n in gt_names if n not in matched_gt]
    
    return matches, unmatched_pred, unmatched_gt


# ======================== Stitch matching with remapping ========================

def _remap_stitches(stitches, name_mapping):
    """Remap panel names in a stitch set according to a name mapping.
    
    Args:
        stitches: set of frozensets of (panel_name, edge_idx) tuples
        name_mapping: dict mapping old_name -> new_name
    
    Returns:
        remapped set of frozensets
    """
    remapped = set()
    for stitch in stitches:
        sides = list(stitch)
        remapped_sides = []
        skip = False
        for panel_name, edge_idx in sides:
            new_name = name_mapping.get(panel_name, panel_name)
            remapped_sides.append((new_name, edge_idx))
        remapped.add(frozenset(remapped_sides))
    return remapped


# ======================== Main evaluation function ========================

def evaluate_prediction(prediction, ground_truth):
    """Evaluate a predicted garment pattern specification against ground truth.
    
    Both inputs should be dicts as loaded from specification JSON files, 
    containing at minimum: pattern -> panels (with vertices, edges, translation, rotation)
    and optionally pattern -> stitches.
    
    Panels are matched using geometry-based matching (Hungarian algorithm on 
    vertex distances), so correct panels with different names are still matched.
    
    Args:
        prediction: dict, predicted specification (e.g. from final_prediction_specification.json)
        ground_truth: dict, ground truth specification (e.g. from initial_prediction_specification.json)
    
    Returns:
        dict with all computed metrics
    """
    pred_panels = _extract_panels(prediction)
    gt_panels = _extract_panels(ground_truth)
    
    results = {}
    
    # ---- 1. Panel count metrics ----
    results['num_panels_pred'] = len(pred_panels)
    results['num_panels_gt'] = len(gt_panels)
    results['num_panels_correct'] = (len(pred_panels) == len(gt_panels))
    
    # ---- 2. Geometry-based panel matching ----
    matches, unmatched_pred, unmatched_gt = _match_panels_by_geometry(pred_panels, gt_panels)
    
    results['panel_matches'] = [(p, g) for p, g in matches]
    results['unmatched_pred_panels'] = unmatched_pred
    results['unmatched_gt_panels'] = unmatched_gt
    results['num_matched_panels'] = len(matches)
    
    # Build name mapping: pred_name -> gt_name (for stitch remapping)
    pred_to_gt_name = {p: g for p, g in matches}
    
    # ---- 3. Per-panel metrics on matched panels ----
    per_panel_edge_accuracy = {}
    per_panel_vertex_l2 = {}
    per_panel_translation_l2 = {}
    per_panel_rotation_l2 = {}
    
    for pred_name, gt_name in matches:
        pred_p = pred_panels[pred_name]
        gt_p = gt_panels[gt_name]
        
        # Edge count accuracy
        edges_correct = (pred_p['num_edges'] == gt_p['num_edges'])
        per_panel_edge_accuracy[f'{pred_name} <-> {gt_name}'] = edges_correct
        
        # Vertex shape L2 (center-aligned)
        vertex_dist = _panel_geometry_distance(pred_p, gt_p)
        per_panel_vertex_l2[f'{pred_name} <-> {gt_name}'] = vertex_dist
        
        # Translation L2
        trans_diff = pred_p['translation'] - gt_p['translation']
        trans_l2 = np.sqrt(np.sum(trans_diff ** 2))
        per_panel_translation_l2[f'{pred_name} <-> {gt_name}'] = trans_l2
        
        # Rotation L2
        rot_diff = pred_p['rotation'] - gt_p['rotation']
        rot_l2 = np.sqrt(np.sum(rot_diff ** 2))
        per_panel_rotation_l2[f'{pred_name} <-> {gt_name}'] = rot_l2
    
    # ---- 4. Aggregate metrics ----
    if matches:
        edge_acc_values = list(per_panel_edge_accuracy.values())
        results['avg_num_edges_accuracy'] = sum(edge_acc_values) / len(edge_acc_values)
        
        vertex_l2_values = list(per_panel_vertex_l2.values())
        results['avg_vertex_l2'] = np.mean(vertex_l2_values)
        
        trans_l2_values = list(per_panel_translation_l2.values())
        results['avg_translation_l2'] = np.mean(trans_l2_values)
        
        rot_l2_values = list(per_panel_rotation_l2.values())
        results['avg_rotation_l2'] = np.mean(rot_l2_values)
    else:
        results['avg_num_edges_accuracy'] = 0.0
        results['avg_vertex_l2'] = float('inf')
        results['avg_translation_l2'] = float('inf')
        results['avg_rotation_l2'] = float('inf')
    
    results['per_panel_num_edges_accuracy'] = per_panel_edge_accuracy
    results['per_panel_vertex_l2'] = per_panel_vertex_l2
    results['per_panel_translation_l2'] = per_panel_translation_l2
    results['per_panel_rotation_l2'] = per_panel_rotation_l2
    
    # ---- 5. Stitch precision & recall ----
    pred_stitches = _extract_stitches(prediction)
    gt_stitches = _extract_stitches(ground_truth)
    
    # Remap predicted stitch panel names to GT names using the matching
    pred_stitches_remapped = _remap_stitches(pred_stitches, pred_to_gt_name)
    
    if pred_stitches_remapped:
        correct_pred_stitches = pred_stitches_remapped & gt_stitches
        results['stitch_precision'] = len(correct_pred_stitches) / len(pred_stitches_remapped)
    else:
        results['stitch_precision'] = 0.0
    
    if gt_stitches:
        correct_gt_stitches = pred_stitches_remapped & gt_stitches
        results['stitch_recall'] = len(correct_gt_stitches) / len(gt_stitches)
    else:
        results['stitch_recall'] = 1.0  # no stitches to recall
    
    results['num_stitches_pred'] = len(pred_stitches)
    results['num_stitches_gt'] = len(gt_stitches)
    results['num_correct_stitches'] = len(pred_stitches_remapped & gt_stitches) if gt_stitches else 0
    
    return results


# ======================== Convenience: pretty printing ========================

def print_metrics(metrics):
    """Print metrics dict in a readable format."""
    print("=" * 60)
    print("  GARMENT PATTERN EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n--- Panel Counts ---")
    print(f"  Predicted panels:  {metrics['num_panels_pred']}")
    print(f"  GT panels:         {metrics['num_panels_gt']}")
    print(f"  Count correct:     {metrics['num_panels_correct']}")
    print(f"  Matched panels:    {metrics['num_matched_panels']}")
    
    if metrics['unmatched_pred_panels']:
        print(f"  Unmatched (pred):  {metrics['unmatched_pred_panels']}")
    if metrics['unmatched_gt_panels']:
        print(f"  Unmatched (GT):    {metrics['unmatched_gt_panels']}")
    
    print(f"\n--- Panel Matching ---")
    for pred_name, gt_name in metrics['panel_matches']:
        print(f"  {pred_name:20s} <-> {gt_name}")
    
    print(f"\n--- Aggregate Metrics (on matched panels) ---")
    print(f"  Avg edge count accuracy:    {metrics['avg_num_edges_accuracy']:.4f}")
    print(f"  Avg vertex L2 (cm):         {metrics['avg_vertex_l2']:.4f}")
    print(f"  Avg translation L2 (cm):    {metrics['avg_translation_l2']:.4f}")
    print(f"  Avg rotation L2 (rad):      {metrics['avg_rotation_l2']:.4f}")
    
    print(f"\n--- Stitch Metrics ---")
    print(f"  Predicted stitches:  {metrics['num_stitches_pred']}")
    print(f"  GT stitches:         {metrics['num_stitches_gt']}")
    print(f"  Correct stitches:    {metrics['num_correct_stitches']}")
    print(f"  Precision:           {metrics['stitch_precision']:.4f}")
    print(f"  Recall:              {metrics['stitch_recall']:.4f}")
    
    print(f"\n--- Per-Panel Details ---")
    for key in metrics['per_panel_vertex_l2']:
        print(f"  {key}:")
        print(f"    edges correct: {metrics['per_panel_num_edges_accuracy'][key]}")
        print(f"    vertex L2:     {metrics['per_panel_vertex_l2'][key]:.4f} cm")
        print(f"    translation:   {metrics['per_panel_translation_l2'][key]:.4f} cm")
        print(f"    rotation:      {metrics['per_panel_rotation_l2'][key]:.4f} rad")
    
    print("=" * 60)


if __name__ == '__main__':
    import json
    import sys

    # res_path = './results_gcd_testset_2/default_body/data'
    res_path = './results_gcd_testset_rdm_loss_ckptv20_lr1e-3/default_body/data'
    # res_path = './edit_predictions_in_latent_space_test_samples_500_steps_full_testset'

    data_paths = os.listdir(res_path)
    
    all_initial_metrics = []
    all_final_metrics = []
    all_paths = []
    for el in data_paths:
        # gt_pattern = os.path.join(res_path, el, 'gt/gt_specification.json')
        # initial_pred_pattern = os.path.join(res_path, el, 'edit_prediction_0/edit_prediction_0_specification.json')
        # final_pred_pattern = os.path.join(res_path, el, 'edit_prediction_499/edit_prediction_499_specification.json')
        gt_pattern = os.path.join(res_path, el, 'gt_pattern.json')
        initial_pred_pattern = os.path.join(res_path, el, 'initial_prediction/initial_prediction_specification.json')
        final_pred_pattern = os.path.join(res_path, el, 'final_prediction/final_prediction_specification.json')
        try:
            with open(gt_pattern) as f:
                ground_truth = json.load(f)
            with open(initial_pred_pattern) as f:
                initial_prediction = json.load(f)
            with open(final_pred_pattern) as f:
                final_prediction = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {el}")
            continue

        # print(f"\n{'=' * 60}")
        # print(f"Evaluating {el}")
        # print(f"{'=' * 60}")

        metrics = evaluate_prediction(initial_prediction, ground_truth)
        # print_metrics(metrics)
        all_initial_metrics.append(metrics)

        metrics = evaluate_prediction(final_prediction, ground_truth)
        # print_metrics(metrics)
        all_final_metrics.append(metrics)

        all_paths.append(os.path.join(res_path, el))

    print("\n" + "=" * 60)
    print("  AVERAGE METRICS")
    print("=" * 60)
    print(f"  Avg initial num panels:              {np.mean([m['num_panels_correct'] for m in all_initial_metrics]):.4f}")
    print(f"  Avg final num panels:              {np.mean([m['num_panels_correct'] for m in all_final_metrics]):.4f}")
    print(f"  Avg initial vertex L2:     {np.mean([m['avg_vertex_l2'] for m in all_initial_metrics]):.4f} cm")
    print(f"  Avg final vertex L2:     {np.mean([m['avg_vertex_l2'] for m in all_final_metrics]):.4f} cm")
    print(f"  Avg initial translation L2:    {np.mean([m['avg_translation_l2'] for m in all_initial_metrics]):.4f} cm")
    print(f"  Avg final translation L2:    {np.mean([m['avg_translation_l2'] for m in all_final_metrics]):.4f} cm")
    print(f"  Avg initial rotation L2:      {np.mean([m['avg_rotation_l2'] for m in all_initial_metrics]):.4f} rad")
    print(f"  Avg final rotation L2:      {np.mean([m['avg_rotation_l2'] for m in all_final_metrics]):.4f} rad")
    print(f"  Avg initial stitch precision:           {np.mean([m['stitch_precision'] for m in all_initial_metrics]):.4f}")
    print(f"  Avg final stitch precision:           {np.mean([m['stitch_precision'] for m in all_final_metrics]):.4f}")
    print(f"  Avg initial stitch recall:              {np.mean([m['stitch_recall'] for m in all_initial_metrics]):.4f}")
    print(f"  Avg final stitch recall:              {np.mean([m['stitch_recall'] for m in all_final_metrics]):.4f}")
    print(f"  Avg initial num stitches:              {np.mean([m['num_stitches_pred'] for m in all_initial_metrics]):.4f}")
    print(f"  Avg final num stitches:              {np.mean([m['num_stitches_pred'] for m in all_final_metrics]):.4f}")
    print(f"  Avg initial num correct stitches:              {np.mean([m['num_correct_stitches'] for m in all_initial_metrics]):.4f}")
    print(f"  Avg final num correct stitches:              {np.mean([m['num_correct_stitches'] for m in all_final_metrics]):.4f}")
    print(f"  Avg initial avg num edges accuracy:              {np.mean([m['avg_num_edges_accuracy'] for m in all_initial_metrics]):.4f}")
    print(f"  Avg final avg num edges accuracy:              {np.mean([m['avg_num_edges_accuracy'] for m in all_final_metrics]):.4f}")
    

    # ── Per-sample improvement: positive = final better than initial ──────────
    improvements = []
    for i, (init_m, final_m, path) in enumerate(zip(all_initial_metrics, all_final_metrics, all_paths)):
        # improvement = initial_error - final_error (positive means optimization helped)
        delta = init_m['avg_vertex_l2'] - final_m['avg_vertex_l2']
        improvements.append((delta, init_m['avg_vertex_l2'], final_m['avg_vertex_l2'], path))

    improvements.sort(key=lambda x: x[0], reverse=True)

    N = 5
    print(f"\n{'=' * 60}")
    print(f"  TOP {N} — Final BETTER than Initial (largest improvement)")
    print(f"{'=' * 60}")
    for delta, init_l2, final_l2, path in improvements[:N]:
        print(f"  Δ={delta:+.4f} cm  (init={init_l2:.4f} → final={final_l2:.4f})  {path}")

    print(f"\n{'=' * 60}")
    print(f"  TOP {N} — Final WORSE than Initial (largest regression)")
    print(f"{'=' * 60}")
    for delta, init_l2, final_l2, path in improvements[-N:]:
        print(f"  Δ={delta:+.4f} cm  (init={init_l2:.4f} → final={final_l2:.4f})  {path}")

