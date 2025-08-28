# # # generate_graph_from_tree.py
# # import os
# # import shutil
# # import subprocess
# # import tempfile
# # from typing import Tuple, Dict, Any

# # import numpy as np
# # import pandas as pd

# # # Try to import graphviz (optional)
# # try:
# #     import graphviz as gv  # type: ignore
# #     _HAS_PYGRAPHVIZ = True
# # except Exception:
# #     _HAS_PYGRAPHVIZ = False

# # # Import utilities and impurity functions from your tree module
# # from tree.utils import one_hot_encoding, check_ifreal, entropy, gini_index, mse


# # def _format_counts(y_sub: pd.Series) -> str:
# #     """Return a short string of class counts like '0:3,1:2'"""
# #     if y_sub.size == 0:
# #         return ""
# #     counts = y_sub.value_counts().to_dict()
# #     return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


# # def _node_impurity_and_value(y_sub: pd.Series, is_regression: bool, criterion: str) -> Tuple[float, Any]:
# #     """
# #     Return (impurity_value, node_value)
# #     - For regression: impurity = mse, node_value = mean
# #     - For classification: impurity = entropy or gini depending on criterion, node_value = mode
# #     """
# #     if y_sub.size == 0:
# #         return 0.0, None
# #     if is_regression:
# #         imp = mse(y_sub)
# #         val = float(y_sub.mean())
# #         return float(imp), val
# #     else:
# #         if criterion == "information_gain":
# #             imp = entropy(y_sub)
# #         else:
# #             imp = gini_index(y_sub)
# #         # mode might be a series; pick first
# #         val = y_sub.mode().iloc[0]
# #         return float(imp), val


# # def generate_graph_from_tree(
# #     tree,
# #     X: pd.DataFrame,
# #     y: pd.Series,
# #     out_filename: str = "tree_graph.png",
# #     format: str = "png",
# #     criterion: str = "information_gain",
# #     rankdir: str = "TB",
# # ):
# #     """
# #     Generate a Graphviz graph for a fitted DecisionTree instance.

# #     Parameters
# #     ----------
# #     tree : DecisionTree
# #         Fitted DecisionTree object (must expose .root and ._processed_feature_names).
# #     X : pd.DataFrame
# #         Original training inputs (before one-hot encoding).
# #     y : pd.Series
# #         Training targets used for fit.
# #     out_filename : str
# #         Output file path (ends with .png or .pdf depending on `format`).
# #     format : str
# #         Graphviz output format, e.g., "png" or "pdf".
# #     criterion : str
# #         "information_gain" or "gini_index" - used for impurity calculation for classification.
# #     rankdir : str
# #         Graphviz rankdir (TB = top->bottom, LR = left->right).
# #     """

# #     if tree.root is None:
# #         raise ValueError("Provided tree has not been fitted (tree.root is None).")

# #     # Preprocess X the same way tree.fit did - use one_hot_encoding
# #     X_proc = one_hot_encoding(X)
# #     # Ensure all processed columns present (columns added by training)
# #     processed_cols = getattr(tree, "_processed_feature_names", X_proc.columns.tolist())
# #     for c in processed_cols:
# #         if c not in X_proc.columns:
# #             X_proc[c] = 0.0
# #     # Reorder
# #     X_proc = X_proc[processed_cols]

# #     # We'll maintain a mapping node_id -> info
# #     lines = []
# #     counter = {"id": 0}  # mutable counter

# #     def _new_id() -> int:
# #         i = counter["id"]
# #         counter["id"] += 1
# #         return i

# #     # We'll recursively traverse the tree and compute masks and statistics
# #     def _traverse(node: Dict, mask: pd.Series) -> int:
# #         """
# #         node: dictionary node from tree.root
# #         mask: boolean pandas Series aligned with X_proc indicating samples that reached this node
# #         returns: node_id
# #         """
# #         nid = _new_id()

# #         # compute stats for this node using mask
# #         y_sub = y[mask]
# #         n_samples = int(mask.sum())
# #         impurity_val, node_value = _node_impurity_and_value(y_sub, tree.is_regression, criterion)

# #         if node.get("is_leaf", False):
# #             # Leaf label
# #             if tree.is_regression:
# #                 label = f"Leaf\\nval={node_value:.6f}\\nMSE={impurity_val:.6f}\\nsamples={n_samples}"
# #             else:
# #                 class_counts = _format_counts(y_sub)
# #                 label = f"Leaf\\nval={node_value}\\nimp={impurity_val:.4f}\\nsamples={n_samples}\\n{class_counts}"
# #             # DOT node line
# #             lines.append((nid, label, True))
# #             return nid

# #         # internal node: compute left/right masks using processed X (one-hot)
# #         feat = node["feature"]
# #         thr = node["threshold"]

# #         # safe extraction of column - if missing use zeros
# #         col = X_proc[feat] if feat in X_proc.columns else pd.Series(0.0, index=X_proc.index)

# #         left_mask = mask & (col <= thr)
# #         right_mask = mask & (~(col <= thr))

# #         # compute child impurities for info gain
# #         left_imp, _ = _node_impurity_and_value(y[left_mask], tree.is_regression, criterion)
# #         right_imp, _ = _node_impurity_and_value(y[right_mask], tree.is_regression, criterion)
# #         # weighted impurity
# #         total = n_samples if n_samples > 0 else 1
# #         weighted_imp = (left_mask.sum() / total) * left_imp + (right_mask.sum() / total) * right_imp
# #         info_gain = float(impurity_val - weighted_imp)

# #         # format label: include feature, threshold, info_gain (or impurity change), samples, value
# #         if tree.is_regression:
# #             label = (
# #                 f"?({feat} <= {thr:.4f})\\n"
# #                 f"imp(MSE)={impurity_val:.6f}\\n"
# #                 f"reduction={info_gain:.6f}\\n"
# #                 f"samples={n_samples}\\nval={node_value:.6f}"
# #             )
# #         else:
# #             # classification
# #             class_counts = _format_counts(y_sub)
# #             label = (
# #                 f"?({feat} <= {thr:.4f})\\n"
# #                 f"imp={impurity_val:.4f}\\n"
# #                 f"gain={info_gain:.4f}\\n"
# #                 f"samples={n_samples}\\n"
# #                 f"val={node_value}\\n{class_counts}"
# #             )

# #         # append this node
# #         lines.append((nid, label, False))

# #         # traverse left and right
# #         left_id = _traverse(node["left"], left_mask)
# #         right_id = _traverse(node["right"], right_mask)

# #         # store edges as special lines with parent/child ids
# #         # We'll store edges in a separate list to preserve order
# #         edges.append((nid, left_id, "Y"))
# #         edges.append((nid, right_id, "N"))

# #         return nid

# #     # start traversal from root
# #     mask_all = pd.Series(True, index=X_proc.index)
# #     edges = []
# #     # lines will be appended in pre-order, edges collected separately
# #     _traverse(tree.root, mask_all)

# #     # Build DOT source
# #     dot_lines = [
# #         'digraph DecisionTree {',
# #         f'  graph [rankdir={rankdir}];',
# #         '  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", fontsize=10];',
# #     ]
# #     # add nodes
# #     for nid, label, is_leaf in lines:
# #         # escape double quotes in label if any
# #         label_safe = label.replace('"', '\\"')
# #         # leafs could have different fill color
# #         if is_leaf:
# #             dot_lines.append(f'  node{nid} [label="{label_safe}", fillcolor="#E8F5E9"];')
# #         else:
# #             dot_lines.append(f'  node{nid} [label="{label_safe}", fillcolor="#FFFFFF"];')

# #     # add edges
# #     for parent, child, edgelabel in edges:
# #         dot_lines.append(f'  node{parent} -> node{child} [label="{edgelabel}", fontsize=9];')

# #     dot_lines.append('}')
# #     dot_text = "\n".join(dot_lines)

# #     # If graphviz python package exists, use it
# #     base_out = out_filename
# #     if _HAS_PYGRAPHVIZ:
# #         try:
# #             src = gv.Source(dot_text)
# #             # Graphviz Source.render will append extension; supply filename without ext if you want that
# #             # To allow user-specified filename with ext, handle:
# #             fname_no_ext, ext = os.path.splitext(base_out)
# #             if ext:
# #                 # graphviz.render will add another ext; instead use gv.Source.pipe
# #                 out_bytes = src.pipe(format=format)
# #                 with open(base_out, "wb") as f:
# #                     f.write(out_bytes)
# #                 print(f"Wrote graph to {base_out} using graphviz python package.")
# #                 return base_out
# #             else:
# #                 outpath = src.render(filename=fname_no_ext, format=format, cleanup=True)
# #                 print(f"Wrote graph to {outpath} using graphviz python package.")
# #                 return outpath
# #         except Exception as e:
# #             print("graphviz python package failed to render:", e)
# #             # fallthrough to system dot

# #     # Try system dot binary
# #     dot_bin = shutil.which("dot")
# #     if dot_bin:
# #         try:
# #             # write dot to temp file
# #             with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".dot") as tf:
# #                 tf.write(dot_text)
# #                 tmp_dot = tf.name
# #             # ensure desired output filename ext matches format
# #             if not base_out.endswith(f".{format}"):
# #                 base_out = f"{base_out}.{format}"
# #             subprocess.run([dot_bin, "-T", format, tmp_dot, "-o", base_out], check=True)
# #             try:
# #                 os.remove(tmp_dot)
# #             except Exception:
# #                 pass
# #             print(f"Wrote graph to {base_out} using system dot binary.")
# #             return base_out
# #         except Exception as e:
# #             print("system dot binary failed:", e)

# #     # If neither method available, save DOT to file and inform user
# #     dot_fname = out_filename if out_filename.endswith(".dot") else f"{out_filename}.dot"
# #     with open(dot_fname, "w") as f:
# #         f.write(dot_text)
# #     print(f"Could not render graph: saved DOT to {dot_fname}. Please install graphviz to render PNG/PDF.")
# #     return dot_fname


# # # ---------------- Example usage ----------------
# # if __name__ == "__main__":
# #     # Example: quick demo using your DecisionTree (if available) and random data
# #     # This block will only run when you execute this script directly.
# #     # If you prefer to call the function from usage.py, import generate_graph_from_tree and call it.
# #     try:
# #         from tree.base import DecisionTree
# #         # create toy data (you can replace with your actual X,y and tree)
# #         N = 30
# #         P = 5
# #         X = pd.DataFrame(np.random.randn(N, P))
# #         y_reg = pd.Series(np.random.randn(N))
# #         y_clf = pd.Series(np.random.randint(3, size=N), dtype="category")

# #         # regression example
# #         dt = DecisionTree(criterion="information_gain", max_depth=4)
# #         dt.fit(X, y_reg)
# #         generate_graph_from_tree(dt, X, y_reg, out_filename="demo_tree_reg.png", format="png", criterion="information_gain")

# #         # classification example
# #         dt2 = DecisionTree(criterion="information_gain", max_depth=4)
# #         dt2.fit(X, y_clf)
# #         generate_graph_from_tree(dt2, X, y_clf, out_filename="demo_tree_clf.png", format="png", criterion="information_gain")

# #     except Exception as e:
# #         print("Demo failed because:", e)
# #         print("Use this module from your usage.py by calling generate_graph_from_tree(tree, X, y, out_filename='tree.png').")




# # generate_graph_from_tree.py
# import os
# import shutil
# import subprocess
# import tempfile
# from typing import Tuple, Dict, Any, List

# import numpy as np
# import pandas as pd

# # Try to import graphviz (optional)
# try:
#     import graphviz as gv  # type: ignore
#     _HAS_PYGRAPHVIZ = True
# except Exception:
#     _HAS_PYGRAPHVIZ = False

# # Import utilities and impurity functions from your tree module
# from tree.utils import one_hot_encoding, check_ifreal, entropy, gini_index, mse


# def _format_counts(y_sub: pd.Series) -> str:
#     """Return a short string of class counts like '0:3,1:2'"""
#     if y_sub.size == 0:
#         return ""
#     counts = y_sub.value_counts().to_dict()
#     # produce "class:count" sorted by class repr
#     parts = [f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))]
#     s = ", ".join(parts)
#     # truncate long counts to keep labels readable
#     if len(s) > 40:
#         return s[:37] + "..."
#     return s


# def _node_impurity_and_value(y_sub: pd.Series, is_regression: bool, criterion: str) -> Tuple[float, Any]:
#     """
#     Return (impurity_value, node_value)
#     - For regression: impurity = mse, node_value = mean
#     - For classification: impurity = entropy or gini depending on criterion, node_value = mode
#     """
#     if y_sub.size == 0:
#         return 0.0, None
#     if is_regression:
#         imp = mse(y_sub)
#         val = float(y_sub.mean())
#         return float(imp), val
#     else:
#         if criterion == "information_gain":
#             imp = entropy(y_sub)
#         else:
#             imp = gini_index(y_sub)
#         # mode might be a series; pick first
#         val = y_sub.mode().iloc[0]
#         return float(imp), val


# def _pretty_feature_name(processed_col: str, original_cols: List[str]) -> str:
#     """
#     Try to map one-hot processed column names back to a readable feature representation.
#     Examples:
#       - "color_red" -> "color==red"
#       - "3_2" -> if "3" in original_cols -> "3==2"
#     If cannot map, return the processed_col unchanged.
#     """
#     # If contains exactly one '_' assume split into orig and val
#     if "_" in processed_col:
#         left, right = processed_col.split("_", 1)
#         # if left matches an original column name (as string)
#         if str(left) in [str(c) for c in original_cols]:
#             return f"{left}=={right}"
#         # sometimes pandas get_dummies uses format 'col_val' where col itself has underscores;
#         # try to find a matching prefix in original_cols
#         for orig in original_cols:
#             orig_str = str(orig)
#             if processed_col.startswith(orig_str + "_"):
#                 suffix = processed_col[len(orig_str) + 1 :]
#                 return f"{orig_str}=={suffix}"
#     # fallback: return as-is
#     return processed_col


# def generate_graph_from_tree(
#     tree,
#     X: pd.DataFrame,
#     y: pd.Series,
#     out_filename: str = "tree_graph.svg",
#     format: str = "svg",
#     criterion: str = "information_gain",
#     rankdir: str = "TB",
#     node_font: str = "Helvetica",
#     node_fontsize: int = 10,
# ):
#     """
#     Generate a Graphviz graph for a fitted DecisionTree instance.

#     Parameters
#     ----------
#     tree : DecisionTree
#         Fitted DecisionTree object (must expose .root and ._processed_feature_names).
#     X : pd.DataFrame
#         Original training inputs (before one-hot encoding).
#     y : pd.Series
#         Training targets used for fit.
#     out_filename : str
#         Output file path (e.g. 'tree.svg' or 'tree.pdf'). If ext present, it will be respected.
#     format : str
#         Graphviz output format, e.g., "svg", "pdf", "png".
#     criterion : str
#         "information_gain" or "gini_index" - used for impurity calculation for classification.
#     rankdir : str
#         Graphviz rankdir (TB = top->bottom, LR = left->right).
#     node_font : str
#         Font family for node text.
#     node_fontsize : int
#         Font size for nodes.
#     """
#     if tree.root is None:
#         raise ValueError("Provided tree has not been fitted (tree.root is None).")

#     # Preprocess X the same way tree.fit did - use one_hot_encoding
#     X_proc = one_hot_encoding(X)

#     # If tree stores processed feature names, use them; otherwise use X_proc columns
#     processed_cols = getattr(tree, "_processed_feature_names", X_proc.columns.tolist())
#     # If tree stored original column list, use it to improve feature label readability
#     original_cols = getattr(tree, "_original_X_columns", list(X.columns))

#     # Ensure all processed columns present (columns added by training)
#     for c in processed_cols:
#         if c not in X_proc.columns:
#             X_proc[c] = 0.0
#     # Reorder
#     X_proc = X_proc[processed_cols]

#     # We'll maintain a mapping node_id -> info
#     lines = []  # list of tuples (nid, label, is_leaf)
#     edges = []  # list of tuples (parent, child, edgelabel)
#     counter = {"id": 0}  # mutable counter

#     def _new_id() -> int:
#         i = counter["id"]
#         counter["id"] += 1
#         return i

#     # We'll recursively traverse the tree and compute masks and statistics
#     def _traverse(node: Dict[str, Any], mask: pd.Series) -> int:
#         """
#         node: dictionary node from tree.root
#         mask: boolean pandas Series aligned with X_proc indicating samples that reached this node
#         returns: node_id
#         """
#         nid = _new_id()

#         # compute stats for this node using mask
#         y_sub = y[mask]
#         n_samples = int(mask.sum())
#         impurity_val, node_value = _node_impurity_and_value(y_sub, tree.is_regression, criterion)

#         if node.get("is_leaf", False):
#             # Leaf label (short)
#             if tree.is_regression:
#                 label = (
#                     f"Leaf\\nval={node_value:.4f}\\nMSE={impurity_val:.4f}\\nsamples={n_samples}"
#                 )
#             else:
#                 class_counts = _format_counts(y_sub)
#                 label = (
#                     f"Leaf\\nval={node_value}\\nimp={impurity_val:.3f}\\nsamples={n_samples}\\n{class_counts}"
#                 )
#             lines.append((nid, label, True))
#             return nid

#         # internal node: compute left/right masks using processed X (one-hot)
#         feat = node["feature"]
#         thr = node["threshold"]

#         # Try to produce a pretty feature label using mapping back to original when possible
#         pretty_feat = _pretty_feature_name(str(feat), original_cols)

#         # safe extraction of column - if missing use zeros
#         col = X_proc[feat] if feat in X_proc.columns else pd.Series(0.0, index=X_proc.index)

#         left_mask = mask & (col <= thr)
#         right_mask = mask & (~(col <= thr))

#         # compute child impurities for info gain
#         left_imp, _ = _node_impurity_and_value(y[left_mask], tree.is_regression, criterion)
#         right_imp, _ = _node_impurity_and_value(y[right_mask], tree.is_regression, criterion)

#         # weighted impurity
#         total = n_samples if n_samples > 0 else 1
#         weighted_imp = (left_mask.sum() / total) * left_imp + (right_mask.sum() / total) * right_imp
#         info_gain = float(impurity_val - weighted_imp)

#         # format label: concise multiline
#         if tree.is_regression:
#             label = (
#                 f"{pretty_feat} <= {thr:.4f}\\n"
#                 f"imp(MSE)={impurity_val:.4f}, red={info_gain:.4f}\\n"
#                 f"samples={n_samples}, val={node_value:.4f}"
#             )
#         else:
#             class_counts = _format_counts(y_sub)
#             label = (
#                 f"{pretty_feat} <= {thr:.4f}\\n"
#                 f"imp={impurity_val:.3f}, gain={info_gain:.3f}\\n"
#                 f"samples={n_samples}, val={node_value}\\n{class_counts}"
#             )

#         # append this node
#         lines.append((nid, label, False))

#         # traverse left and right
#         left_id = _traverse(node["left"], left_mask)
#         right_id = _traverse(node["right"], right_mask)

#         edges.append((nid, left_id, "Y"))
#         edges.append((nid, right_id, "N"))

#         return nid

#     # start traversal from root with mask all True (aligned with X_proc)
#     mask_all = pd.Series(True, index=X_proc.index)
#     _traverse(tree.root, mask_all)

#     # Build DOT source with improved layout attributes
#     dot_lines = [
#         'digraph DecisionTree {',
#         f'  graph [rankdir={rankdir}, dpi=300, nodesep=0.6, ranksep=0.9];',
#         # node defaults
#         '  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF",',
#         f'        fontname="{node_font}", fontsize={node_fontsize}, margin="0.08,0.05"];',
#         f'  edge [fontname="{node_font}", fontsize={max(8, node_fontsize-1)}];',
#     ]

#     # add nodes
#     for nid, label, is_leaf in lines:
#         label_safe = label.replace('"', '\\"')
#         if is_leaf:
#             # leaf greenish fill
#             dot_lines.append(f'  node{nid} [label="{label_safe}", fillcolor="#E8F5E9"];')
#         else:
#             dot_lines.append(f'  node{nid} [label="{label_safe}", fillcolor="#FFFFFF"];')

#     # add edges
#     for parent, child, edgelabel in edges:
#         dot_lines.append(f'  node{parent} -> node{child} [label="{edgelabel}", fontsize=9];')

#     dot_lines.append('}')
#     dot_text = "\n".join(dot_lines)

#     # Ensure output filename has appropriate extension when needed
#     base_out = out_filename
#     base_name, ext = os.path.splitext(base_out)
#     if ext:
#         # If ext exists, honor it but make sure format matches or use requested format
#         target_path = base_out
#     else:
#         # append format extension if not provided
#         target_path = f"{base_out}.{format}"

#     # Try python-graphviz first (vector output is preferred)
#     if _HAS_PYGRAPHVIZ:
#         try:
#             src = gv.Source(dot_text)
#             # If user asked for svg/pdf or any vector, use pipe to get bytes and write directly
#             out_bytes = src.pipe(format=format)
#             with open(target_path, "wb") as f:
#                 f.write(out_bytes)
#             print(f"Wrote graph to {target_path} using graphviz python package.")
#             return target_path
#         except Exception as e:
#             print("graphviz python package failed to render:", e)
#             # fallback to system dot

#     # Try system dot binary if available
#     dot_bin = shutil.which("dot")
#     if dot_bin:
#         try:
#             with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".dot") as tf:
#                 tf.write(dot_text)
#                 tmp_dot = tf.name
#             # prepare target path (ensure extension)
#             if not target_path.endswith(f".{format}"):
#                 target_path = f"{base_name}.{format}"
#             subprocess.run([dot_bin, "-T", format, tmp_dot, "-o", target_path], check=True)
#             try:
#                 os.remove(tmp_dot)
#             except Exception:
#                 pass
#             print(f"Wrote graph to {target_path} using system dot binary.")
#             return target_path
#         except Exception as e:
#             print("system dot binary failed:", e)

#     # If we cannot render, save DOT file for manual rendering later
#     dot_fname = target_path if target_path.endswith(".dot") else f"{base_name}.dot"
#     with open(dot_fname, "w") as f:
#         f.write(dot_text)
#     print(f"Could not render graph automatically. Saved DOT to {dot_fname}. Install Graphviz to render PNG/PDF/SVG.")
#     return dot_fname


# # ---------------- Example usage ----------------
# if __name__ == "__main__":
#     # Demo when run directly
#     try:
#         from tree.base import DecisionTree
#         # create toy data (you can replace with your actual X,y and tree)
#         N = 30
#         P = 5
#         X = pd.DataFrame(np.random.randn(N, P))
#         y_reg = pd.Series(np.random.randn(N))
#         y_clf = pd.Series(np.random.randint(3, size=N), dtype="category")

#         # regression example - produce SVG (vector)
#         dt = DecisionTree(criterion="information_gain", max_depth=4)
#         dt.fit(X, y_reg)
#         out = generate_graph_from_tree(dt, X, y_reg, out_filename="demo_tree_reg.svg", format="svg", criterion="information_gain")
#         print("Saved demo regression tree to:", out)

#         # classification example - produce SVG
#         dt2 = DecisionTree(criterion="information_gain", max_depth=4)
#         dt2.fit(X, y_clf)
#         out2 = generate_graph_from_tree(dt2, X, y_clf, out_filename="demo_tree_clf.svg", format="svg", criterion="information_gain")
#         print("Saved demo classification tree to:", out2)

#     except Exception as e:
#         print("Demo failed because:", e)
#         print("Use this module from your usage.py by calling generate_graph_from_tree(tree, X, y, out_filename='tree.svg').")



# generate_graph_from_tree.py
import os
import shutil
import subprocess
import tempfile
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd

# Try to import graphviz (optional)
try:
    import graphviz as gv  # type: ignore
    _HAS_PYGRAPHVIZ = True
except Exception:
    _HAS_PYGRAPHVIZ = False

# Try to import cairosvg (optional) for high-quality SVG -> PNG conversion
try:
    import cairosvg  # type: ignore
    _HAS_CAIROSVG = True
except Exception:
    _HAS_CAIROSVG = False

# Import utilities from your tree module
from tree.utils import one_hot_encoding, check_ifreal, entropy, gini_index, mse


def _format_counts(y_sub: pd.Series) -> str:
    if y_sub.size == 0:
        return ""
    counts = y_sub.value_counts().to_dict()
    parts = [f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))]
    s = ", ".join(parts)
    if len(s) > 40:
        return s[:37] + "..."
    return s


def _node_impurity_and_value(y_sub: pd.Series, is_regression: bool, criterion: str) -> Tuple[float, Any]:
    if y_sub.size == 0:
        return 0.0, None
    if is_regression:
        imp = mse(y_sub)
        val = float(y_sub.mean())
        return float(imp), val
    else:
        if criterion == "information_gain":
            imp = entropy(y_sub)
        else:
            imp = gini_index(y_sub)
        val = y_sub.mode().iloc[0]
        return float(imp), val


def _pretty_feature_name(processed_col: str, original_cols: List[str]) -> str:
    if "_" in processed_col:
        left, right = processed_col.split("_", 1)
        if str(left) in [str(c) for c in original_cols]:
            return f"{left}=={right}"
        for orig in original_cols:
            orig_str = str(orig)
            if processed_col.startswith(orig_str + "_"):
                suffix = processed_col[len(orig_str) + 1 :]
                return f"{orig_str}=={suffix}"
    return processed_col


def _build_dot_text(lines: List[Tuple[int, str, bool]], edges: List[Tuple[int, int, str]],
                    rankdir: str = "TB", node_font: str = "Helvetica", node_fontsize: int = 10) -> str:
    dot_lines = [
        'digraph DecisionTree {',
        f'  graph [rankdir={rankdir}, dpi=300, nodesep=0.6, ranksep=0.9];',
        '  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF",',
        f'        fontname="{node_font}", fontsize={node_fontsize}, margin="0.08,0.05"];',
        f'  edge [fontname="{node_font}", fontsize={max(8, node_fontsize - 1)}];',
    ]
    for nid, label, is_leaf in lines:
        label_safe = label.replace('"', '\\"')
        if is_leaf:
            dot_lines.append(f'  node{nid} [label="{label_safe}", fillcolor="#E8F5E9"];')
        else:
            dot_lines.append(f'  node{nid} [label="{label_safe}", fillcolor="#FFFFFF"];')
    for parent, child, edgelabel in edges:
        dot_lines.append(f'  node{parent} -> node{child} [label="{edgelabel}", fontsize=9];')
    dot_lines.append('}')
    return "\n".join(dot_lines)


def generate_graph_from_tree_all_formats(
    tree,
    X: pd.DataFrame,
    y: pd.Series,
    out_basename: str = "tree_graph",
    formats: List[str] = None,
    criterion: str = "information_gain",
    rankdir: str = "TB",
    node_font: str = "Helvetica",
    node_fontsize: int = 10,
) -> Dict[str, str]:
    """
    Generate graph files for the given fitted tree.

    - out_basename: base path without extension (e.g. "tree_case1" or "./out/tree_case1")
    - formats: list of desired formats, subset of ['svg','pdf','png']. Default ['svg','pdf','png']
    Returns dict mapping format -> output path (for formats that succeeded).
    """
    if formats is None:
        formats = ["svg", "pdf", "png"]

    if tree.root is None:
        raise ValueError("Provided tree has not been fitted (tree.root is None).")

    # Preprocess X same as training
    X_proc = one_hot_encoding(X)
    processed_cols = getattr(tree, "_processed_feature_names", X_proc.columns.tolist())
    original_cols = getattr(tree, "_original_X_columns", list(X.columns))
    for c in processed_cols:
        if c not in X_proc.columns:
            X_proc[c] = 0.0
    X_proc = X_proc[processed_cols]

    # Collect nodes and edges by traversing tree with masks
    lines = []
    edges = []
    counter = {"id": 0}

    def new_id():
        i = counter["id"]
        counter["id"] += 1
        return i

    def traverse(node: dict, mask: pd.Series):
        nid = new_id()
        y_sub = y[mask]
        n_samples = int(mask.sum())
        imp, val = _node_impurity_and_value(y_sub, tree.is_regression, criterion)
        if node.get("is_leaf", False):
            if tree.is_regression:
                label = f"Leaf\\nval={val:.4f}\\nMSE={imp:.4f}\\nsamples={n_samples}"
            else:
                class_counts = _format_counts(y_sub)
                label = f"Leaf\\nval={val}\\nimp={imp:.3f}\\nsamples={n_samples}\\n{class_counts}"
            lines.append((nid, label, True))
            return nid
        feat = node["feature"]
        thr = node["threshold"]
        pretty_feat = _pretty_feature_name(str(feat), original_cols)
        col = X_proc[feat] if feat in X_proc.columns else pd.Series(0.0, index=X_proc.index)
        left_mask = mask & (col <= thr)
        right_mask = mask & (~(col <= thr))
        left_imp, _ = _node_impurity_and_value(y[left_mask], tree.is_regression, criterion)
        right_imp, _ = _node_impurity_and_value(y[right_mask], tree.is_regression, criterion)
        total = n_samples if n_samples > 0 else 1
        weighted_imp = (left_mask.sum() / total) * left_imp + (right_mask.sum() / total) * right_imp
        gain = float(imp - weighted_imp)
        if tree.is_regression:
            label = (f"{pretty_feat} <= {thr:.4f}\\nimp(MSE)={imp:.4f}, red={gain:.4f}\\n"
                     f"samples={n_samples}, val={val:.4f}")
        else:
            class_counts = _format_counts(y_sub)
            label = (f"{pretty_feat} <= {thr:.4f}\\nimp={imp:.3f}, gain={gain:.3f}\\n"
                     f"samples={n_samples}, val={val}\\n{class_counts}")
        lines.append((nid, label, False))
        left_id = traverse(node["left"], left_mask)
        right_id = traverse(node["right"], right_mask)
        edges.append((nid, left_id, "Y"))
        edges.append((nid, right_id, "N"))
        return nid

    mask_all = pd.Series(True, index=X_proc.index)
    traverse(tree.root, mask_all)

    dot_text = _build_dot_text(lines, edges, rankdir=rankdir, node_font=node_font, node_fontsize=node_fontsize)

    results = {}
    # Preferred: produce SVG and PDF from python-graphviz (vector) if available
    svg_bytes = None
    pdf_bytes = None
    dot_bin = shutil.which("dot")

    # 1) Try python graphviz to produce svg/pdf bytes
    if _HAS_PYGRAPHVIZ:
        try:
            src = gv.Source(dot_text)
            if "svg" in formats:
                svg_bytes = src.pipe(format="svg")
                svg_path = f"{out_basename}.svg"
                with open(svg_path, "wb") as f:
                    f.write(svg_bytes)
                results["svg"] = svg_path
            if "pdf" in formats:
                pdf_bytes = src.pipe(format="pdf")
                pdf_path = f"{out_basename}.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)
                results["pdf"] = pdf_path
        except Exception as e:
            print("graphviz python package failed to render vector outputs:", e)
            svg_bytes = None
            pdf_bytes = None

    # 2) If python-graphviz failed to produce SVG/PDF and system dot exists, use it to make SVG/PDF
    if (("svg" in formats and "svg" not in results) or ("pdf" in formats and "pdf" not in results)) and dot_bin:
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".dot") as tf:
                tf.write(dot_text)
                tmpdot = tf.name
            if "svg" in formats and "svg" not in results:
                svg_path = f"{out_basename}.svg"
                subprocess.run([dot_bin, "-Tsvg", tmpdot, "-o", svg_path], check=True)
                results["svg"] = svg_path
            if "pdf" in formats and "pdf" not in results:
                pdf_path = f"{out_basename}.pdf"
                subprocess.run([dot_bin, "-Tpdf", tmpdot, "-o", pdf_path], check=True)
                results["pdf"] = pdf_path
            try:
                os.remove(tmpdot)
            except Exception:
                pass
        except Exception as e:
            print("system dot failed to render SVG/PDF:", e)

    # 3) PNG: best approach is to convert SVG produced above to PNG using cairosvg (high quality).
    if "png" in formats:
        png_path = f"{out_basename}.png"
        png_written = False

        # if we have svg_bytes (from python-graphviz) or svg file produced, use cairosvg
        if _HAS_CAIROSVG:
            try:
                # if we have svg bytes in memory use them, else read svg file
                if svg_bytes is not None:
                    cairosvg.svg2png(bytestring=svg_bytes, write_to=png_path, dpi=300)
                else:
                    svg_file = results.get("svg", f"{out_basename}.svg")
                    if os.path.exists(svg_file):
                        cairosvg.svg2png(url=svg_file, write_to=png_path, dpi=300)
                    else:
                        raise RuntimeError("No SVG available to convert to PNG.")
                results["png"] = png_path
                png_written = True
            except Exception as e:
                print("cairosvg conversion to PNG failed:", e)
                png_written = False

        # fallback: use ImageMagick 'magick' (Windows) or 'convert' (older) to rasterize PDF or SVG
        if not png_written:
            magick_bin = shutil.which("magick") or shutil.which("convert")
            if magick_bin:
                try:
                    # prefer converting PDF -> PNG if pdf exists (better preserved text)
                    if "pdf" in results and os.path.exists(results["pdf"]):
                        tmp_pdf = results["pdf"]
                        subprocess.run([magick_bin, "-density", "300", tmp_pdf, "-quality", "95", png_path], check=True)
                        results["png"] = png_path
                        png_written = True
                    elif "svg" in results and os.path.exists(results["svg"]):
                        tmp_svg = results["svg"]
                        subprocess.run([magick_bin, "-density", "300", tmp_svg, "-quality", "95", png_path], check=True)
                        results["png"] = png_path
                        png_written = True
                except Exception as e:
                    print("ImageMagick conversion to PNG failed:", e)
                    png_written = False

        # final fallback: ask dot to output PNG directly
        if not png_written and dot_bin:
            try:
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".dot") as tf:
                    tf.write(dot_text)
                    tmpdot = tf.name
                subprocess.run([dot_bin, "-Tpng", tmpdot, "-o", png_path], check=True)
                try:
                    os.remove(tmpdot)
                except Exception:
                    pass
                results["png"] = png_path
                png_written = True
            except Exception as e:
                print("dot failed to produce PNG directly:", e)
                png_written = False

        if not png_written:
            print("Could not produce PNG automatically. Ensure 'cairosvg' or ImageMagick 'magick' is installed, or use dot to render PNG manually.")

    # final: report what we have
    print("Generated files:", results)
    return results


# Example main for quick test
if __name__ == "__main__":
    try:
        from tree.base import DecisionTree
        N = 30
        P = 5
        X = pd.DataFrame(np.random.randn(N, P))
        y_reg = pd.Series(np.random.randn(N))
        dt = DecisionTree(criterion="information_gain", max_depth=4)
        dt.fit(X, y_reg)
        res = generate_graph_from_tree_all_formats(dt, X, y_reg, out_basename="tree_case_test")
        print(res)
    except Exception as ex:
        print("Demo failed:", ex)
