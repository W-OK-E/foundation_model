import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

def analyze(root_dir):
    case_class_counts = {}
    case_classes_map = {}
    class_file_sets = defaultdict(set)
    class_object_counts = defaultdict(int)

    for case in sorted(os.listdir(root_dir)):
        case_path = os.path.join(root_dir, case)
        ann_dir = os.path.join(case_path, "ann")
        if not os.path.isdir(ann_dir):
            continue

        case_classes = set()
        for fn in sorted(os.listdir(ann_dir)):
            if not fn.endswith(".json"):
                continue
            json_path = os.path.join(ann_dir, fn)
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            for obj in data.get("objects", []):
                cls = obj.get("classTitle") or obj.get("label") or None
                if not cls:
                    continue
                case_classes.add(cls)
                class_file_sets[cls].add(json_path)
                class_object_counts[cls] += 1

        case_class_counts[case] = len(case_classes)
        case_classes_map[case] = sorted(case_classes)

    return case_class_counts, case_classes_map, class_file_sets, class_object_counts

def plot_and_save(root_dir, case_class_counts, class_file_sets, class_object_counts):
    out_dir = os.path.join(root_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Histogram: distribution of number of distinct classes per case
    counts = list(case_class_counts.values())
    plt.figure(figsize=(8,5))
    plt.hist(counts, bins=range(0, max(counts)+2), align='left', color='C0', edgecolor='k')
    plt.xlabel("Number of distinct classes in case")
    plt.ylabel("Number of cases")
    plt.title("Distribution of distinct class counts per case")
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "cases_class_count_hist.png")
    plt.savefig(hist_path)
    plt.close()

    # 2) Bar: how many annotation files contain each class
    classes = sorted(class_file_sets.keys(), key=lambda c: len(class_file_sets[c]), reverse=True)
    file_counts = [len(class_file_sets[c]) for c in classes]
    plt.figure(figsize=(max(6, len(classes)*0.5), 5))
    plt.bar(classes, file_counts, color='C1')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of annotation files containing class")
    plt.title("Per-class file frequency")
    plt.tight_layout()
    bar_files_path = os.path.join(out_dir, "class_file_frequency.png")
    plt.savefig(bar_files_path)
    plt.close()

    # 3) Bar: total object occurrences per class
    obj_counts = [class_object_counts[c] for c in classes]
    plt.figure(figsize=(max(6, len(classes)*0.5), 5))
    plt.bar(classes, obj_counts, color='C2')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Total object occurrences across all files")
    plt.title("Per-class object occurrences")
    plt.tight_layout()
    bar_objs_path = os.path.join(out_dir, "class_object_counts.png")
    plt.savefig(bar_objs_path)
    plt.close()

    return hist_path, bar_files_path, bar_objs_path

def main():
    parser = argparse.ArgumentParser(description="Count classes per case and plot distribution.")
    parser.add_argument("root", nargs="?", default="/home/asavari/Cataract1K/Images-and-Supervisely-Annotations",
                        help="Root folder containing case_*/ann folders")
    args = parser.parse_args()

    case_class_counts, case_classes_map, class_file_sets, class_object_counts = analyze(args.root)

    # Print per-case summary
    print("Case\t#classes\tclasses")
    for case, cnt in sorted(case_class_counts.items()):
        classes = ", ".join(case_classes_map.get(case, []))
        print(f"{case}\t{cnt}\t{classes}")

    # Save plots
    hist_path, bar_files_path, bar_objs_path = plot_and_save(args.root, case_class_counts, class_file_sets, class_object_counts)
    print("\nSaved plots:")
    print(hist_path)
    print(bar_files_path)
    print(bar_objs_path)

if __name__ == "__main__":
    main()