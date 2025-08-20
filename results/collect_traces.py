import os
import shutil

def collect_traces(src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)

    for instance in os.listdir(src_root):
        instance_dir = os.path.join(src_root, instance)
        if not os.path.isdir(instance_dir):
            continue  # skip non-directories

        # 1. Look for trace.0.et
        et_file = os.path.join(instance_dir, "trace.0.et")
        if os.path.isfile(et_file):
            dst_et = os.path.join(dst_root, f"{instance}.et")
            shutil.copy2(et_file, dst_et)
            print(f"Copied {et_file} -> {dst_et}")

        # 2. Look for kineto_trace_rank0 folder
        kineto_dir = os.path.join(instance_dir, "kineto_trace_rank0")
        if os.path.isdir(kineto_dir):
            json_files = [f for f in os.listdir(kineto_dir) if f.endswith(".json")]
            if len(json_files) == 1:
                src_json = os.path.join(kineto_dir, json_files[0])
                dst_json = os.path.join(dst_root, f"{instance}.json")
                shutil.copy2(src_json, dst_json)
                print(f"Copied {src_json} -> {dst_json}")
            elif len(json_files) > 1:
                print(f"⚠️ Multiple JSON files found in {kineto_dir}, skipping")
            else:
                print(f"No JSON file found in {kineto_dir}")

if __name__ == "__main__":
    src_root = "./raw"   # change this
    dst_root = "./processed"   # change this
    collect_traces(src_root, dst_root)
