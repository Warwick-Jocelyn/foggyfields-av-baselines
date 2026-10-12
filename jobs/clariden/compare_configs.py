"""Text-level diff of two nerfstudio config.yml files, ignoring run-specific keys (paths, timestamps,
iteration counts, logging). Used to prove the 'faithful' run uses the published hyper-parameters."""
import re, sys, difflib
IGN = re.compile(r"^(timestamp|experiment_name|output_dir|vis|max_num_iterations|steps_per_save|steps_per_eval|relative_log|project_name|load_|data:|method_name)|^\s*- (/|networkhome|WMGDS|wang3_y|ETH|NVIDIA|capstor|scratch|cscs|yitiwang|splatad|outputs|data|Dataset|_smoke|'00)|^\s*(data|sequence|max_steps|use_valid_masks|lidar_points_frame):")
def load(p):
    out=[]
    for ln in open(p):
        ln=ln.rstrip("\n")
        if IGN.search(ln.strip()) or IGN.search(ln): continue
        out.append(ln)
    return out
a,b=sys.argv[1],sys.argv[2]; A,B=load(a),load(b)
d=[l for l in difflib.unified_diff(A,B,fromfile=a,tofile=b,lineterm="",n=0) if not l.startswith(("---","+++","@@"))]
print(f"{len(d)} differing lines after ignoring run-specific keys"); print("\n".join(d[:80]))
sys.exit(0 if not d else 1)
