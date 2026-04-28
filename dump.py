import json
ev = json.load(open("reports/evaluation.json", encoding="utf-8"))
print("thr:", ev["optimal_threshold"])
print("ece:", ev["calibration"]["ece"])
print("default:", ev["default"])
print("tuned:", ev["tuned"])
print("shap_top:", ev["shap"]["top_features"])
print()
print("SEGMENTS:")
for s in ev["fairness"]["segments"]:
    print(f"{s[chr(39)+chr(115)+chr(101)+chr(103)+chr(109)+chr(101)+chr(110)+chr(116)+chr(39)]}")
